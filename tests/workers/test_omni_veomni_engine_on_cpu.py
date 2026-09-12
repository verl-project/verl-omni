# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared VeOmni dispatch, with CPU tensors and a stub distributed base."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from tensordict import TensorDict

_ROOT = Path(__file__).resolve().parents[2]
_QWEN_BACKEND = "verl_omni.pipelines.qwen3_omni.veomni"


def _load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, _ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Keep the real registry and model hooks, isolating package __init__ imports
# that would otherwise load all rollout engines during CPU test collection.
with patch.dict(sys.modules, {"verl_omni.workers.config": SimpleNamespace(DiffusionModelConfig=object)}):
    _model_base = _load_source("omni_model_base_test", "verl_omni/pipelines/model_base.py")
OmniModelBase = _model_base.OmniModelBase
qwen_backend = _load_source(_QWEN_BACKEND, "verl_omni/pipelines/qwen3_omni/veomni.py")
with patch.dict(sys.modules, {"verl_omni.pipelines.model_base": _model_base}):
    _qwen_adapter = _load_source(
        "verl_omni.pipelines.qwen3_omni.thinker_training_adapter",
        "verl_omni/pipelines/qwen3_omni/thinker_training_adapter.py",
    )
Qwen3OmniThinkerAdapter = _qwen_adapter.Qwen3OmniThinkerAdapter


class _BaseEngine:
    def _apply_veomni_input_transforms(self, model_inputs, micro_batch):
        model_inputs["base_transform_called"] = True

    def prepare_model_inputs(self, micro_batch):
        inputs = {"input_ids": micro_batch["input_ids"].values().unsqueeze(0)}
        self._apply_veomni_input_transforms(inputs, micro_batch)
        return inputs, {"output_metadata": True}

    def _get_model_config_path(self):
        return self.model_config.local_hf_config_path

    def _build_optimizer(self, module):
        return [p for p in module.parameters() if p.requires_grad]


_registry = MagicMock()
_registry.register.return_value = lambda cls: cls
_path = Path(__file__).resolve().parents[2] / "verl_omni/workers/engine/veomni/omni_impl.py"
_spec = importlib.util.spec_from_file_location("omni_veomni_test_engine", _path)
_engine_module = importlib.util.module_from_spec(_spec)
with patch.dict(
    sys.modules,
    {
        "verl_omni.pipelines.model_base": _model_base,
        "verl.workers.engine.base": SimpleNamespace(EngineRegistry=_registry),
        "verl.workers.engine.veomni.transformer_impl": SimpleNamespace(VeOmniEngineWithLMHead=_BaseEngine),
    },
):
    _spec.loader.exec_module(_engine_module)
OmniVeOmniEngine = _engine_module.OmniVeOmniEngine


@pytest.fixture
def backend_setup():
    handlers = {}
    with (
        patch.object(qwen_backend, "patch_veomni_causal_mask_kwargs") as compat,
        patch.dict(
            sys.modules,
            {
                "verl.workers.engine.veomni.utils": SimpleNamespace(MOE_PARAM_HANDERS=handlers),
                _QWEN_BACKEND: qwen_backend,
            },
        ),
    ):
        yield handlers, compat


def _engine(architecture="Qwen3OmniMoeForConditionalGeneration"):
    engine = OmniVeOmniEngine()
    engine.model_config = SimpleNamespace(
        architecture=architecture,
        hf_config=SimpleNamespace(
            model_type="qwen3_omni_moe",
            thinker_config=SimpleNamespace(image_token_id=151655, video_token_id=151656, audio_token_id=151675),
        ),
        model_stage="thinker",
        use_remove_padding=True,
        lora_rank=0,
        lora={},
        local_hf_config_path="checkpoint",
    )
    engine.engine_config = SimpleNamespace(ulysses_parallel_size=1)
    return engine


def _batch():
    return TensorDict(
        {
            "input_ids": torch.nested.as_nested_tensor([torch.tensor([1, 151655, 2, 151655])], layout=torch.jagged),
            "response_mask": torch.nested.as_nested_tensor([torch.ones(2, dtype=torch.bool)], layout=torch.jagged),
        },
        batch_size=1,
    )


def test_registers_one_shared_omni_engine():
    _registry.register.assert_called_once_with(model_type="omni_model", backend="veomni", device="cuda")


def test_resolves_existing_adapter_and_sets_up_before_loading(backend_setup):
    handlers, compat = backend_setup
    engine = _engine()
    assert engine._get_model_config_path() == "checkpoint"
    assert engine.model_adapter_cls is Qwen3OmniThinkerAdapter
    assert handlers["qwen3_omni_moe"] is qwen_backend.map_qwen3_omni_moe_param
    compat.assert_called_once_with()


def test_engine_masks_the_prompt_image_and_preserves_response_ids(backend_setup):
    engine = _engine()
    engine._get_model_config_path()
    batch = _batch()
    inputs = {"input_ids": batch["input_ids"].values().unsqueeze(0), "pixel_values": torch.ones(2, 3)}
    expected = inputs["input_ids"].clone()
    engine._apply_veomni_input_transforms(inputs, batch)
    torch.testing.assert_close(inputs["input_ids"], expected)
    torch.testing.assert_close(inputs["image_mask"], torch.tensor([[False, True, False, False]]))
    assert inputs["base_transform_called"]


@pytest.mark.parametrize("key", ["pixel_values_videos", "input_features"])
def test_engine_rejects_unsupported_features(key, backend_setup):
    engine = _engine()
    engine._get_model_config_path()
    with pytest.raises(NotImplementedError, match="inputs yet"):
        engine._apply_veomni_input_transforms({"input_ids": torch.tensor([[1]]), key: torch.ones(1)}, _batch())


def test_engine_requires_prompt_boundaries(backend_setup):
    engine = _engine()
    engine._get_model_config_path()
    with pytest.raises(ValueError, match="response_mask"):
        engine._apply_veomni_input_transforms({"input_ids": torch.tensor([[1]])}, {})


@pytest.mark.parametrize(
    ("owner", "key", "value"),
    [
        ("model_config", "model_stage", "talker"),
        ("model_config", "lora_rank", 8),
        ("model_config", "lora", {"rank": 8}),
        ("model_config", "use_remove_padding", False),
        ("engine_config", "ulysses_parallel_size", 2),
    ],
)
def test_engine_rejects_unsupported_config_before_model_loading(owner, key, value, backend_setup):
    engine = _engine()
    setattr(getattr(engine, owner), key, value)
    with pytest.raises(NotImplementedError):
        engine._get_model_config_path()
    backend_setup[1].assert_not_called()


def test_optimizer_excludes_modality_towers(backend_setup):
    model = torch.nn.Module()
    model.thinker = torch.nn.Module()
    model.thinker.visual = torch.nn.Linear(2, 2)
    model.thinker.audio_tower = torch.nn.Linear(2, 2)
    model.thinker.model = torch.nn.Linear(2, 2)
    engine = _engine()
    engine._get_model_config_path()
    params = engine._build_optimizer(model)
    assert {id(p) for p in params} == {id(p) for p in model.thinker.model.parameters()}


def test_another_architecture_uses_same_engine_and_replay_hooks(monkeypatch):
    calls = []

    class AnotherAdapter(OmniModelBase):
        @classmethod
        def setup_veomni(cls, model_config, engine_config):
            calls.append("setup")

        @classmethod
        def prepare_veomni_inputs(cls, model_inputs, micro_batch, model_config):
            assert model_inputs["base_transform_called"]
            calls.append("backend_inputs")
            return {"input_ids": model_inputs["input_ids"], "conditioning": 42}

        @classmethod
        def prepare_model_inputs(cls, model_inputs, micro_batch, model_config):
            calls.append("replay_inputs")
            return {**model_inputs, "trajectory": micro_batch["response_mask"]}

        @classmethod
        def configure_veomni_trainable_params(cls, module, model_config):
            calls.append("trainability")
            module.bias.requires_grad_(False)

    monkeypatch.setitem(OmniModelBase._registry, ("AnotherArchitecture", "thinker"), AnotherAdapter)
    engine = _engine("AnotherArchitecture")
    engine.model_config.hf_config.model_type = "another_model"
    assert engine._get_model_config_path() == "checkpoint"
    assert engine.model_adapter_cls is AnotherAdapter
    inputs, output_args = engine.prepare_model_inputs(_batch())
    assert set(inputs) == {"input_ids", "conditioning", "trajectory"}
    assert inputs["conditioning"] == 42
    assert output_args == {"output_metadata": True}
    model = torch.nn.Linear(2, 2)
    assert [id(p) for p in engine._build_optimizer(model)] == [id(model.weight)]
    assert calls == ["setup", "backend_inputs", "replay_inputs", "trainability"]


def test_unported_adapter_fails_before_model_loading(monkeypatch):
    class NativeOnlyAdapter(OmniModelBase):
        pass

    monkeypatch.setitem(OmniModelBase._registry, ("NativeOnly", "thinker"), NativeOnlyAdapter)
    with pytest.raises(NotImplementedError, match="NativeOnlyAdapter does not support the VeOmni backend"):
        _engine("NativeOnly")._get_model_config_path()


def test_external_adapter_uses_existing_registry_loader(monkeypatch):
    engine = _engine("ExternalArchitecture")
    engine.model_config.external_lib = "test_external_adapter"
    external_adapter = MagicMock()

    def register_external(name):
        assert name == "test_external_adapter"
        monkeypatch.setitem(OmniModelBase._registry, ("ExternalArchitecture", "thinker"), external_adapter)

    with patch("verl.utils.import_utils.import_external_libs", side_effect=register_external):
        assert engine._get_model_config_path() == "checkpoint"
    external_adapter.setup_veomni.assert_called_once_with(engine.model_config, engine.engine_config)


@pytest.mark.parametrize("hook", ["prepare_veomni_inputs", "prepare_model_inputs"])
def test_rejects_invalid_adapter_input_return(hook, monkeypatch):
    class InvalidAdapter(OmniModelBase):
        @classmethod
        def setup_veomni(cls, model_config, engine_config):
            pass

    monkeypatch.setitem(OmniModelBase._registry, ("Invalid", "thinker"), InvalidAdapter)
    monkeypatch.setattr(InvalidAdapter, hook, MagicMock(return_value=None))
    engine = _engine("Invalid")
    engine._get_model_config_path()
    with pytest.raises(TypeError, match=f"{hook} must return a dict"):
        engine.prepare_model_inputs(_batch())


def test_native_qwen_adapter_does_not_require_veomni():
    model = torch.nn.Module()
    model.thinker = SimpleNamespace(
        forward=MagicMock(return_value="thinker output"),
        get_input_embeddings=MagicMock(),
        set_input_embeddings=MagicMock(),
    )
    model.talker = torch.nn.Linear(2, 2)
    model.code2wav = torch.nn.Linear(2, 2)
    model.code_predictor = torch.nn.Linear(2, 2)
    with patch.dict(sys.modules, {_QWEN_BACKEND: None, "veomni": None}):
        configured = Qwen3OmniThinkerAdapter.configure_model(model, _engine().model_config)
    assert configured is model
    assert configured.forward() == "thinker output"
    assert not any(hasattr(configured, name) for name in ("talker", "code2wav", "code_predictor"))


def test_opted_in_adapter_can_keep_default_inputs_and_trainability(monkeypatch):
    class PlainAdapter(OmniModelBase):
        @classmethod
        def setup_veomni(cls, model_config, engine_config):
            pass

    monkeypatch.setitem(OmniModelBase._registry, ("Plain", "thinker"), PlainAdapter)
    engine = _engine("Plain")
    engine._get_model_config_path()
    inputs, output_args = engine.prepare_model_inputs(_batch())
    assert inputs["base_transform_called"]
    torch.testing.assert_close(inputs["input_ids"], _batch()["input_ids"].values().unsqueeze(0))
    assert output_args == {"output_metadata": True}
    model = torch.nn.Linear(2, 2)
    assert {id(p) for p in engine._build_optimizer(model)} == {id(p) for p in model.parameters()}
