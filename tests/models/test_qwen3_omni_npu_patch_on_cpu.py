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
"""CPU coverage for Qwen3-Omni NPU patch boundaries; kernels are mocked."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(relative_path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def patch_env(monkeypatch):
    modeling = ModuleType("modeling_qwen3_omni_moe")
    names = (
        "Qwen3OmniMoeThinkerTextExperts",
        "Qwen3OmniMoeThinkerTextRMSNorm",
        "Qwen3OmniMoeTextRMSNorm",
        "Qwen3OmniMoeThinkerTextMLP",
        "Qwen3OmniMoeTalkerTextExperts",
        "Qwen3OmniMoeTalkerTextMLP",
    )
    for name in names:
        setattr(modeling, name, type(name, (), {"forward": Mock(name=name)}))
    modeling.apply_rotary_pos_emb = Mock(name="original_rope")
    kernels = ModuleType("verl.models.transformers.npu_patch")
    for name in (
        "rms_norm_forward_npu",
        "silu_forward_npu",
        "apply_rotary_pos_emb_npu",
    ):
        setattr(kernels, name, Mock(name=name))
    package = ModuleType("transformers.models.qwen3_omni_moe")
    package.modeling_qwen3_omni_moe = modeling
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, kernels.__name__, kernels)
    patch = _load("verl_omni/models/transformers/qwen3_omni_npu.py", "qwen3_omni_npu_under_test")
    return patch, modeling, kernels


def test_patch_covers_experts_and_final_norm_and_preserves_talker(patch_env):
    patch, modeling, kernels = patch_env
    talker = modeling.Qwen3OmniMoeTalkerTextExperts.forward
    talker_mlp = modeling.Qwen3OmniMoeTalkerTextMLP.forward
    for _ in range(2):
        patch.apply_qwen3_omni_thinker_npu_patch(SimpleNamespace(hidden_act="silu"))
        assert modeling.Qwen3OmniMoeThinkerTextExperts.forward is patch.qwen3_omni_experts_forward_npu
        assert modeling.Qwen3OmniMoeThinkerTextRMSNorm.forward is kernels.rms_norm_forward_npu
        assert modeling.Qwen3OmniMoeTextRMSNorm.forward is kernels.rms_norm_forward_npu
        assert modeling.Qwen3OmniMoeThinkerTextMLP.forward is kernels.silu_forward_npu
        assert modeling.Qwen3OmniMoeTalkerTextExperts.forward is talker
        assert modeling.Qwen3OmniMoeTalkerTextMLP.forward is talker_mlp


@pytest.mark.parametrize("mode", ["default", "positional", "keyword"])
def test_rope_preserves_transformers_call_signature(patch_env, mode):
    patch, modeling, kernels = patch_env
    patch.apply_qwen3_omni_thinker_npu_patch(SimpleNamespace(hidden_act="silu"))
    args = tuple(object() for _ in range(4))
    if mode == "default":
        result = modeling.apply_rotary_pos_emb(*args)
    elif mode == "positional":
        result = modeling.apply_rotary_pos_emb(*args, 2)
    else:
        result = modeling.apply_rotary_pos_emb(*args, unsqueeze_dim=2)
    kernels.apply_rotary_pos_emb_npu.assert_called_once_with(*args, unsqueeze_dim=1 if mode == "default" else 2)
    assert result is kernels.apply_rotary_pos_emb_npu.return_value


def test_unsupported_activation_does_not_partially_patch(patch_env):
    patch, modeling, _ = patch_env
    original = modeling.Qwen3OmniMoeThinkerTextExperts.forward
    rope = modeling.apply_rotary_pos_emb
    with pytest.raises(ValueError, match="hidden_act='silu'"):
        patch.apply_qwen3_omni_thinker_npu_patch(SimpleNamespace(hidden_act="gelu"))
    assert modeling.Qwen3OmniMoeThinkerTextExperts.forward is original
    assert modeling.apply_rotary_pos_emb is rope


def test_missing_transformers_class_does_not_partially_patch(patch_env):
    patch, modeling, _ = patch_env
    original = modeling.Qwen3OmniMoeThinkerTextExperts.forward
    del modeling.Qwen3OmniMoeTextRMSNorm
    with pytest.raises(AttributeError):
        patch.apply_qwen3_omni_thinker_npu_patch(SimpleNamespace(hidden_act="silu"))
    assert modeling.Qwen3OmniMoeThinkerTextExperts.forward is original


def test_patch_reaches_already_loaded_transformers_modules(monkeypatch):
    modeling = pytest.importorskip("transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe")
    config = modeling.Qwen3OmniMoeTextConfig(
        hidden_size=128, intermediate_size=256, moe_intermediate_size=256, num_experts=4
    )
    layers = [
        modeling.Qwen3OmniMoeThinkerTextExperts(config),
        modeling.Qwen3OmniMoeThinkerTextMLP(config),
        modeling.Qwen3OmniMoeThinkerTextRMSNorm(128),
        modeling.Qwen3OmniMoeTextRMSNorm(128),
    ]
    parameters = [dict(layer.named_parameters()) for layer in layers]
    for layer in layers:
        monkeypatch.setattr(type(layer), "forward", type(layer).forward)
    monkeypatch.setattr(modeling, "apply_rotary_pos_emb", modeling.apply_rotary_pos_emb)

    def forward(self, *args):
        return self, args

    kernels = SimpleNamespace(
        rms_norm_forward_npu=forward,
        silu_forward_npu=forward,
        apply_rotary_pos_emb_npu=Mock(),
    )
    monkeypatch.setitem(sys.modules, "verl.models.transformers.npu_patch", kernels)
    patch = _load("verl_omni/models/transformers/qwen3_omni_npu.py", "qwen3_omni_npu_real_classes")
    monkeypatch.setattr(patch, "qwen3_omni_experts_forward_npu", forward)
    patch.apply_qwen3_omni_thinker_npu_patch(config)
    hidden_states = SimpleNamespace(device="cpu", dtype="float32", shape=(16, 128))
    for layer, original_parameters in zip(layers, parameters, strict=True):
        inputs = (hidden_states,)
        instance, args = layer(*inputs)
        assert instance is layer
        assert args == inputs
        assert dict(layer.named_parameters()).keys() == original_parameters.keys()
        assert all(param is original_parameters[name] for name, param in layer.named_parameters())


def test_patch_log_reports_gmm_experts(patch_env, caplog):
    patch, _, _ = patch_env
    with caplog.at_level("INFO", logger=patch.logger.name):
        patch.apply_qwen3_omni_thinker_npu_patch(SimpleNamespace(hidden_act="silu"))
    assert "MoE GMM, RMSNorm, MLP and RoPE" in caplog.text


@pytest.mark.parametrize("on_npu", [False, True])
def test_adapter_gates_patch_and_keeps_thinker_forward(monkeypatch, on_npu):
    class Base:
        @classmethod
        def register(cls, *args, **kwargs):
            return lambda adapter: adapter

        @classmethod
        def configure_model(cls, module, model_config):
            for name in cls.get_strip_modules(model_config):
                delattr(module, name)
            return module

    monkeypatch.setitem(sys.modules, "verl_omni.pipelines.model_base", SimpleNamespace(OmniModelBase=Base))
    monkeypatch.setitem(sys.modules, "verl.utils.device", SimpleNamespace(is_npu_available=on_npu))
    patch = Mock()
    monkeypatch.setitem(
        sys.modules,
        "verl_omni.models.transformers.qwen3_omni_npu",
        SimpleNamespace(apply_qwen3_omni_thinker_npu_patch=patch),
    )
    adapter = _load("verl_omni/pipelines/qwen3_omni/thinker_training_adapter.py", "thinker_adapter_under_test")
    text_config = SimpleNamespace(hidden_act="silu")
    thinker = SimpleNamespace(
        config=SimpleNamespace(text_config=text_config),
        forward=Mock(),
        get_input_embeddings=Mock(),
        set_input_embeddings=Mock(),
    )
    module = SimpleNamespace(thinker=thinker, talker=object(), code2wav=object(), code_predictor=object())
    result = adapter.Qwen3OmniThinkerAdapter.configure_model(module, SimpleNamespace())
    assert result is module
    assert result.forward is thinker.forward
    assert result.get_input_embeddings is thinker.get_input_embeddings
    assert not hasattr(result, "talker")
    if on_npu:
        patch.assert_called_once_with(text_config)
    else:
        patch.assert_not_called()
