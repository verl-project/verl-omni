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
"""Shared diffusion packing configuration and engine dispatch, without GPUs."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config import DiffusionModelConfig
from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine


def _config(tmp_path, architecture="TestPackedPipeline", **kwargs):
    return DiffusionModelConfig(
        path=str(tmp_path),
        load_tokenizer=False,
        architecture=architecture,
        algorithm="flow_grpo",
        attn_backend="native",
        enable_gradient_checkpointing=False,
        **kwargs,
    )


def _engine(config):
    engine = object.__new__(PPODiffusersFSDPEngine)
    engine.model_config = config
    engine.engine_config = SimpleNamespace(model_dtype="fp32", strategy="fsdp2", ulysses_sequence_parallel_size=1)
    return engine


@pytest.mark.parametrize("enabled", [False, True])
def test_diffusion_remove_padding_config_and_hydra(tmp_path, enabled):
    from pathlib import Path

    config = _config(tmp_path, use_remove_padding=enabled)
    assert config.use_remove_padding is enabled
    assert not hasattr(config, "use_packed_batch")
    config_dir = Path(__file__).resolve().parents[2] / "verl_omni/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        default = compose(config_name="diffusion_trainer")
        overridden = compose(
            config_name="diffusion_trainer",
            overrides=[f"actor_rollout_ref.model.use_remove_padding={str(enabled).lower()}"],
        )
    assert default.actor_rollout_ref.model.use_remove_padding is False
    assert overridden.actor_rollout_ref.model.use_remove_padding is enabled
    assert "use_packed_batch" not in overridden.actor_rollout_ref.model
    # Packing must not change the actor's batching configuration.
    assert OmegaConf.to_container(overridden.actor_rollout_ref.actor, resolve=False) == OmegaConf.to_container(
        default.actor_rollout_ref.actor, resolve=False
    )


def test_omni_remove_padding_config_is_unchanged():
    from dataclasses import fields

    from verl_omni.workers.config.omni.model import OmniModelConfig

    model_fields = {field.name: field for field in fields(OmniModelConfig)}
    assert model_fields["use_remove_padding"].default is True
    assert "use_packed_batch" not in model_fields


@pytest.mark.parametrize("enabled", [False, True])
def test_custom_loader_uses_shared_packing_hook(tmp_path, monkeypatch, enabled):
    events = []
    module = torch.nn.Linear(2, 2)

    class Adapter(DiffusionModelBase):
        supports_remove_padding = True

        @classmethod
        def build_module(cls, model_config, torch_dtype):
            events.append("load")
            return module

        @classmethod
        def apply_remove_padding(cls, loaded, model_config):
            assert loaded is module
            events.append("pack")

    monkeypatch.setitem(DiffusionModelBase._registry, ("TestPackedPipeline", "flow_grpo"), Adapter)
    config = _config(tmp_path, architecture="TestPackedPipeline", use_remove_padding=enabled)
    engine = _engine(config)
    assert engine._build_module() is module
    assert events == (["load", "pack"] if enabled else ["load"])


def test_unsupported_adapter_fails_before_model_loading(tmp_path, monkeypatch):
    class Unsupported(DiffusionModelBase):
        @classmethod
        def build_module(cls, model_config, torch_dtype):
            pytest.fail("Unsupported packing must fail before loading weights")

    monkeypatch.setitem(DiffusionModelBase._registry, ("TestPackedPipeline", "flow_grpo"), Unsupported)
    with pytest.raises(NotImplementedError, match="use_remove_padding"):
        _engine(_config(tmp_path, use_remove_padding=True))._build_module()


@pytest.mark.parametrize(
    "strategy,sp,error", [("veomni", 1, "fsdp/fsdp2 engine"), ("fsdp2", 2, "sequence parallelism")]
)
def test_unsupported_execution_fails_before_model_loading(tmp_path, monkeypatch, strategy, sp, error):
    class Supported(DiffusionModelBase):
        supports_remove_padding = True

    monkeypatch.setitem(DiffusionModelBase._registry, ("TestPackedPipeline", "flow_grpo"), Supported)
    config = _config(tmp_path, use_remove_padding=True)
    engine = _engine(config)
    engine.engine_config.strategy = strategy
    engine.engine_config.ulysses_sequence_parallel_size = sp
    monkeypatch.setattr(engine, "_build_module_from_registry", lambda *_: pytest.fail("Loaded weights"))
    with pytest.raises(NotImplementedError, match=error):
        engine._build_module()


def test_disabled_packing_leaves_unrelated_adapters_unchanged(tmp_path, monkeypatch):
    module = torch.nn.Linear(2, 2)

    class Unsupported(DiffusionModelBase):
        @classmethod
        def build_module(cls, model_config, torch_dtype):
            return module

    monkeypatch.setitem(DiffusionModelBase._registry, ("TestPackedPipeline", "flow_grpo"), Unsupported)
    config = _config(tmp_path)
    assert config.use_remove_padding is False
    DiffusionModelBase.validate_remove_padding(config, SimpleNamespace(strategy="veomni"))
    assert _engine(config)._build_module() is module


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_automodel_loader_uses_shared_packing_hook(tmp_path, monkeypatch, enabled, strategy):
    from diffusers import AutoModel

    from verl_omni.workers.engine.fsdp import diffusers_impl

    events = []
    module = torch.nn.Linear(2, 2)
    module.config = SimpleNamespace()
    module.set_attention_backend = lambda backend: events.append(("attention", backend))
    module.enable_gradient_checkpointing = lambda: events.append("checkpointing")

    class Adapter(DiffusionModelBase):
        supports_remove_padding = True

        @classmethod
        def apply_remove_padding(cls, loaded, model_config):
            assert loaded is module
            events.append("pack")

    def load(*args, **kwargs):
        events.append("load")
        return module

    monkeypatch.setitem(DiffusionModelBase._registry, ("TestPackedPipeline", "flow_grpo"), Adapter)
    monkeypatch.setattr(AutoModel, "from_pretrained", load)
    monkeypatch.setattr(diffusers_impl, "get_init_weight_context_manager", lambda **_: nullcontext)
    config = _config(tmp_path, use_remove_padding=enabled)
    object.__setattr__(config, "enable_gradient_checkpointing", True)
    engine = _engine(config)
    engine.engine_config.strategy = strategy
    engine.device_mesh = None
    assert engine._build_module() is module
    assert events == ["load", ("attention", "native"), "checkpointing"] + (["pack"] if enabled else [])
