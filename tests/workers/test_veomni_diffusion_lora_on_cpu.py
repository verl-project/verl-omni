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
"""CPU checks for LoRA support in the VeOmni diffusion engine."""

import builtins
from unittest.mock import MagicMock

import pytest
import torch

import verl_omni.workers.engine.veomni.diffusion_impl as veomni_impl
from verl_omni.workers.config.diffusion import DiffusionModelConfig
from verl_omni.workers.engine.veomni.diffusion_impl import VeOmniDiffusionEngine

veomni_lora = pytest.importorskip("veomni.lora")


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = torch.nn.Linear(8, 8, bias=False)
        self.to_k = torch.nn.Linear(8, 8, bias=False)


class _ToyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([_Block()])
        self.proj_out = torch.nn.Linear(8, 8, bias=False)


def _make_engine(
    module=None,
    *,
    lora_rank: int = 0,
    lora_alpha: int = 64,
    target_modules=None,
    lora_adapter_path=None,
    lora: dict | None = None,
) -> VeOmniDiffusionEngine:
    """Build an engine without running ``__init__`` (which needs torch.distributed)."""
    engine = object.__new__(VeOmniDiffusionEngine)
    engine.module = module
    engine._is_offload_param = False
    engine._is_lora = lora_rank > 0 or lora_adapter_path is not None
    # DiffusionModelConfig.__post_init__ does I/O; set the fields under test directly.
    model_config = object.__new__(DiffusionModelConfig)
    object.__setattr__(model_config, "lora_rank", lora_rank)
    object.__setattr__(model_config, "lora_alpha", lora_alpha)
    object.__setattr__(model_config, "target_modules", target_modules)
    object.__setattr__(model_config, "exclude_modules", None)
    object.__setattr__(model_config, "lora_adapter_path", lora_adapter_path)
    object.__setattr__(model_config, "lora", lora if lora is not None else {})
    engine.model_config = model_config
    engine.engine_config = MagicMock(model_dtype="bf16")
    return engine


def _lora_module(target_modules=("to_q", "to_k")):
    config = veomni_lora.VeOmniLoraConfig(r=4, lora_alpha=8, target_modules=list(target_modules))
    return veomni_lora.VeOmniLoraModel(_ToyTransformer(), config)


# --------------------------------------------------------------------------
# lora_config translation
# --------------------------------------------------------------------------


def test_lora_config_is_empty_without_lora():
    engine = _make_engine()
    assert engine._build_veomni_lora_config() == {}


def test_lora_config_maps_verl_omni_fields_to_veomni_names():
    engine = _make_engine(lora_rank=64, lora_alpha=128, target_modules=["to_q", "to_k"])

    config = engine._build_veomni_lora_config()

    assert config["rank"] == 64
    assert config["alpha"] == 128
    # VeOmni calls the target list ``lora_modules``.
    assert config["lora_modules"] == ["to_q", "to_k"]
    assert "lora_adapter" not in config


def test_lora_config_requires_veomni_0_1_12(monkeypatch):
    engine = _make_engine(lora_rank=64, target_modules=["to_q"])
    real_import = builtins.__import__

    def import_without_veomni_lora(name, *args, **kwargs):
        if name == "veomni.lora":
            raise ModuleNotFoundError(name, name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_veomni_lora)
    with pytest.raises(RuntimeError, match="veomni>=0.1.12"):
        engine._build_veomni_lora_config()


def test_lora_config_forwards_the_adapter_path_for_resume():
    engine = _make_engine(lora_rank=64, target_modules=["to_q"], lora_adapter_path="/tmp/adapter")
    assert engine._build_veomni_lora_config()["lora_adapter"] == "/tmp/adapter"


def test_lora_config_rejects_all_linear():
    """VeOmni has no ``all-linear`` shorthand and would inject zero adapters."""
    engine = _make_engine(lora_rank=64, target_modules="all-linear")

    with pytest.raises(ValueError, match="all-linear"):
        engine._build_veomni_lora_config()


def test_veomni_really_matches_nothing_for_all_linear():
    """Guards the reason the check above exists, not just the check itself."""
    config = veomni_lora.VeOmniLoraConfig(r=4, lora_alpha=8, target_modules="all-linear")
    from veomni.lora.mapping import find_target_linear_names

    assert find_target_linear_names(_ToyTransformer(), config) == []


# --------------------------------------------------------------------------
# weight export
# --------------------------------------------------------------------------


def _export(engine, monkeypatch, **kwargs):
    monkeypatch.setattr(veomni_impl, "load_model_to_gpu", MagicMock())
    monkeypatch.setattr(veomni_impl, "offload_model_to_cpu", MagicMock())
    monkeypatch.setattr(veomni_impl, "get_device_id", lambda: torch.device("cpu"))
    monkeypatch.setattr(veomni_impl.PrecisionType, "to_dtype", staticmethod(lambda _: torch.float32))
    generator, peft_config = engine.get_per_tensor_param(**kwargs)
    return dict(generator), peft_config


def test_adapter_sync_exports_peft_formatted_lora_tensors(monkeypatch):
    engine = _make_engine(_lora_module(), lora_rank=4, target_modules=["to_q", "to_k"])

    params, peft_config = _export(engine, monkeypatch, base_sync_done=True)

    assert peft_config["peft_type"] == "LORA"
    assert peft_config["r"] == 4
    # Transformer-relative keys for live rollout sync, not PEFT file prefixes.
    assert set(params) == {
        "transformer.transformer_blocks.0.to_q.lora_A.weight",
        "transformer.transformer_blocks.0.to_q.lora_B.weight",
        "transformer.transformer_blocks.0.to_k.lora_A.weight",
        "transformer.transformer_blocks.0.to_k.lora_B.weight",
    }


def test_adapter_sync_keys_resolve_in_actual_diffusion_lora_manager(monkeypatch):
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.peft_helper import PEFTHelper
    from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager

    engine = _make_engine(_lora_module(), lora_rank=4, target_modules=["to_q", "to_k"])
    params, config = _export(engine, monkeypatch, base_sync_done=True)
    loaded = LoRAModel.from_lora_tensors(
        lora_model_id=1,
        tensors=params,
        peft_helper=PEFTHelper.from_dict(config),
        device="cpu",
        dtype=torch.float32,
    )
    manager = object.__new__(DiffusionLoRAManager)
    for target in ("to_q", "to_k"):
        runtime_name = f"transformer.transformer_blocks.0.{target}"
        assert manager._get_lora_weights(loaded, runtime_name) is not None, (
            f"Exported LoRA cannot bind to {runtime_name}; loaded names: {list(loaded.loras)}"
        )


def test_base_sync_exports_plain_transformer_keys(monkeypatch):
    """The first sync must not leak the LoRA wrapper prefix or ``base_layer``."""
    engine = _make_engine(_lora_module(), lora_rank=4, target_modules=["to_q", "to_k"])

    params, peft_config = _export(engine, monkeypatch, base_sync_done=False)

    expected = {f"transformer.{name}" for name in _ToyTransformer().state_dict()}
    assert set(params) == expected
    assert not any("base_model" in name or "base_layer" in name for name in params)
    assert peft_config is not None


def test_export_fails_closed_when_the_adapter_was_never_injected(monkeypatch):
    """A plain module with LoRA configured means the sync would ship base weights."""
    engine = _make_engine(_ToyTransformer(), lora_rank=4, target_modules=["to_q"])

    with pytest.raises(RuntimeError, match="not a LoRA model"):
        _export(engine, monkeypatch, base_sync_done=True)


def test_export_without_lora_keeps_the_full_state_dict(monkeypatch):
    engine = _make_engine(_ToyTransformer())

    params, peft_config = _export(engine, monkeypatch)

    assert peft_config is None
    assert set(params) == {f"transformer.{name}" for name in _ToyTransformer().state_dict()}


# --------------------------------------------------------------------------
# disable_adapter
# --------------------------------------------------------------------------


def test_disable_adapter_bypasses_the_lora_delta():
    module = _lora_module()
    engine = _make_engine(module, lora_rank=4, target_modules=["to_q", "to_k"])
    linear = module.get_base_model().transformer_blocks[0].to_q
    with torch.no_grad():
        linear.lora_A["default"].weight.fill_(0.3)
        linear.lora_B["default"].weight.fill_(0.7)

    x = torch.randn(2, 8)
    with torch.no_grad():
        enabled = linear(x)
        with engine.disable_adapter():
            disabled = linear(x)
        restored = linear(x)

    # The adapter is actually contributing, so a no-op context would be invisible.
    assert not torch.allclose(enabled, disabled)
    assert torch.equal(disabled, linear.base_layer(x))
    assert torch.equal(enabled, restored)


def test_disable_adapter_restores_the_adapter_on_error():
    module = _lora_module()
    engine = _make_engine(module, lora_rank=4, target_modules=["to_q"])
    linear = module.get_base_model().transformer_blocks[0].to_q
    active = linear.active_adapter

    with pytest.raises(RuntimeError, match="boom"), engine.disable_adapter():
        raise RuntimeError("boom")

    assert linear.active_adapter == active


def test_disable_adapter_fails_when_lora_was_not_injected():
    engine = _make_engine(_ToyTransformer(), lora_rank=4, target_modules=["to_q"])
    with pytest.raises(RuntimeError, match="no VeOmni LoRA layers"):
        with engine.disable_adapter():
            pass


def test_disable_adapter_is_a_no_op_without_lora():
    engine = _make_engine(_ToyTransformer())
    with engine.disable_adapter():
        pass
