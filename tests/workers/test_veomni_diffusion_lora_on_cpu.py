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

import asyncio
import builtins
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

# The engine imports veomni, which the CPU CI does not install.
pytest.importorskip("veomni")
import verl_omni.workers.engine.veomni.diffusion_impl as veomni_impl
import verl_omni.workers.engine.veomni.lora_utils as veomni_lora_utils
from tests.workers.veomni_lora_helpers import export_veomni_params, make_veomni_engine
from verl_omni.workers.config.diffusion import DiffusionModelConfig

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


def _lora_module(target_modules=("to_q", "to_k")):
    config = veomni_lora.VeOmniLoraConfig(r=4, lora_alpha=8, target_modules=list(target_modules))
    return veomni_lora.VeOmniLoraModel(_ToyTransformer(), config)


# --------------------------------------------------------------------------
# lora_config translation
# --------------------------------------------------------------------------


def test_lora_config_is_empty_without_lora():
    engine = make_veomni_engine()
    assert engine._build_veomni_lora_config() == {}
    assert engine.get_lora_peft_config() is None


def test_lora_config_maps_verl_omni_fields_to_veomni_names():
    engine = make_veomni_engine(lora_rank=64, lora_alpha=128, target_modules=["to_q", "to_k"])

    config = engine._build_veomni_lora_config()

    assert config["rank"] == 64
    assert config["alpha"] == 128
    # VeOmni calls the target list ``lora_modules``.
    assert config["lora_modules"] == ["to_q", "to_k"]
    assert "lora_adapter" not in config


def test_lora_config_requires_veomni_0_1_12(monkeypatch):
    engine = make_veomni_engine(lora_rank=64, target_modules=["to_q"])
    real_import = builtins.__import__

    def import_without_veomni_lora(name, *args, **kwargs):
        if name == "veomni.lora":
            raise ModuleNotFoundError(name, name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_veomni_lora)
    with pytest.raises(RuntimeError, match="veomni>=0.1.12"):
        engine._build_veomni_lora_config()


def test_lora_config_forwards_the_adapter_path_for_resume():
    engine = make_veomni_engine(lora_rank=64, target_modules=["to_q"], lora_adapter_path="/tmp/adapter")
    assert engine._build_veomni_lora_config()["lora_adapter"] == "/tmp/adapter"


def test_lora_config_allows_all_linear_when_loading_an_adapter():
    """VeOmni rebuilds the targets from adapter_config.json, so the default target list is unused."""
    engine = make_veomni_engine(target_modules="all-linear", lora_adapter_path="/tmp/adapter")
    assert engine._build_veomni_lora_config()["lora_adapter"] == "/tmp/adapter"


def test_lora_config_rejects_all_linear():
    """VeOmni has no ``all-linear`` shorthand and would inject zero adapters."""
    engine = make_veomni_engine(lora_rank=64, target_modules="all-linear")

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


def test_adapter_sync_exports_peft_formatted_lora_tensors(monkeypatch):
    engine = make_veomni_engine(_lora_module(), lora_rank=4, target_modules=["to_q", "to_k"])

    params, peft_config = export_veomni_params(engine, monkeypatch, base_sync_done=True)

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

    engine = make_veomni_engine(_lora_module(), lora_rank=4, target_modules=["to_q", "to_k"])
    params, config = export_veomni_params(engine, monkeypatch, base_sync_done=True)
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


def test_worker_reads_veomni_lora_metadata_and_checksum_without_export():
    from verl_omni.workers.engine_workers import ActorRolloutRefWorker

    engine = make_veomni_engine(_lora_module(), lora_rank=4)
    engine.get_per_tensor_param = MagicMock(side_effect=AssertionError("metadata must not gather weights"))
    worker = object.__new__(ActorRolloutRefWorker)
    worker.role = "actor"
    worker.peft_merge = False
    worker.actor = SimpleNamespace(engine=engine)

    config = worker.get_lora_peft_config()
    assert config is not None
    assert config["r"] == 4
    assert config["lora_alpha"] == 8
    checksum = worker.get_lora_weight_checksum()
    assert checksum["num_lora_tensors"] == 4
    engine.get_per_tensor_param.assert_not_called()
    worker.peft_merge = True
    assert worker.get_lora_peft_config() is None
    assert worker.get_lora_weight_checksum() is None


def test_non_naive_worker_sends_veomni_adapters_on_every_update(monkeypatch):
    from verl_omni.workers.engine_workers import ActorRolloutRefWorker

    engine = make_veomni_engine(_lora_module(), lora_rank=4)
    monkeypatch.setattr(veomni_impl, "load_model_to_gpu", MagicMock())
    monkeypatch.setattr(veomni_impl, "get_device_id", lambda: torch.device("cpu"))
    worker = object.__new__(ActorRolloutRefWorker)
    worker.actor = SimpleNamespace(engine=engine)
    worker.config = SimpleNamespace(rollout=SimpleNamespace(checkpoint_engine=SimpleNamespace(backend="nccl")))
    worker.peft_merge = False
    worker.rollout_adapter = "default"
    worker._rank = 0
    sent = []

    async def capture(params, global_steps):
        sent.append((global_steps, dict(params)))

    worker.checkpoint_engine = SimpleNamespace(send_weights=AsyncMock(side_effect=capture))
    for step in (1, 2):
        asyncio.run(worker.update_weights(global_steps=step))
    assert [step for step, _ in sent] == [1, 2]
    for _, params in sent:
        assert len(params) == 4
        assert all(".lora_" in name for name in params)


def test_colocated_worker_uses_veomni_lora_fast_path(monkeypatch):
    from tests.workers.test_omni_lora_weight_sync_on_cpu import _fast_path_worker, _run_update

    worker = _fast_path_worker()
    engine = make_veomni_engine(_lora_module(), lora_rank=4)
    monkeypatch.setattr(veomni_impl, "load_model_to_gpu", MagicMock())
    monkeypatch.setattr(veomni_impl, "get_device_id", lambda: torch.device("cpu"))
    worker.actor = SimpleNamespace(engine=engine)
    sender = _run_update(worker, global_steps=2)
    sender.async_send_weights.assert_awaited_once()
    exported = dict(sender.async_send_weights.call_args.args[0])
    assert len(exported) == 4
    assert all(".lora_" in name for name in exported)
    assert worker.rollout._execute_method.call_args.kwargs["kwargs"]["peft_config"]["r"] == 4
    worker.rollout.update_weights.assert_not_called()


def test_base_sync_exports_plain_transformer_keys(monkeypatch):
    """The first sync must not leak the LoRA wrapper prefix or ``base_layer``."""
    engine = make_veomni_engine(_lora_module(), lora_rank=4, target_modules=["to_q", "to_k"])

    params, peft_config = export_veomni_params(engine, monkeypatch, base_sync_done=False)

    expected = {f"transformer.{name}" for name in _ToyTransformer().state_dict()}
    assert set(params) == expected
    assert not any("base_model" in name or "base_layer" in name for name in params)
    assert peft_config is not None


def test_export_fails_closed_when_the_adapter_was_never_injected(monkeypatch):
    """A plain module with LoRA configured means the sync would ship base weights."""
    engine = make_veomni_engine(_ToyTransformer(), lora_rank=4, target_modules=["to_q"])

    with pytest.raises(RuntimeError, match="not a LoRA model"):
        export_veomni_params(engine, monkeypatch, base_sync_done=True)


def test_export_without_lora_keeps_the_full_state_dict(monkeypatch):
    engine = make_veomni_engine(_ToyTransformer())

    params, peft_config = export_veomni_params(engine, monkeypatch)

    assert peft_config is None
    assert set(params) == {f"transformer.{name}" for name in _ToyTransformer().state_dict()}


# --------------------------------------------------------------------------
# disable_adapter
# --------------------------------------------------------------------------


def test_disable_adapter_bypasses_the_lora_delta():
    module = _lora_module()
    engine = make_veomni_engine(module, lora_rank=4, target_modules=["to_q", "to_k"])
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
    engine = make_veomni_engine(module, lora_rank=4, target_modules=["to_q"])
    linear = module.get_base_model().transformer_blocks[0].to_q
    active = linear.active_adapter

    with pytest.raises(RuntimeError, match="boom"), engine.disable_adapter():
        raise RuntimeError("boom")

    assert linear.active_adapter == active


def test_disable_adapter_fails_when_lora_was_not_injected():
    engine = make_veomni_engine(_ToyTransformer(), lora_rank=4, target_modules=["to_q"])
    with pytest.raises(RuntimeError, match="no VeOmni LoRA layers"):
        with engine.disable_adapter():
            pass


def test_disable_adapter_is_a_no_op_without_lora():
    engine = make_veomni_engine(_ToyTransformer())
    with engine.disable_adapter():
        pass


# --------------------------------------------------------------------------
# startup validation
# --------------------------------------------------------------------------


def _model_config(**overrides):
    """A DiffusionModelConfig carrying only the fields the validator reads."""
    config = object.__new__(DiffusionModelConfig)
    defaults = {
        "lora": {},
        "policy_state_adapters": ("default",),
        "lora_dtype": None,
        "target_parameters": None,
        "lora_init_weights": "true",
        "lora_adapter_path": None,
    }
    for name, value in {**defaults, **overrides}.items():
        object.__setattr__(config, name, value)
    return config


def test_validation_accepts_the_supported_lora_setup():
    veomni_lora_utils._validate_veomni_lora_support(_model_config())


@pytest.mark.parametrize("value", [True, "true", "True", "kaiming"])
def test_validation_accepts_every_spelling_of_kaiming_init(value):
    veomni_lora_utils._validate_veomni_lora_support(_model_config(lora_init_weights=value))


def test_validation_rejects_merge():
    with pytest.raises(NotImplementedError, match="merge=True"):
        veomni_lora_utils._validate_veomni_lora_support(_model_config(lora={"merge": True}))


def test_validation_rejects_named_policy_state_adapters():
    """VeOmniLoraModel is single-adapter, so old/EMA policy states cannot exist."""
    with pytest.raises(NotImplementedError, match="policy_state_adapters"):
        veomni_lora_utils._validate_veomni_lora_support(_model_config(policy_state_adapters=("default", "old")))


def test_validation_accepts_the_logical_reference_policy_state():
    """``reference`` is served by disable_adapter, not by a second adapter."""
    veomni_lora_utils._validate_veomni_lora_support(_model_config(policy_state_adapters=("default", "reference")))


def test_validation_rejects_lora_dtype():
    with pytest.raises(NotImplementedError, match="lora_dtype"):
        veomni_lora_utils._validate_veomni_lora_support(_model_config(lora_dtype="float32"))


def test_validation_rejects_target_parameters():
    with pytest.raises(NotImplementedError, match="target_parameters"):
        veomni_lora_utils._validate_veomni_lora_support(_model_config(target_parameters=["experts.gate_up_proj"]))


def test_validation_rejects_gaussian_init():
    """The verl-omni default; VeOmni would silently Kaiming-init instead."""
    with pytest.raises(NotImplementedError, match="lora_init_weights"):
        veomni_lora_utils._validate_veomni_lora_support(_model_config(lora_init_weights="gaussian"))


def test_validation_ignores_init_when_loading_an_adapter():
    """Loaded adapter weights replace the initialization, including the gaussian default."""
    veomni_lora_utils._validate_veomni_lora_support(
        _model_config(lora_init_weights="gaussian", lora_adapter_path="/tmp/adapter")
    )


def test_build_rejects_moe_expert_lora():
    """disable_adapter only bypasses dense LoRA layers, so expert LoRA must fail at startup."""
    from veomni.lora.moe_layers import LoraIndependentExperts

    class _Experts(LoraIndependentExperts):
        def __init__(self):
            torch.nn.Module.__init__(self)

    model = _lora_module()
    model.get_base_model().experts = _Experts()
    with pytest.raises(NotImplementedError, match="MoE expert LoRA"):
        veomni_lora_utils._reject_veomni_moe_expert_lora(model)


def test_build_accepts_dense_lora():
    veomni_lora_utils._reject_veomni_moe_expert_lora(_lora_module())


def test_veomni_really_only_implements_kaiming_init():
    """Guards the reason the check above exists, not just the check itself."""
    import inspect

    from veomni.lora.layers import LoraLinear

    source = inspect.getsource(LoraLinear.reset_lora_parameters)
    assert "kaiming_uniform_" in source
    assert "normal_" not in source


def test_gaussian_default_would_have_reached_the_engine_unnoticed():
    """``lora_init_weights`` defaults to PEFT's gaussian, which VeOmni cannot honor."""
    field = DiffusionModelConfig.__dataclass_fields__["lora_init_weights"]
    assert field.default == "gaussian"


# --------------------------------------------------------------------------
# export-side guards
# --------------------------------------------------------------------------


def test_export_rejects_a_named_rollout_adapter(monkeypatch):
    """``rollout_adapter=old`` would otherwise raise a bare KeyError mid-training."""
    engine = make_veomni_engine(_lora_module(), lora_rank=4, target_modules=["to_q", "to_k"])

    with pytest.raises(NotImplementedError, match="'default' adapter only"):
        export_veomni_params(engine, monkeypatch, base_sync_done=True, adapter_name="old")


def test_export_allows_repeated_adapter_syncs(monkeypatch):
    """The steady-state path: base once, then adapter every step."""
    engine = make_veomni_engine(_lora_module(), lora_rank=4, target_modules=["to_q", "to_k"])

    export_veomni_params(engine, monkeypatch, base_sync_done=False)
    for _ in range(3):
        params, _ = export_veomni_params(engine, monkeypatch, base_sync_done=True)
        assert params
