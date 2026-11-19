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
"""Exercise stock H3 Image and PEFT through the training engine's real lifecycle."""

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from peft import get_peft_model_state_dict

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import map_lora_tensors
from verl_omni.pipelines.minimax_h3_flow_grpo.common import configure_flow_scheduler
from verl_omni.pipelines.minimax_h3_image_flow_grpo.diffusers_training_adapter import (
    MiniMaxH3ImageFlowGRPO,
    _actor_model_class,
)
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.engine.lora_adapter_mixin import LoRAAdapterMixin
from verl_omni.workers.engine.lora_export import export_lora_for_sync

diffusers = pytest.importorskip("diffusers", minversion="0.43.0")
MiniMaxH3ImageTransformer2DModel = diffusers.MiniMaxH3ImageTransformer2DModel


def tiny_model():
    """Create a real six-head model using only the released Diffusers constructor."""
    torch.manual_seed(17)
    return _actor_model_class()(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=32,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=48,
        text_dim=24,
        freq_dim=16,
        time_embed_hidden_dim=32,
        time_embed_dim=16,
        rope_freq_dim=2,
        num_pdd_heads=6,
    )


def engine_for(model, adapter_path=None):
    """Build LoRA with the production helper, without distributed initialization."""
    engine = LoRAAdapterMixin()
    engine.model_config = SimpleNamespace(
        lora_rank=2,
        lora_alpha=2,
        lora_init_weights=True,
        target_modules=["to_q", "to_v"],
        target_parameters=None,
        exclude_modules=None,
        policy_state_adapters=["default", "old"],
        lora_adapter_path=adapter_path,
        use_shm=False,
        fsdp_layer_prefixes=["transformer_blocks."],
    )
    model.requires_grad_(False)
    engine.module = engine._build_lora_module(model)
    return engine


def replay(model, head):
    """Replay a transition with the image adapter and the actual native backbone."""
    torch.manual_seed(23)
    scheduler = FlowMatchSDEDiscreteScheduler()
    configure_flow_scheduler(scheduler, [1.0, 0.96752, 0.88755, 0.75, 0.53277, 0.23201, 0.0], "cpu")
    inputs = {
        "rows": torch.randn(1, 4, 96),
        "heads": torch.tensor([head]),
        "grid": (4, 4),
        "selected_sigmas": scheduler.sigmas[head].expand(1),
        "embeddings": torch.randn(1, 3, 24),
        "mask": torch.ones(1, 3),
        "tags": torch.ones(1, 3, dtype=torch.long),
    }
    cfg = SimpleNamespace(algo=SimpleNamespace(noise_level=0.8, sde_type="cps"))
    return MiniMaxH3ImageFlowGRPO.forward_and_sample_previous_step(
        model, scheduler, cfg, inputs, None, {"all_latents": torch.randn(1, 2, 4, 96)}, 0
    )


@pytest.mark.parametrize("head", range(6))
def test_stock_heads_lora_backward_and_no_forward_rebinding(head):
    """All native PDD heads support checkpointed LoRA replay without patched forwards."""
    stock = tiny_model()
    assert hasattr(stock, "add_adapter")
    with torch.no_grad():
        expected = replay(stock, head)[0]
    engine = engine_for(stock)
    assert isinstance(engine.module, MiniMaxH3ImageTransformer2DModel)
    engine.module.enable_gradient_checkpointing()
    engine.module.train()
    score, *_ = replay(engine.module, head)
    torch.testing.assert_close(score, expected, rtol=0, atol=0)
    (-score.mean()).backward()
    grads = [p.grad for p in engine.module.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    assert all(p.device.type == "cpu" for p in engine.module.parameters())
    assert all("forward" not in m.__dict__ for m in engine.module.modules())
    assert all("_stock_forward" not in m.__dict__ for m in engine.module.modules())
    assert stock.forward.__func__ is MiniMaxH3ImageTransformer2DModel.forward
    assert all(p.grad is None for n, p in stock.named_parameters() if ".lora_" not in n)


def test_stock_adapter_copy_disable_restore_and_optimizer():
    """Verify policy isolation and exception-safe restoration using real PEFT layers."""
    engine = engine_for(tiny_model())
    model = engine.module
    baseline = replay(model, 2)[0].detach()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    (-replay(model, 2)[0].mean()).backward()
    optimizer.step()
    changed = replay(model, 2)[0].detach()
    assert not torch.equal(changed, baseline)
    engine.copy_adapter("default", "old")
    with engine.use_adapter("old"):
        torch.testing.assert_close(replay(model, 2)[0], changed)
    assert model.active_adapters() == ["default"]
    with pytest.raises(RuntimeError, match="sentinel"):
        with engine.disable_adapter():
            torch.testing.assert_close(replay(model, 2)[0], baseline)
            raise RuntimeError("sentinel")
    torch.testing.assert_close(replay(model, 2)[0], changed)
    assert model.active_adapters() == ["default"]


def test_stock_adapter_save_load_and_weight_export(tmp_path):
    """Export both policy states through the production collector and reload PEFT files."""
    stock = tiny_model()
    base_state = copy.deepcopy(stock.state_dict())
    engine = engine_for(stock)
    with torch.no_grad():
        for name, param in engine.module.named_parameters():
            if ".lora_B.default." in name:
                param.fill_(0.025)
    engine.copy_adapter("default", "old")
    expected = replay(engine.module, 1)[0].detach()
    for adapter in ("default", "old"):
        stream, config = export_lora_for_sync(
            engine.module,
            model_config=engine.model_config,
            base_sync_done=True,
            adapter_name=adapter,
            is_diffusers=True,
        )
        exported = dict(stream)
        assert exported and config["r"] == 2
        expected_weights = get_peft_model_state_dict(engine.module, adapter_name=adapter)
        assert set(exported) == set(expected_weights)
        for name in exported:
            torch.testing.assert_close(exported[name], expected_weights[name])
        mapped = map_lora_tensors(exported, "transformer", 24, strict=True)
        assert len(mapped) == len(exported)
        assert all(name.startswith(("transformer.blocks.", "transformer.token_refiner.blocks.")) for name in mapped)
    engine.module.save_lora_adapter(tmp_path / "model")
    restored_model = tiny_model()
    restored_model.load_state_dict(base_state)
    restored_model.load_lora_adapter(tmp_path / "model", prefix=None, use_safetensors=True, adapter_name="default")
    torch.testing.assert_close(replay(restored_model, 1)[0], expected)

    # The shared engine loads pipeline-prefixed weights using the default .bin filename.
    stream, _ = export_lora_for_sync(
        engine.module,
        model_config=engine.model_config,
        base_sync_done=True,
        is_diffusers=True,
        key_prefix="transformer.",
    )
    torch.save(dict(stream), tmp_path / "pytorch_lora_weights.bin")
    restored_base = tiny_model()
    restored_base.load_state_dict(base_state)
    restored = engine_for(restored_base, str(tmp_path))
    assert set(restored.module.peft_config) == {"default", "old"}
    assert any(p.requires_grad for p in restored.module.parameters())
    torch.testing.assert_close(replay(restored.module, 1)[0], expected)


def test_layered_collection_keeps_native_prefix(monkeypatch):
    """Mock only collective contexts; collect real tuner weights under both root layouts."""
    import verl.utils.device as device_utils
    from torch.distributed.fsdp import FullyShardedDataParallel

    import verl_omni.utils.fsdp_utils as fsdp_utils

    engine = engine_for(tiny_model())
    model = engine.module
    blocks = set(model.transformer_blocks) | set(model.token_refiner.refiner_blocks)
    monkeypatch.setattr(fsdp_utils, "fsdp_version", lambda module: 2 if module in blocks else 0)
    monkeypatch.setattr(FullyShardedDataParallel, "summon_full_params", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(device_utils, "get_torch_device", lambda: SimpleNamespace(empty_cache=lambda: None))
    result = fsdp_utils._layered_summon_lora_params_diffusers(
        model,
        layer_prefixes=["transformer_blocks.", "token_refiner.refiner_blocks."],
    )
    expected = get_peft_model_state_dict(model)
    assert result and set(result) == set(expected)
    for name in result:
        torch.testing.assert_close(result[name], expected[name])


def test_native_checkpoint_loader_and_peft_interface(tmp_path):
    """The production loader preserves native parameters, outputs and adapter APIs."""
    original = tiny_model()
    original.save_pretrained(tmp_path)
    config = SimpleNamespace(config_path=str(tmp_path), local_path=None, attn_backend="native")
    restored = MiniMaxH3ImageFlowGRPO.build_module(config, torch.float32)
    assert restored.forward.__func__ is MiniMaxH3ImageTransformer2DModel.forward
    assert set(restored.state_dict()) == set(original.state_dict())
    torch.testing.assert_close(replay(restored, 2)[0], replay(original, 2)[0])
    assert hasattr(restored, "add_adapter")


def test_missing_native_class_has_actionable_error(monkeypatch):
    """Older Diffusers installs fail at model construction with a recipe-specific message."""
    import builtins

    original_import = builtins.__import__

    def without_image(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "diffusers" and "MiniMaxH3ImageTransformer2DModel" in fromlist:
            raise ImportError("image model absent")
        return original_import(name, globals, locals, fromlist, level)

    _actor_model_class.cache_clear()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", without_image)
            with pytest.raises(ImportError, match="Diffusers 0.43.0"):
                _actor_model_class()
    finally:
        _actor_model_class.cache_clear()
