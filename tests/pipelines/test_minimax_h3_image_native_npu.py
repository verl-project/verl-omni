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
"""Exercise native image preparation and PDD forward/backward on an Ascend device."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl_omni.pipelines.minimax_h3_image_diffusion_nft.common import (
    image_latent_meta,
    pdd_dit_timestep,
    pdd_step_indices,
    pdd_train_timesteps,
)
from verl_omni.pipelines.minimax_h3_image_diffusion_nft.diffusers_training_adapter import MiniMaxH3ImageDiffusionNFT


def test_native_device_preparation_all_heads_and_backward(monkeypatch):
    """Start with NPU tensors so neither lookup nor model-time conversion bypasses the device path."""
    pytest.importorskip("torch_npu")
    from verl_omni.pipelines.minimax_h3_image_diffusion_nft.diffusers_training_adapter import _actor_model_class
    from verl_omni.workers.engine.fsdp.diffusers_impl import _cast_loaded_diffusers_module
    from verl_omni.workers.engine.lora_adapter_mixin import LoRAAdapterMixin

    assert torch.npu.is_available()
    monkeypatch.setattr(MiniMaxH3ImageDiffusionNFT, "pack_micro_batch", False)
    sigmas_cpu = torch.tensor(
        [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132]
    )
    sigmas = sigmas_cpu.to("npu")
    expected_indices = torch.arange(6)
    # These endpoints round to a gap of 2.0 after scaling both to FP32 timesteps.
    close_grid = torch.tensor([0.9675236940383911, 0.9655236601829529], device="npu")
    assert pdd_step_indices(close_grid * 1000, close_grid).tolist() == [0, 1]
    with pytest.raises(ValueError, match="decrease by more than"):
        pdd_step_indices(close_grid[:1] * 1000, close_grid + torch.tensor([0.0, 1e-6], device="npu"))
    for timesteps in (pdd_train_timesteps(sigmas_cpu), pdd_train_timesteps(sigmas_cpu).long()):
        torch.testing.assert_close(pdd_step_indices(timesteps.to("npu"), sigmas).cpu(), expected_indices)
    torch.testing.assert_close(pdd_dit_timestep(sigmas).cpu(), pdd_dit_timestep(sigmas_cpu), rtol=0, atol=0)
    actor = _actor_model_class()(
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
    )
    _cast_loaded_diffusers_module(actor, torch.bfloat16)
    actor.set_attention_backend("_native_npu")
    actor.requires_grad_(False)
    engine = LoRAAdapterMixin()
    engine.model_config = SimpleNamespace(
        lora_rank=2,
        lora_alpha=4,
        lora_init_weights=True,
        target_modules=["to_q", "to_v"],
        target_parameters=None,
        exclude_modules=None,
        policy_state_adapters=["default", "old"],
        lora_adapter_path=None,
    )
    engine.module = engine._build_lora_module(actor).to("npu")
    assert actor.transformer_blocks[0].attn.to_q.weight.dtype == torch.bfloat16
    assert actor.pdd_proj_out[0].weight.dtype == actor.proj_in.weight.dtype == torch.float32
    actor.enable_gradient_checkpointing()
    engine.module.train()
    batch = TensorDict(
        {
            "latents_clean": torch.randn(1, 4 * 96, device="npu"),
            "forward_noise": torch.randn(1, 4 * 96, device="npu"),
            "train_timesteps": pdd_train_timesteps(sigmas_cpu).long().unsqueeze(0).to("npu"),
            "pdd_sigmas": sigmas.unsqueeze(0),
            "latent_meta": image_latent_meta(4, 4, 4).to("npu"),
        },
        batch_size=1,
    )
    scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
    for step in range(6):
        engine.module.zero_grad(set_to_none=True)
        noisy, timesteps, _ = MiniMaxH3ImageDiffusionNFT.prepare_training_inputs(batch, step, scheduler)
        inputs, _ = MiniMaxH3ImageDiffusionNFT.prepare_model_inputs(
            engine.module,
            engine.model_config,
            noisy,
            timesteps,
            torch.randn(1, 3, 24, device="npu"),
            torch.ones(1, 3, device="npu"),
            None,
            None,
            batch,
            step,
        )
        assert inputs["timestep"].device.type == "npu"
        assert inputs["pdd_step"].item() == step
        prediction = MiniMaxH3ImageDiffusionNFT.forward(engine.module, engine.model_config, inputs)
        assert prediction.shape == (1, 4 * 96) and torch.isfinite(prediction).all()
        prediction.square().mean().backward()
        grad = actor.transformer_blocks[0].attn.to_q.lora_B["default"].weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        assert all(parameter.grad is None for name, parameter in actor.named_parameters() if ".default." not in name)
