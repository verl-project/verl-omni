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
"""Ascend forward consistency and gradients for the MiniMax H3 rollout kernels."""

from types import SimpleNamespace

import pytest
import torch

from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from peft import LoraConfig, inject_adapter_in_model
from vllm_omni.diffusion.layers.indexed_modulation import (
    indexed_gate,
    indexed_gate_rms_norm_scale_shift,
    rms_norm_indexed_scale_shift,
)

from tests.pipelines.test_minimax_h3_image_diffusion_nft_on_cpu import TestForward as CPUForwardHelper
from tests.pipelines.test_minimax_h3_image_diffusion_nft_on_cpu import _micro_batch
from verl_omni.pipelines.minimax_h3_image_diffusion_nft.common import install_pdd_heads
from verl_omni.pipelines.minimax_h3_image_diffusion_nft.diffusers_training_adapter import MiniMaxH3ImageDiffusionNFT
from verl_omni.pipelines.minimax_h3_image_diffusion_nft.rollout_kernels import CALLS, install_rollout_kernels


@pytest.fixture
def actor(tmp_path):
    """Build a small mixed-precision H3 Actor with its six PDD heads."""
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.npu.set_device(0)
    torch.manual_seed(42)
    model = MiniMaxH3Transformer3DModel(
        num_attention_heads=2,
        attention_head_dim=128,
        hidden_size=256,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=384,
        text_dim=24,
        freq_dim=16,
        time_embed_hidden_dim=64,
        time_embed_dim=32,
        rope_freq_dim=16,
    )
    inject_adapter_in_model(LoraConfig(r=2, lora_alpha=4, target_modules=["to_q", "ff.net.2"]), model)
    model.to(device="npu", dtype=torch.bfloat16)
    model.time_embedder.float()
    model.rope.float()
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.05)
    install_pdd_heads(
        model,
        torch.randn(6, 96, 256, device="npu", dtype=torch.float32),
        torch.randn(6, 96, device="npu", dtype=torch.float32),
    )
    model.set_attention_backend("_native_npu")
    install_rollout_kernels(model)
    return model, SimpleNamespace(config_path=str(tmp_path), local_path=None, path=None)


def test_npu_block_matches_rollout_forward_and_keeps_lora_gradients(actor):
    """Match the rollout block numerically and retain the LoRA gradient."""
    model, _ = actor
    block = model.transformer_blocks[0]
    hidden = torch.randn(1, 37, 256, device="npu", dtype=torch.bfloat16, requires_grad=True)
    temb = torch.randn(2, 32, device="npu", dtype=torch.float32)
    indices = torch.arange(37, device="npu") % 6
    rotary = model.rope(torch.arange(111, device="npu").reshape(37, 3).float())
    calls = CALLS["kernel"]
    actual = block(hidden, temb, indices, rotary)
    assert CALLS["kernel"] > calls

    with torch.no_grad():
        shift, scale, gate, shift_ff, scale_ff, gate_ff = [x.contiguous() for x in block.adaln_proj(temb)]
        rows = hidden[0].detach().contiguous()
        modulated = rms_norm_indexed_scale_shift(rows, block.norm1.weight, shift, scale, indices, block.norm1.eps)
        attended = block.attn(modulated.unsqueeze(0), rotary)[0].contiguous()
        residual, modulated = indexed_gate_rms_norm_scale_shift(
            rows, gate, attended, block.norm2.weight, shift_ff, scale_ff, indices, block.norm2.eps
        )
        expected = indexed_gate(residual, gate_ff, block.ff(modulated.unsqueeze(0))[0].contiguous(), indices)
    assert torch.equal(actual[0], expected)
    actual.float().square().mean().backward()
    gradient = block.attn.to_q.lora_B["default"].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize("step", range(6))
def test_npu_nft_adapter_all_pdd_heads_have_finite_gradients(actor, monkeypatch, step):
    """Exercise the normal NFT adapter path for each release timestep."""
    model, config = actor
    monkeypatch.setattr(MiniMaxH3ImageDiffusionNFT, "use_rollout_kernels", True)
    micro = _micro_batch(1, forward_noise=torch.randn(1, 6 * 96))
    inputs = CPUForwardHelper()._inputs(micro, step, model, config)
    inputs = {key: value.to("npu") if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    for key in ["image_rows", "condition_rows", "encoder_hidden_states"]:
        inputs[key] = inputs[key].to(torch.bfloat16)
    prediction = MiniMaxH3ImageDiffusionNFT.forward(model, config, inputs)
    assert prediction.shape == (1, 6 * 96) and torch.isfinite(prediction).all()
    prediction.float().square().mean().backward()
    gradient = model.transformer_blocks[0].attn.to_q.lora_B["default"].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
