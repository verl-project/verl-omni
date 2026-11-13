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
"""Image-only packing and zero-adapter identity on a real Ascend NPU."""

import pytest
import torch
from peft.tuners.tuners_utils import BaseTunerLayer

from tests.pipelines.test_minimax_h3_image_diffusion_nft_on_cpu import TestForward as CPUForwardHelper
from tests.pipelines.test_minimax_h3_image_diffusion_nft_on_cpu import _micro_batch
from tests.pipelines.test_minimax_h3_image_rollout_kernels_npu import actor  # noqa: F401
from verl_omni.pipelines.minimax_h3_image_diffusion_nft.common import build_image_layout
from verl_omni.pipelines.minimax_h3_image_diffusion_nft.diffusers_training_adapter import MiniMaxH3ImageDiffusionNFT


def inputs_for(model, config, step):
    micro = _micro_batch(1, forward_noise=torch.randn(1, 6 * 96))
    inputs = CPUForwardHelper()._inputs(micro, step, model, config)
    inputs = {key: value.to("npu") if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    for key in ["image_rows", "condition_rows", "encoder_hidden_states"]:
        inputs[key] = inputs[key].to(torch.bfloat16)
    return inputs


def test_image_only_packing_preserves_every_projected_row(actor, monkeypatch):
    """An empty audio stream must preserve the actual text and image embeddings."""
    model, config = actor
    monkeypatch.setattr(MiniMaxH3ImageDiffusionNFT, "use_rollout_kernels", True)
    inputs = inputs_for(model, config, 0)
    seen = {}

    def image_output(module, args, output):
        seen["image"] = output.detach().to(torch.bfloat16).cpu().clone()

    def text_output(module, args, output):
        seen["text"] = output.detach().cpu().clone()

    def block_input(module, args):
        seen["packed"] = args[0].detach().cpu().clone()

    handles = [
        model.proj_in.register_forward_hook(image_output),
        model.token_refiner.register_forward_hook(text_output),
        model.transformer_blocks[0].register_forward_pre_hook(block_input),
    ]
    try:
        with torch.no_grad():
            MiniMaxH3ImageDiffusionNFT.forward(model, config, inputs)
    finally:
        for handle in handles:
            handle.remove()
    _, _, image_indices, text_indices, ncond = build_image_layout(*inputs["latent_grid"], seen["text"].shape[1])
    assert ncond == 0
    expected = torch.zeros_like(seen["packed"])
    expected.index_copy_(1, text_indices, seen["text"])
    expected.index_copy_(1, image_indices, seen["image"])
    assert torch.equal(seen["packed"], expected)


@pytest.mark.parametrize("step", range(6))
def test_zero_lora_matches_disabled_adapter_for_every_head(actor, monkeypatch, step):
    """Zero LoRA must leave the complete Actor prediction unchanged."""
    model, config = actor
    monkeypatch.setattr(MiniMaxH3ImageDiffusionNFT, "use_rollout_kernels", True)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.zero_()
    inputs = inputs_for(model, config, step)
    with torch.no_grad():
        enabled = MiniMaxH3ImageDiffusionNFT.forward(model, config, inputs)
    layers = [child for child in model.modules() if isinstance(child, BaseTunerLayer)]
    for child in layers:
        child.enable_adapters(False)
    try:
        with torch.no_grad():
            disabled = MiniMaxH3ImageDiffusionNFT.forward(model, config, inputs)
    finally:
        for child in layers:
            child.enable_adapters(True)
    assert torch.equal(enabled, disabled)
