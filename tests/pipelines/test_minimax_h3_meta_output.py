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
"""GPU regressions for H3 offload placeholders with GPU training tensors."""

import pytest
import torch
from test_minimax_h3_rollout_contract_on_cpu import _make_h3_rollout_output
from verl.utils.device import get_device_name, get_torch_device


@pytest.mark.parametrize("algorithm", ["diffusion_nft", "flow_grpo"])
def test_meta_placeholder_with_cuda_training_inputs(monkeypatch, algorithm):
    assert get_torch_device().is_available()
    device = get_device_name()
    placeholder = torch.empty(1, device="meta")
    result = _make_h3_rollout_output(monkeypatch, algorithm, placeholder=placeholder, device=device)
    assert result.output["payload"]["image"] is placeholder
    metadata = result.output["metadata"]
    assert "media_artifacts" not in metadata
    prompt_embeds = metadata["prompt_embeddings"]["prompt_embeds"]
    output_device = device if algorithm == "diffusion_nft" else "cpu"
    assert prompt_embeds.device.type == output_device
    torch.testing.assert_close(prompt_embeds, torch.ones(1, 5, 8, device=output_device))
    if algorithm == "diffusion_nft":
        latents = metadata["rl"]["latents_clean"]
        expected = torch.cat([torch.ones(1, 192, device=device), torch.full((1, 512), 2.0, device=device)], dim=1)
    else:
        latents = result.trajectory_latents
        expected = torch.ones(1, 1, 704)
    assert latents.device.type == output_device
    torch.testing.assert_close(latents, expected)
