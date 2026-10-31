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
"""Ascend port of the H3 offload-placeholder regression with device tensors.

Mirrors ``test_minimax_h3_meta_output.py`` (GPU diffusion E2E ID 11 part 2)
with NPU device placement; the rollout-output fixture and value assertions
are the CPU-module originals.
"""

import pytest
import torch
from test_minimax_h3_rollout_contract_on_cpu import _make_h3_rollout_output


@pytest.mark.parametrize("algorithm", ["diffusion_nft", "flow_grpo"])
def test_meta_placeholder_with_npu_training_inputs(monkeypatch, algorithm):
    import torch_npu  # noqa: F401

    assert torch.npu.is_available()
    placeholder = torch.empty(1, device="meta")
    result = _make_h3_rollout_output(monkeypatch, algorithm, placeholder=placeholder, device="npu")
    assert result.output["payload"]["image"] is placeholder
    metadata = result.output["metadata"]
    assert "media_artifacts" not in metadata
    prompt_embeds = metadata["prompt_embeddings"]["prompt_embeds"]
    output_device = "npu" if algorithm == "diffusion_nft" else "cpu"
    assert prompt_embeds.device.type == output_device
    torch.testing.assert_close(prompt_embeds, torch.ones(1, 5, 8, device=output_device))
    if algorithm == "diffusion_nft":
        latents = metadata["rl"]["latents_clean"]
        expected = torch.cat([torch.ones(1, 192, device="npu"), torch.full((1, 512), 2.0, device="npu")], dim=1)
    else:
        latents = result.trajectory_latents
        expected = torch.ones(1, 1, 704)
    assert latents.device.type == output_device
    torch.testing.assert_close(latents, expected)
