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

from types import SimpleNamespace

import torch
from vllm_omni.diffusion.data import DiffusionOutput

from verl_omni.pipelines.diffusion_rollout_output import (
    rollout_output,
    with_rollout_data,
    wrap_rollout_postprocessor,
)
from verl_omni.pipelines.request_batch import split_diffusion_output_by_request


def test_rollout_output_uses_native_trajectory_and_metadata_fields() -> None:
    latents = torch.randn(2, 3, 4)
    timesteps = torch.randn(2, 2)
    log_probs = torch.randn(2, 2)
    prompt_embeds = torch.randn(2, 5, 6)
    latents_clean = torch.randn(2, 4)

    output = rollout_output(
        media=torch.randn(2, 3, 8, 8),
        trajectory_latents=latents,
        trajectory_timesteps=timesteps,
        trajectory_log_probs=log_probs,
        prompt_embeddings={"prompt_embeds": prompt_embeds, "negative_prompt_embeds": None},
        rl={"latents_clean": latents_clean},
    )

    torch.testing.assert_close(output.trajectory_latents, latents)
    torch.testing.assert_close(output.trajectory_timesteps, timesteps)
    torch.testing.assert_close(output.trajectory_log_probs, log_probs)
    assert "trajectory" not in output.output["payload"]
    assert "custom_output" not in output.__dataclass_fields__
    torch.testing.assert_close(output.output["metadata"]["prompt_embeddings"]["prompt_embeds"], prompt_embeds)
    assert output.output["metadata"]["prompt_embeddings"]["negative_prompt_embeds"] is None
    torch.testing.assert_close(output.output["metadata"]["rl"]["latents_clean"], latents_clean)


def test_with_rollout_data_preserves_base_output_fields() -> None:
    base = DiffusionOutput(
        output=torch.ones(1, 3, 4, 4),
        stage_durations={"decode": 1.5},
        finished=False,
        chunk_index=2,
        total_chunks=3,
    )
    latents = torch.randn(1, 2, 4)

    output = with_rollout_data(
        base,
        trajectory_latents=latents,
        prompt_embeddings={"prompt_embeds": torch.randn(1, 5, 6)},
    )

    torch.testing.assert_close(output.trajectory_latents, latents)
    assert output.stage_durations == {"decode": 1.5}
    assert output.finished is False
    assert output.chunk_index == 2
    assert output.total_chunks == 3


def test_wrap_rollout_postprocessor_preserves_metadata() -> None:
    def media_postprocess(media: torch.Tensor) -> torch.Tensor:
        return media + 1

    postprocess = wrap_rollout_postprocessor(media_postprocess)
    output = postprocess(
        {
            "payload": {"image": torch.zeros(1)},
            "metadata": {"rl": {"latents_clean": torch.ones(1)}},
        }
    )

    torch.testing.assert_close(output["payload"]["image"], torch.ones(1))
    torch.testing.assert_close(output["metadata"]["rl"]["latents_clean"], torch.ones(1))


def test_request_batch_split_slices_native_envelope_recursively() -> None:
    output = rollout_output(
        media=torch.arange(8).view(2, 4),
        trajectory_latents=torch.arange(16).view(2, 2, 4),
        prompt_embeddings={"prompt_embeds": torch.arange(6).view(2, 3)},
        rl={"latents_clean": torch.arange(8).view(2, 4)},
    )

    split = split_diffusion_output_by_request(
        output,
        SimpleNamespace(num_reqs=2),
        num_outputs_per_prompt=1,
    )

    assert len(split) == 2
    assert split[0].trajectory_latents.shape == (1, 2, 4)
    assert split[1].output["payload"]["image"].tolist() == [[4, 5, 6, 7]]
    assert split[0].output["metadata"]["prompt_embeddings"]["prompt_embeds"].tolist() == [[0, 1, 2]]
    assert split[1].output["metadata"]["rl"]["latents_clean"].tolist() == [[4, 5, 6, 7]]
