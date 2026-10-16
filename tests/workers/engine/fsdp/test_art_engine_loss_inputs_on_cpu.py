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
"""CPU test for the per-step loss inputs the DiffusionART engine hands to the loss.

Necessity: the first GPU run crashed here (a non-tensor round trip of ``num_train_timesteps`` read without the
``default`` the pinned verl requires); the computation is now a pure static method over the micro batch, so
``sigma_next``, the optional keys and the rollout-velocity finite difference are pinned without an FSDP engine.
"""

import torch
from tensordict import TensorDict

from verl_omni.workers.engine.fsdp.diffusers_impl import ARTDiffusersFSDPEngine

B, T, TOKENS, C = 3, 2, 6, 4


def make_micro_batch(with_optional: bool) -> TensorDict:
    data = {
        "train_latents": torch.randn(B, T, TOKENS, C),
        "train_timesteps": torch.tensor([[900.0, 500.0]] * B),
        "train_timesteps_next": torch.tensor([[500.0, 0.0]] * B),
        "art_guidance": torch.randn(B, TOKENS, C),
    }
    if with_optional:
        data["train_latents_next"] = torch.randn(B, T, TOKENS, C)
        data["video_seq_len"] = torch.full((B,), 4, dtype=torch.long)
        data["art_anchor"] = torch.tensor([False, True, False])
        data["_art_metric::reward/clap/mean"] = torch.full((B,), 0.25)
    return TensorDict(data, batch_size=[B])


def flow_time(micro_batch: TensorDict, step: int) -> torch.Tensor:
    return (micro_batch["train_timesteps"][:, step] / 1000.0).view(B, 1, 1)


def test_loss_inputs_sigma_next_and_rollout_velocity_from_scheduler_count():
    micro_batch = make_micro_batch(with_optional=True)
    xt = micro_batch["train_latents"][:, 0]
    inputs = ARTDiffusersFSDPEngine.art_loss_inputs(micro_batch, 0, xt, flow_time(micro_batch, 0), 1000)
    assert set(inputs) == {
        "art_guidance",
        "sigma_next",
        "video_seq_len",
        "art_anchor",
        "rollout_velocity",
        "_art_metric::reward/clap/mean",
    }
    torch.testing.assert_close(inputs["sigma_next"], torch.full((B,), 0.5))
    expected_velocity = (micro_batch["train_latents_next"][:, 0] - xt) / (0.5 - 0.9)
    torch.testing.assert_close(inputs["rollout_velocity"], expected_velocity)
    assert inputs["art_guidance"] is micro_batch["art_guidance"]
    # Last ODE step: sigma_next is 0 and the finite difference uses the final step size.
    last = ARTDiffusersFSDPEngine.art_loss_inputs(
        micro_batch, 1, micro_batch["train_latents"][:, 1], flow_time(micro_batch, 1), 1000
    )
    torch.testing.assert_close(last["sigma_next"], torch.zeros(B))


def test_loss_inputs_omit_optional_keys_when_absent():
    micro_batch = make_micro_batch(with_optional=False)
    inputs = ARTDiffusersFSDPEngine.art_loss_inputs(
        micro_batch, 0, micro_batch["train_latents"][:, 0], flow_time(micro_batch, 0), 1000
    )
    assert set(inputs) == {"art_guidance", "sigma_next"}


def test_loss_inputs_pass_per_segment_flow_times_when_the_rollout_published_them():
    """MiniMax-H3 denoises video and audio on different sigma grids: ``train_segment_timesteps`` (B, S, 2) becomes
    ``segment_sigma`` / ``segment_sigma_next`` (B, 2) in flow-time units, sliced at the current step."""
    micro_batch = make_micro_batch(with_optional=True)
    micro_batch["train_segment_timesteps"] = torch.tensor([[[900.0, 450.0], [500.0, 250.0]]] * B)
    micro_batch["train_segment_timesteps_next"] = torch.tensor([[[500.0, 250.0], [0.0, 0.0]]] * B)
    inputs = ARTDiffusersFSDPEngine.art_loss_inputs(
        micro_batch, 0, micro_batch["train_latents"][:, 0], flow_time(micro_batch, 0), 1000
    )
    torch.testing.assert_close(inputs["segment_sigma"], torch.tensor([[0.9, 0.45]] * B))
    torch.testing.assert_close(inputs["segment_sigma_next"], torch.tensor([[0.5, 0.25]] * B))
    # The video column equals the shared flow time, so single-grid models are unaffected by the extra keys.
    torch.testing.assert_close(inputs["segment_sigma"][:, 0], flow_time(micro_batch, 0).reshape(B))
    last = ARTDiffusersFSDPEngine.art_loss_inputs(
        micro_batch, 1, micro_batch["train_latents"][:, 1], flow_time(micro_batch, 1), 1000
    )
    torch.testing.assert_close(last["segment_sigma_next"], torch.zeros(B, 2))
