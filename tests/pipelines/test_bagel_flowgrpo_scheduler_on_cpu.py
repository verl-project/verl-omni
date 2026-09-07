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
"""CPU contract for the BAGEL FlowGRPO scheduler grid on diffusers >= 0.40.

diffusers 0.40 made ``FlowMatchEulerDiscreteScheduler.set_timesteps`` ignore
an explicit ``timesteps=`` argument and store ``sigmas * num_train_timesteps``.
The replay resolves rollout-recorded timesteps by exact value, so the lookup
grid must stay in the normalized [0, 1] space the rollout records.
"""

import pytest
import torch

from verl_omni.pipelines.bagel_flow_grpo.common import BAGEL_TIMESTEP_SHIFT, bagel_time_shift, setup_bagel_sigmas
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler


def _rollout_timesteps(num_steps: int) -> torch.Tensor:
    """Mirror the rollout-side recording: shifted ``linspace(1, 0, N)[:-1]``."""
    t = torch.linspace(1, 0, max(num_steps, 2), dtype=torch.float32)
    return bagel_time_shift(BAGEL_TIMESTEP_SHIFT, t)[:-1]


@pytest.mark.parametrize("num_steps", [1, 4])
def test_setup_bagel_sigmas_keeps_normalized_timestep_grid(num_steps):
    scheduler = FlowMatchSDEDiscreteScheduler()
    setup_bagel_sigmas(scheduler, num_steps)

    recorded = _rollout_timesteps(num_steps)
    assert scheduler.timesteps.dtype == torch.float32
    assert torch.equal(scheduler.timesteps, recorded)


@pytest.mark.parametrize("num_steps", [1, 4])
def test_replay_timestep_lookup_hits_recorded_grid(num_steps):
    scheduler = FlowMatchSDEDiscreteScheduler()
    setup_bagel_sigmas(scheduler, num_steps)

    for expected_index, t in enumerate(_rollout_timesteps(num_steps).tolist()):
        assert scheduler.index_for_timestep(t) == expected_index


def test_timestep_lookup_rejects_train_timestep_scale():
    scheduler = FlowMatchSDEDiscreteScheduler()
    setup_bagel_sigmas(scheduler, 4)

    with pytest.raises(IndexError):
        scheduler.index_for_timestep(float(scheduler.timesteps[0]) * 1000.0)


def test_sample_previous_step_replays_recorded_trajectory():
    torch.manual_seed(0)
    scheduler = FlowMatchSDEDiscreteScheduler()
    num_steps = 4
    setup_bagel_sigmas(scheduler, num_steps)

    recorded = _rollout_timesteps(num_steps)
    batch, seq, ch = 2, 3, 4
    latents = torch.randn(batch, num_steps + 1, seq, ch)
    model_output = torch.randn(batch, seq, ch)
    for step in range(recorded.numel()):
        scheduler.sample_previous_step(
            sample=latents[:, step],
            model_output=model_output,
            timestep=recorded[step].expand(batch),
            prev_sample=latents[:, step + 1],
            sde_type="sde",
            return_logprobs=True,
            return_sqrt_dt=True,
            include_logprob_normalizer=False,
        )
