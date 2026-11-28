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

"""L1 CPU contracts for launcher-seeded SDE windows and two-run replay."""

import torch

from verl_omni.pipelines.request_batch import (
    sample_per_sample_sde_windows,
    sde_window_generator_for_step,
    sde_window_seed_for_step,
)
from verl_omni.pipelines.schedulers.flow_match_sde import FlowMatchSDEDiscreteScheduler


def _window_for_step(
    base_seed: int,
    global_steps: int,
    *,
    batch_size: int = 4,
) -> list[tuple[int, int]]:
    generator = torch.Generator().manual_seed(sde_window_seed_for_step(base_seed, global_steps))
    return sample_per_sample_sde_windows(
        sde_window_size=3,
        sde_window_range=(0, 5),
        num_timesteps=10,
        batch_size=batch_size,
        generator=generator,
        device="cpu",
    )


def test_window_seed_advances_by_step_and_clamps_at_first_step():
    assert sde_window_seed_for_step(42, 1) == 42
    assert sde_window_seed_for_step(42, 3) == 44
    assert sde_window_seed_for_step(42, 0) == 42
    assert sde_window_seed_for_step(None, 5) is None


def test_prompt_group_shares_one_window_per_step():
    windows = _window_for_step(7, 5)
    assert windows == [windows[0]] * len(windows)


def test_window_is_replayable_across_runs_and_batch_sizes():
    assert _window_for_step(7, 5) == _window_for_step(7, 5)
    assert _window_for_step(7, 5, batch_size=1) == _window_for_step(7, 5, batch_size=8)[:1]


def test_window_generator_draws_from_the_step_seed_on_cpu():
    generator, device = sde_window_generator_for_step(
        {"sde_window_seed": 7, "global_steps": 5},
        fallback=[torch.Generator().manual_seed(1)],
        fallback_device="cuda:0",
    )
    expected = torch.Generator().manual_seed(11)

    assert device == "cpu"
    assert (
        torch.randint(0, 100, (4,), generator=generator).tolist()
        == torch.randint(0, 100, (4,), generator=expected).tolist()
    )


def test_window_generator_keeps_the_fallback_without_an_explicit_seed():
    fallback = [torch.Generator().manual_seed(1)]

    for extra_args in ({"sde_window_seed": None}, {}):
        generator, device = sde_window_generator_for_step(
            extra_args,
            fallback=fallback,
            fallback_device="cuda:0",
        )
        assert generator is fallback
        assert device == "cuda:0"


def _rollout_once(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    window: tuple[int, int],
):
    scheduler = FlowMatchSDEDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    scheduler.set_timesteps(10, device="cpu")
    return scheduler.step(
        model_output=model_output,
        timestep=scheduler.timesteps[window[0]],
        sample=sample,
        generator=torch.Generator(device="cpu").manual_seed(1234),
        noise_level=0.7,
        sde_type="sde",
        return_logprobs=True,
    )


def test_two_runs_replay_latents_and_log_probs():
    torch.manual_seed(0)
    sample = torch.randn(4, 8, 16, dtype=torch.float32)
    model_output = torch.randn(4, 8, 16, dtype=torch.float32)
    window = _window_for_step(7, 5)[0]

    first = _rollout_once(sample, model_output, window)
    second = _rollout_once(sample, model_output, window)

    assert torch.equal(first.prev_sample, second.prev_sample)
    assert torch.equal(first.log_prob, second.log_prob)
