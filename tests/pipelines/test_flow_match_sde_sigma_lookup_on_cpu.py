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

"""CPU tests for the vectorized sigma-index lookup on the SDE training hot path.

`FlowMatchSDEDiscreteScheduler.sample_previous_step` (training log-prob replay)
and `verl_omni.pipelines.utils.get_sigmas` (DPO noisy-latent prep) used to run
a per-row `index_for_timestep` / `nonzero().item()` loop, which hard-syncs the
device once per batch element per denoising step. Both now use a vectorized
first-match lookup against the schedule. These tests pin the new lookup to the
legacy semantics.
"""

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from verl_omni.pipelines.schedulers.flow_match_sde import FlowMatchSDEDiscreteScheduler
from verl_omni.pipelines.utils import get_sigmas

SDE_TYPES = ["sde", "cps", "dance_sde"]


def _make_scheduler(shift: float = 3.0, num_inference_steps: int = 10) -> FlowMatchSDEDiscreteScheduler:
    scheduler = FlowMatchSDEDiscreteScheduler(num_train_timesteps=1000, shift=shift)
    scheduler.set_timesteps(num_inference_steps)
    return scheduler


def _legacy_index_lookup(scheduler, timesteps: torch.Tensor) -> torch.Tensor:
    """The pre-optimization per-row lookup, kept as the behavioral reference."""
    return torch.tensor([scheduler.index_for_timestep(t) for t in timesteps])


@pytest.mark.parametrize("shift", [1.0, 3.0])
@pytest.mark.parametrize("num_inference_steps", [5, 10, 33])
def test_vectorized_lookup_matches_legacy_indices(shift, num_inference_steps):
    scheduler = _make_scheduler(shift=shift, num_inference_steps=num_inference_steps)
    generator = torch.Generator().manual_seed(0)
    # Rows may repeat and share timesteps, but never select the final schedule
    # entry (sigma_prev would be out of bounds — same constraint as before).
    ks = torch.randint(0, num_inference_steps - 1, (9,), generator=generator)
    timesteps = scheduler.timesteps[ks]

    matches = scheduler.timesteps.reshape(1, -1) == timesteps.reshape(-1, 1)
    sigma_idx = matches.to(torch.int64).argmax(dim=1)

    assert torch.equal(sigma_idx, _legacy_index_lookup(scheduler, timesteps))


def _run_step(scheduler, sample, model_output, prev_sample, sde_type, timestep=None):
    returns = scheduler.sample_previous_step(
        sample=sample,
        model_output=model_output,
        timestep=timestep,
        generator=None,
        noise_level=0.7,
        prev_sample=prev_sample,
        sde_type=sde_type,
        return_logprobs=True,
        return_sqrt_dt=True,
    )
    prev, log_prob, mean, std_dev_t, sqrt_dt = returns
    # The timestep-indexed branch returns per-row (B, 1, ...) tensors while the
    # sequential branch broadcasts 0-dim ones; normalize for comparison.
    std_dev_t = std_dev_t.flatten()
    if std_dev_t.numel() == 1:
        std_dev_t = std_dev_t.expand(sample.shape[0]).clone()
    return prev, log_prob, mean, std_dev_t, sqrt_dt


def test_uniform_timestep_rows_match_sequential_branch():
    """All rows at schedule[k] must reproduce the sequential step_index=k path."""
    for sde_type in SDE_TYPES:
        scheduler = _make_scheduler()
        torch.manual_seed(0)
        sample = torch.randn(3, 4, 8, 8)
        model_output = torch.randn(3, 4, 8, 8)
        prev_sample = torch.randn(3, 4, 8, 8)

        for k in range(len(scheduler.timesteps) - 1):
            step_index_before = scheduler._step_index
            seq_scheduler = _make_scheduler()
            seq_scheduler._step_index = k
            sequential = _run_step(seq_scheduler, sample, model_output, prev_sample, sde_type)
            uniform_ts = scheduler.timesteps[k : k + 1].expand(3)
            indexed = _run_step(scheduler, sample, model_output, prev_sample, sde_type, timestep=uniform_ts)
            assert scheduler._step_index == step_index_before, "indexed branch must not touch step_index"
            for got, want in zip(indexed, sequential, strict=True):
                torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-7)


def test_mixed_timestep_rows_match_per_row_calls():
    """Rows at different schedule positions each get their own sigma."""
    scheduler = _make_scheduler()
    torch.manual_seed(1)
    ks = torch.tensor([3, 0, 5, 5, 8, 1, 7])
    timesteps = scheduler.timesteps[ks]
    batch = ks.numel()
    sample = torch.randn(batch, 4, 8, 8)
    model_output = torch.randn(batch, 4, 8, 8)
    prev_sample = torch.randn(batch, 4, 8, 8)

    prev, log_prob, mean, std_dev_t, sqrt_dt = _run_step(
        scheduler, sample, model_output, prev_sample, "sde", timestep=timesteps
    )

    for i in range(batch):
        row = _run_step(
            scheduler,
            sample[i : i + 1],
            model_output[i : i + 1],
            prev_sample[i : i + 1],
            "sde",
            timestep=timesteps[i : i + 1],
        )
        torch.testing.assert_close(log_prob[i : i + 1], row[1], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(mean[i : i + 1], row[2], rtol=1e-6, atol=1e-7)
        assert torch.allclose(std_dev_t[i : i + 1], row[3], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(sqrt_dt[i : i + 1], row[4], rtol=1e-6, atol=1e-7)
        # prev_sample is echoed back unchanged when provided.
        assert torch.equal(prev[i : i + 1], row[0])


def test_lookup_rejects_off_schedule_timestep():
    scheduler = _make_scheduler()
    bogus = scheduler.timesteps[:1] + 0.5
    # torch._assert raises AssertionError on CPU and RuntimeError on CUDA devices.
    with pytest.raises((AssertionError, RuntimeError)):
        scheduler.sample_previous_step(
            sample=torch.randn(2, 4, 8, 8),
            model_output=torch.randn(2, 4, 8, 8),
            timestep=bogus.expand(2),
            prev_sample=torch.randn(2, 4, 8, 8),
            sde_type="sde",
        )


def test_get_sigmas_matches_legacy_lookup():
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    scheduler.set_timesteps(10)
    generator = torch.Generator().manual_seed(0)
    ks = torch.randint(0, 9, (6,), generator=generator)
    timesteps = scheduler.timesteps[ks]

    sigma = get_sigmas(scheduler, timesteps, device=timesteps.device, n_dim=4, dtype=torch.float32)

    reference = scheduler.sigmas.to(dtype=torch.float32)[_legacy_index_lookup(scheduler, timesteps)]
    assert sigma.shape == (6, 1, 1, 1)
    torch.testing.assert_close(sigma.flatten(), reference.flatten(), rtol=0, atol=0)
