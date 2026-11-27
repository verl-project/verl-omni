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
"""L1 CPU contract tests for FlowMatchSDEDiscreteScheduler fp32 and parity invariants."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

_SCHEDULER_PATH = Path(__file__).resolve().parents[2] / "verl_omni" / "pipelines" / "schedulers" / "flow_match_sde.py"


def _load_scheduler_cls():
    try:
        from verl_omni.pipelines.schedulers.flow_match_sde import FlowMatchSDEDiscreteScheduler

        return FlowMatchSDEDiscreteScheduler
    except ModuleNotFoundError:
        spec = importlib.util.spec_from_file_location("flow_match_sde_contract_under_test", _SCHEDULER_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module.FlowMatchSDEDiscreteScheduler


FlowMatchSDEDiscreteScheduler = _load_scheduler_cls()


def _make_scheduler(num_inference_steps: int, shift: float = 1.0) -> FlowMatchSDEDiscreteScheduler:
    scheduler = FlowMatchSDEDiscreteScheduler(num_train_timesteps=1000, shift=shift)
    scheduler.set_timesteps(num_inference_steps, device="cpu")
    return scheduler


@pytest.mark.parametrize("sde_type", ["sde", "cps", "dance_sde"])
@pytest.mark.parametrize("sample_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("step_idx", [0, 1, -1])
@pytest.mark.parametrize(("num_inference_steps", "shift"), [(6, 1.0), (20, 3.0)])
def test_step_preserves_fp32_trajectory_tensors_regardless_of_sample_dtype(
    sde_type: str, sample_dtype: torch.dtype, step_idx: int, num_inference_steps: int, shift: float
) -> None:
    scheduler = _make_scheduler(num_inference_steps=num_inference_steps, shift=shift)
    timestep = scheduler.timesteps[step_idx]
    sample = torch.randn(2, 4, 8, dtype=sample_dtype)
    model_output = torch.randn(2, 4, 8, dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(1234)

    out = scheduler.step(
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        generator=generator,
        noise_level=0.7,
        sde_type=sde_type,
        return_logprobs=True,
    )

    assert out.prev_sample.dtype == torch.float32
    assert out.prev_sample_mean.dtype == torch.float32
    assert out.log_prob is not None and out.log_prob.dtype == torch.float32
    assert torch.isfinite(out.prev_sample).all()
    assert torch.isfinite(out.log_prob).all()


@pytest.mark.parametrize("sde_type", ["sde", "cps", "dance_sde"])
@pytest.mark.parametrize("step_idx", [0, 1, -1])
def test_sample_previous_step_returns_finite_fp32_sqrt_dt(sde_type: str, step_idx: int) -> None:
    scheduler = _make_scheduler(num_inference_steps=20, shift=3.0)
    prev_sample, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = scheduler.sample_previous_step(
        sample=torch.randn(2, 4, 8, dtype=torch.float32),
        model_output=torch.randn(2, 4, 8, dtype=torch.float32),
        timestep=scheduler.timesteps[step_idx].expand(2),
        generator=torch.Generator(device="cpu").manual_seed(1234),
        noise_level=0.7,
        sde_type=sde_type,
        return_logprobs=True,
        return_sqrt_dt=True,
    )
    assert prev_sample.dtype == torch.float32
    assert prev_sample_mean.dtype == torch.float32
    assert log_prob is not None and log_prob.dtype == torch.float32
    assert std_dev_t.dtype == torch.float32
    assert sqrt_dt.dtype == torch.float32
    assert torch.isfinite(prev_sample).all()
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(sqrt_dt).all()


@pytest.mark.parametrize("bad_dtype", [torch.bfloat16, torch.float16])
def test_step_rejects_non_fp32_model_output(bad_dtype: torch.dtype) -> None:
    scheduler = _make_scheduler(num_inference_steps=6)
    with pytest.raises(AssertionError):
        scheduler.step(
            model_output=torch.randn(2, 4, 8, dtype=bad_dtype),
            timestep=scheduler.timesteps[1],
            sample=torch.randn(2, 4, 8, dtype=torch.bfloat16),
            noise_level=0.7,
            return_logprobs=True,
        )


@pytest.mark.parametrize("bad_arg", ["sample", "model_output", "prev_sample"])
@pytest.mark.parametrize("bad_dtype", [torch.bfloat16, torch.float16])
def test_sample_previous_step_rejects_non_fp32_inputs(bad_arg: str, bad_dtype: torch.dtype) -> None:
    scheduler = _make_scheduler(num_inference_steps=6)
    kwargs = {
        "sample": torch.randn(2, 4, 8, dtype=torch.float32),
        "model_output": torch.randn(2, 4, 8, dtype=torch.float32),
        "prev_sample": torch.randn(2, 4, 8, dtype=torch.float32),
        "timestep": scheduler.timesteps[1].expand(2),
    }
    kwargs[bad_arg] = kwargs[bad_arg].to(bad_dtype)

    with pytest.raises(AssertionError):
        scheduler.sample_previous_step(**kwargs)


@pytest.mark.parametrize("sde_type", ["sde", "cps", "dance_sde"])
@pytest.mark.parametrize("num_inference_steps", [5, 13])
def test_rollout_to_actor_replay_matches_log_prob_with_unit_ratio(sde_type: str, num_inference_steps: int) -> None:
    rollout_scheduler = _make_scheduler(num_inference_steps=num_inference_steps)
    actor_scheduler = _make_scheduler(num_inference_steps=num_inference_steps)

    torch.manual_seed(42)
    sample = torch.randn(3, 8, 16, dtype=torch.float32)
    model_output = torch.randn(3, 8, 16, dtype=torch.float32)
    timestep = rollout_scheduler.timesteps[1]

    rollout_out = rollout_scheduler.step(
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        generator=torch.Generator(device="cpu").manual_seed(7),
        noise_level=0.7,
        sde_type=sde_type,
        return_logprobs=True,
    )
    _, actor_log_prob, _, _ = actor_scheduler.sample_previous_step(
        sample=sample,
        model_output=model_output,
        timestep=timestep.expand(sample.shape[0]),
        noise_level=0.7,
        prev_sample=rollout_out.prev_sample,
        sde_type=sde_type,
        return_logprobs=True,
    )

    torch.testing.assert_close(actor_log_prob, rollout_out.log_prob, rtol=0.0, atol=1e-6)
    ratio = torch.exp(actor_log_prob - rollout_out.log_prob)
    ppo_kl = torch.mean(rollout_out.log_prob - actor_log_prob)
    torch.testing.assert_close(ratio.mean(), torch.tensor(1.0), rtol=0.0, atol=1e-6)
    assert abs(ppo_kl.item()) < 1e-6


def test_bf16_truncation_of_stored_latents_breaks_step1_ratio_parity() -> None:
    """Negative control reproducing the common_pitfalls.md bf16 latent-truncation bias."""
    rollout_scheduler = _make_scheduler(num_inference_steps=10)
    actor_scheduler = _make_scheduler(num_inference_steps=10)

    torch.manual_seed(99)
    sample = torch.randn(4, 16, 16, dtype=torch.float32)
    model_output = torch.randn(4, 16, 16, dtype=torch.float32)
    timestep = rollout_scheduler.timesteps[2]

    rollout_out = rollout_scheduler.step(
        model_output=model_output,
        timestep=timestep,
        sample=sample,
        generator=torch.Generator(device="cpu").manual_seed(11),
        noise_level=0.7,
        sde_type="sde",
        return_logprobs=True,
    )
    truncated_prev = rollout_out.prev_sample.to(torch.bfloat16).to(torch.float32)
    _, biased_actor_log_prob, _, _ = actor_scheduler.sample_previous_step(
        sample=sample,
        model_output=model_output,
        timestep=timestep.expand(sample.shape[0]),
        noise_level=0.7,
        prev_sample=truncated_prev,
        sde_type="sde",
        return_logprobs=True,
    )

    assert not torch.allclose(biased_actor_log_prob, rollout_out.log_prob, rtol=0.0, atol=1e-5)
