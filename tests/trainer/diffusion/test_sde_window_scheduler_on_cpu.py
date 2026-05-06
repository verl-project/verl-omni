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
"""CPU tests for the SDE-window scheduler.

These tests pin down the FlowGRPO baseline and the two MixGRPO sliding-window
strategies (random / progressive) so future refactors keep the scheduling
semantics intact.
"""

from __future__ import annotations

import pytest

from verl_omni.trainer.diffusion.sde_window_scheduler import (
    FlowGRPOWindowScheduler,
    MixGRPOProgressiveScheduler,
    MixGRPORandomScheduler,
    _resolve_envelope,
    build_sde_window_scheduler,
)
from verl_omni.workers.config.diffusion.rollout import DiffusionRolloutAlgoConfig


# ---------------------------------------------------------------------------
# FlowGRPO baseline -- static overrides
# ---------------------------------------------------------------------------


class TestFlowGRPOWindowScheduler:
    def test_returns_static_window(self):
        sched = FlowGRPOWindowScheduler(sde_window_size=2, sde_window_range=[0, 5])
        for step in [0, 1, 100]:
            ovr = sched.get_window(step)
            assert ovr["sde_window_size"] == 2
            assert ovr["sde_window_range"] == [0, 5]

    def test_none_size_propagates_none(self):
        sched = FlowGRPOWindowScheduler(sde_window_size=None, sde_window_range=None)
        ovr = sched.get_window(0)
        # ``None`` size signals "use entire trajectory" -- nothing forwarded.
        assert ovr == {}


# ---------------------------------------------------------------------------
# MixGRPO -- progressive
# ---------------------------------------------------------------------------


class TestMixGRPOProgressiveScheduler:
    def test_window_advances_every_iters_per_group(self):
        sched = MixGRPOProgressiveScheduler(
            group_size=4, init_timestep=0, max_timestep=14, iters_per_group=2
        )
        assert sched.get_window(0)["sde_window_range"] == [0, 4]
        assert sched.get_window(1)["sde_window_range"] == [0, 4]
        assert sched.get_window(2)["sde_window_range"] == [4, 8]
        assert sched.get_window(3)["sde_window_range"] == [4, 8]
        assert sched.get_window(4)["sde_window_range"] == [8, 12]

    def test_window_clips_at_max(self):
        sched = MixGRPOProgressiveScheduler(
            group_size=4, init_timestep=0, max_timestep=14, iters_per_group=1
        )
        # With group_size=4 and max_timestep=14, max valid window_start = 11.
        assert sched.get_window(2)["sde_window_range"] == [8, 12]
        assert sched.get_window(3)["sde_window_range"] == [11, 15]
        assert sched.get_window(100)["sde_window_range"] == [11, 15]

    def test_window_is_deterministic_at_rollout(self):
        # Rollout backend draws ``start`` from ``[lo, hi - size + 1)``.
        # We collapse ``hi - lo == size`` to make the draw deterministic.
        sched = MixGRPOProgressiveScheduler(
            group_size=3, init_timestep=0, max_timestep=10, iters_per_group=1
        )
        ovr = sched.get_window(1)
        lo, hi = ovr["sde_window_range"]
        assert hi - lo == ovr["sde_window_size"]

    def test_invalid_iters_per_group_raises(self):
        with pytest.raises(ValueError, match="iters_per_group"):
            MixGRPOProgressiveScheduler(
                group_size=2, init_timestep=0, max_timestep=4, iters_per_group=0
            )

    def test_invalid_group_size_raises(self):
        with pytest.raises(ValueError, match="group_size"):
            MixGRPOProgressiveScheduler(
                group_size=0, init_timestep=0, max_timestep=4
            )

    def test_window_does_not_fit_raises(self):
        with pytest.raises(ValueError, match="does not fit"):
            MixGRPOProgressiveScheduler(
                group_size=10, init_timestep=0, max_timestep=4
            )


# ---------------------------------------------------------------------------
# MixGRPO -- random
# ---------------------------------------------------------------------------


class TestMixGRPORandomScheduler:
    def test_window_size_constant(self):
        sched = MixGRPORandomScheduler(
            group_size=3, init_timestep=0, max_timestep=10, base_seed=7
        )
        for step in range(5):
            ovr = sched.get_window(step)
            assert ovr["sde_window_size"] == 3

    def test_window_in_range(self):
        sched = MixGRPORandomScheduler(
            group_size=3, init_timestep=0, max_timestep=10, base_seed=7
        )
        for step in range(50):
            lo, hi = sched.get_window(step)["sde_window_range"]
            assert 0 <= lo <= 8  # max valid start = 10-3+1 = 8
            assert hi == lo + 3

    def test_deterministic_for_same_seed(self):
        a = MixGRPORandomScheduler(group_size=3, init_timestep=0, max_timestep=10, base_seed=0)
        b = MixGRPORandomScheduler(group_size=3, init_timestep=0, max_timestep=10, base_seed=0)
        for step in range(10):
            assert a.get_window(step)["sde_window_range"] == b.get_window(step)["sde_window_range"]

    def test_different_seeds_diverge(self):
        a = MixGRPORandomScheduler(group_size=3, init_timestep=0, max_timestep=10, base_seed=0)
        b = MixGRPORandomScheduler(group_size=3, init_timestep=0, max_timestep=10, base_seed=999)
        diffs = sum(
            a.get_window(s)["sde_window_range"] != b.get_window(s)["sde_window_range"]
            for s in range(10)
        )
        assert diffs > 0


# ---------------------------------------------------------------------------
# Envelope resolution: sde_window_range -> [init_timestep, max_timestep]
# ---------------------------------------------------------------------------


class TestResolveEnvelope:
    def test_default_uses_full_trajectory_minus_final_step(self):
        assert _resolve_envelope(None, num_inference_steps=10) == (0, 8)

    def test_default_handles_zero_steps(self):
        assert _resolve_envelope(None, num_inference_steps=1) == (0, 0)

    def test_explicit_range_is_used(self):
        # ``[lo, hi)`` half-open in YAML -> inclusive ``max_timestep`` here.
        assert _resolve_envelope([2, 8], num_inference_steps=50) == (2, 7)

    def test_explicit_range_clipped_to_trajectory_limit(self):
        # [0, 10) with 10 steps: hi-1=9 but trajectory max is 8 (final ODE step excluded).
        assert _resolve_envelope([0, 10], num_inference_steps=10) == (0, 8)

    def test_invalid_length_raises(self):
        with pytest.raises(ValueError, match="must be a list of two ints"):
            _resolve_envelope([0, 1, 2], num_inference_steps=10)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestBuildSDEWindowScheduler:
    def test_default_is_flow_grpo(self):
        cfg = DiffusionRolloutAlgoConfig()
        sched = build_sde_window_scheduler(cfg, num_inference_steps=10)
        assert isinstance(sched, FlowGRPOWindowScheduler)

    def test_mix_grpo_random(self):
        cfg = DiffusionRolloutAlgoConfig(
            algo_type="mix_grpo",
            sample_strategy="random",
            sde_window_size=4,
        )
        sched = build_sde_window_scheduler(cfg, num_inference_steps=10)
        assert isinstance(sched, MixGRPORandomScheduler)

    def test_mix_grpo_progressive(self):
        cfg = DiffusionRolloutAlgoConfig(
            algo_type="mix_grpo",
            sample_strategy="progressive",
            sde_window_size=4,
            iters_per_group=2,
        )
        sched = build_sde_window_scheduler(cfg, num_inference_steps=16)
        assert isinstance(sched, MixGRPOProgressiveScheduler)
        assert sched.iters_per_group == 2

    def test_mix_grpo_envelope_defaults_from_num_inference_steps(self):
        """When ``sde_window_range`` is null, the envelope is the full
        trajectory minus the final ODE step."""
        cfg = DiffusionRolloutAlgoConfig(
            algo_type="mix_grpo",
            sample_strategy="progressive",
            sde_window_size=4,
            iters_per_group=1,
            sde_window_range=None,
        )
        sched = build_sde_window_scheduler(cfg, num_inference_steps=16)
        assert sched.init_timestep == 0
        assert sched.max_timestep == 14

    def test_mix_grpo_envelope_from_sde_window_range(self):
        """Setting ``sde_window_range`` overrides the envelope."""
        cfg = DiffusionRolloutAlgoConfig(
            algo_type="mix_grpo",
            sample_strategy="progressive",
            sde_window_size=2,
            iters_per_group=1,
            sde_window_range=[2, 8],
        )
        sched = build_sde_window_scheduler(cfg, num_inference_steps=50)
        assert sched.init_timestep == 2
        assert sched.max_timestep == 7

    def test_mix_grpo_seed_renamed(self):
        """``sde_window_seed`` drives the random draw."""
        cfg = DiffusionRolloutAlgoConfig(
            algo_type="mix_grpo",
            sample_strategy="random",
            sde_window_size=3,
            sde_window_seed=42,
        )
        sched = build_sde_window_scheduler(cfg, num_inference_steps=10)
        assert isinstance(sched, MixGRPORandomScheduler)
        assert sched.base_seed == 42

    def test_mix_grpo_requires_window_size(self):
        with pytest.raises(ValueError, match="sde_window_size"):
            DiffusionRolloutAlgoConfig(algo_type="mix_grpo", sde_window_size=None)

    def test_unknown_algo_type_raises(self):
        with pytest.raises(ValueError):
            DiffusionRolloutAlgoConfig(algo_type="bogus")

    def test_unknown_sample_strategy_raises(self):
        with pytest.raises(ValueError):
            DiffusionRolloutAlgoConfig(
                algo_type="mix_grpo",
                sample_strategy="bogus",
                sde_window_size=4,
            )


