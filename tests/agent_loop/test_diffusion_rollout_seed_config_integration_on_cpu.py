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

"""CPU-only config-to-meta_info integration tests for the rollout seed contract.

These tests verify that the config-level ``data.seed`` flows correctly
through the trainer's per-step ``rollout_seed`` computation and into the
agent loop's per-rollout seed expansion — all without instantiating a
trainer, model, or vLLM-Omni server.

The contract under test:

1. ``_build_rollout_seed(data_seed, global_steps)`` — trainer-side helper
   that converts ``data.seed`` + 1-indexed step into a per-step base seed.
2. ``_maybe_per_rollout_seeds(meta_info, batch_size)`` — agent-loop helper
   that expands the base seed into per-rollout seeds.
3. The combined pipeline: config → meta_info → sampling_params seeds — as
   a pure function of (data_seed, global_steps, batch_size).
"""

import numpy as np

from verl_omni.agent_loop.diffusion_agent_loop import (
    _build_rollout_seed,
    _derive_rollout_seed,
    _maybe_per_rollout_seeds,
)


# ---------------------------------------------------------------------------
# _build_rollout_seed: trainer-side helper properties
# ---------------------------------------------------------------------------


def test_build_rollout_seed_first_step_uses_base_seed():
    """Step 1 (global_steps=1) produces base = data_seed (not data_seed+1)."""
    assert _build_rollout_seed(1234, 1) == 1234


def test_build_rollout_seed_second_step_increments_once():
    """Step 2 -> base = data_seed + 1."""
    assert _build_rollout_seed(1234, 2) == 1235


def test_build_rollout_seed_later_step():
    """Step 100 -> base = data_seed + 99."""
    assert _build_rollout_seed(42, 100) == 141


def test_build_rollout_seed_none_disables():
    """data_seed=None returns None (seeding disabled)."""
    assert _build_rollout_seed(None, 1) is None
    assert _build_rollout_seed(None, 100) is None


def test_build_rollout_seed_zero_is_valid():
    """data_seed=0 is a valid seed; step 1 base = 0."""
    assert _build_rollout_seed(0, 1) == 0
    assert _build_rollout_seed(0, 5) == 4


def test_build_rollout_seed_numpy_int_is_coerced():
    """numpy.int64 data_seed is coerced to int (safe for meta_info)."""
    assert _build_rollout_seed(np.int64(42), 1) == 42
    assert _build_rollout_seed(np.int64(42), 3) == 44


def test_build_rollout_seed_deterministic():
    """Same (data_seed, step) -> same result across repeated calls."""
    for data_seed in [1, 42, 1234, None]:
        for step in [1, 7, 100]:
            assert _build_rollout_seed(data_seed, step) == _build_rollout_seed(data_seed, step)


# ---------------------------------------------------------------------------
# Config-to-meta_info-to-per-rollout-seeds pipeline
# ---------------------------------------------------------------------------


def _config_to_per_rollout_seeds(data_seed, global_steps, batch_size):
    """End-to-end pipeline mirroring trainer + agent loop without side effects.

    Equivalent to:
      trainer:  meta_info["rollout_seed"] = _build_rollout_seed(data_seed, global_steps)
      agent:    per_rollout_seeds = _maybe_per_rollout_seeds(meta_info, batch_size)
    """
    rollout_seed = _build_rollout_seed(data_seed, global_steps)
    if rollout_seed is None:
        meta_info = {}
    else:
        meta_info = {"rollout_seed": rollout_seed}
    return _maybe_per_rollout_seeds(meta_info, batch_size)


def test_pipeline_data_seed_none_no_seeds():
    """When data.seed=null, the full pipeline produces no per-rollout seeds."""
    assert _config_to_per_rollout_seeds(None, 1, batch_size=8) is None


def test_pipeline_first_step_unique_seeds():
    """data_seed=42, step=1, batch=8 -> 8 distinct seeds matching _derive_rollout_seed."""
    seeds = _config_to_per_rollout_seeds(42, 1, batch_size=8)
    assert seeds is not None
    assert len(seeds) == 8
    assert len(set(seeds)) == 8, "per-rollout seeds collided"
    expected = [_derive_rollout_seed(42, i) for i in range(8)]
    assert seeds == expected


def test_pipeline_step_increment_changes_all_seeds():
    """Seeds at step N+1 must differ from seeds at step N for all rollout indices."""
    seeds_step1 = _config_to_per_rollout_seeds(42, 1, batch_size=16)
    seeds_step2 = _config_to_per_rollout_seeds(42, 2, batch_size=16)
    for i in range(16):
        assert seeds_step1[i] != seeds_step2[i], (
            f"rollout index {i} has same seed across steps 1 and 2"
        )


def test_pipeline_seeds_deterministic():
    """Same (data_seed, step, batch_size) -> identical seed list."""
    a = _config_to_per_rollout_seeds(42, 7, batch_size=16)
    b = _config_to_per_rollout_seeds(42, 7, batch_size=16)
    assert a == b


def test_pipeline_large_step_no_overflow():
    """Large global_steps (e.g. step 100000) must still produce in-bounds seeds."""
    seeds = _config_to_per_rollout_seeds(42, 100_000, batch_size=4)
    assert seeds is not None
    for s in seeds:
        assert 0 <= s < (1 << 63) - 1


# ---------------------------------------------------------------------------
# sampling_params construction contract
# ---------------------------------------------------------------------------


def test_sampling_params_seed_override_contract():
    """Simulate the agent loop's per-rollout sampling_params construction.

    This is what ``DiffusionAgentLoopWorker.generate_sequences()`` does:
    - build per-rollout seeds from meta_info
    - for each batch item, create a shallow copy of sampling_params
    - override ``seed`` with the per-rollout value
    """
    meta_info = {"rollout_seed": 42, "global_steps": 1}
    batch_size = 4

    per_rollout_seeds = _maybe_per_rollout_seeds(meta_info, batch_size)
    assert per_rollout_seeds is not None

    base_sampling_params = {
        "height": 512,
        "width": 512,
        "num_inference_steps": 10,
        "global_steps": 1,
    }

    task_params_list = []
    for i in range(batch_size):
        task_params = dict(base_sampling_params)
        task_params["seed"] = per_rollout_seeds[i]
        task_params_list.append(task_params)

    assert len(task_params_list) == batch_size
    for i, tp in enumerate(task_params_list):
        assert tp["seed"] == _derive_rollout_seed(42, i)
        assert tp["height"] == 512
        assert tp["global_steps"] == 1

    seeds_in_params = [tp["seed"] for tp in task_params_list]
    assert len(set(seeds_in_params)) == batch_size, "per-rollout seeds in sampling_params collided"


def test_sampling_params_seed_not_injected_when_disabled():
    """When _maybe_per_rollout_seeds returns None, sampling_params is NOT mutated."""
    meta_info = {}  # no rollout_seed
    per_rollout_seeds = _maybe_per_rollout_seeds(meta_info, batch_size=4)
    assert per_rollout_seeds is None

    base_sampling_params = {"height": 512, "num_inference_steps": 10}
    assert "seed" not in base_sampling_params
