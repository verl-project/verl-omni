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

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_seed_utils():
    # Keep this CPU unit test independent of package-level diffusion stack auto-registration.
    utils_path = Path(__file__).resolve().parents[2] / "verl_omni" / "agent_loop" / "utils.py"
    spec = importlib.util.spec_from_file_location("diffusion_seed_utils", utils_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_seed_utils = _load_seed_utils()
_derive_rollout_seed = _seed_utils._derive_rollout_seed
_maybe_per_rollout_seeds = _seed_utils._maybe_per_rollout_seeds


def test_per_rollout_seeds_use_explicit_global_row_ids_across_chunks():
    base_seed = 42
    first_chunk = _maybe_per_rollout_seeds(
        {"rollout_seed": base_seed},
        batch_size=2,
        global_indices=np.array([0, 1], dtype=np.int64),
    )
    second_chunk = _maybe_per_rollout_seeds(
        {"rollout_seed": base_seed},
        batch_size=2,
        global_indices=np.array([2, 3], dtype=np.int64),
    )

    assert first_chunk == [_derive_rollout_seed(base_seed, 0), _derive_rollout_seed(base_seed, 1)]
    assert second_chunk == [_derive_rollout_seed(base_seed, 2), _derive_rollout_seed(base_seed, 3)]
    assert set(first_chunk).isdisjoint(second_chunk)


def test_per_rollout_seeds_keep_worker_local_fallback():
    base_seed = 42

    assert _maybe_per_rollout_seeds({"rollout_seed": base_seed}, batch_size=3) == [
        _derive_rollout_seed(base_seed, 0),
        _derive_rollout_seed(base_seed, 1),
        _derive_rollout_seed(base_seed, 2),
    ]


def test_per_rollout_seeds_disabled_when_rollout_seed_is_unset():
    assert _maybe_per_rollout_seeds({}, batch_size=2, global_indices=np.array([0, 1], dtype=np.int64)) is None


def test_per_rollout_seeds_reject_malformed_global_row_ids():
    with pytest.raises(ValueError, match="Expected 2 global rollout indices, got 1"):
        _maybe_per_rollout_seeds({"rollout_seed": 42}, batch_size=2, global_indices=np.array([0], dtype=np.int64))
