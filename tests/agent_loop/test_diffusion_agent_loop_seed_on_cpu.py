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

"""CPU unit tests for rollout seed helpers.

GPU integration coverage lives in ``test_diffusion_rollout_seed_gpu.py``.
"""

import warnings

import pytest

from verl_omni.agent_loop.diffusion_agent_loop import (
    _build_rollout_seed,
    _derive_rollout_seed,
    _maybe_per_rollout_seeds,
)


def test_build_rollout_seed_uses_rollout_seed():
    assert _build_rollout_seed(42, global_steps=1) == 42
    assert _build_rollout_seed(42, global_steps=3) == 44


def test_build_rollout_seed_null_disables():
    assert _build_rollout_seed(None, global_steps=1) is None
    assert _build_rollout_seed(None, global_steps=1, data_seed=None) is None


def test_build_rollout_seed_prefers_rollout_seed_over_data_seed():
    assert _build_rollout_seed(7, global_steps=2, data_seed=99) == 8


def test_build_rollout_seed_falls_back_to_data_seed_with_warning():
    with pytest.warns(DeprecationWarning, match="actor_rollout_ref.rollout.seed"):
        assert _build_rollout_seed(None, global_steps=5, data_seed=42) == 46


def test_build_rollout_seed_warns_only_when_data_seed_used():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _build_rollout_seed(42, global_steps=1, data_seed=99) == 42
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_derive_seed_unique_within_group():
    base_seed = 42
    seeds = [_derive_rollout_seed(base_seed, i) for i in range(32 * 16)]
    assert len(set(seeds)) == len(seeds)


def test_disable_seed_returns_none_when_rollout_seed_absent():
    assert _maybe_per_rollout_seeds({}, batch_size=16) is None
    assert _maybe_per_rollout_seeds({"global_steps": 5}, batch_size=16) is None


def test_disable_seed_returns_none_when_rollout_seed_is_null():
    assert _maybe_per_rollout_seeds({"rollout_seed": None}, batch_size=16) is None


def test_enable_seed_returns_unique_per_rollout_seeds():
    seeds = _maybe_per_rollout_seeds({"rollout_seed": 42}, batch_size=8)
    assert seeds is not None
    assert len(set(seeds)) == 8
    assert seeds == [_derive_rollout_seed(42, i) for i in range(8)]
