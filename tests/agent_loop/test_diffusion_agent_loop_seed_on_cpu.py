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

"""CPU regression tests for rollout seed derivation and enable/disable contracts.

GPU integration coverage lives in ``test_diffusion_rollout_seed_gpu.py``.
"""

import torch

from verl_omni.agent_loop.diffusion_agent_loop import (
    _derive_rollout_seed,
    _maybe_per_rollout_seeds,
)


def _per_step_base_seed(data_seed: int, global_step: int) -> int:
    """Mirror trainer logic: ``data.seed + global_step - 1`` (1-indexed steps)."""
    return int(data_seed) + int(global_step) - 1


def _draw_initial_latent(seed: int, shape=(1, 4, 8, 8)) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=gen)


# ---------------------------------------------------------------------------
# Seed derivation invariants
# ---------------------------------------------------------------------------


def test_derive_seed_unique_within_group():
    base_seed = 42
    seeds = [_derive_rollout_seed(base_seed, i) for i in range(32 * 16)]
    assert len(set(seeds)) == len(seeds)


def test_derive_seed_changes_across_steps():
    base_a, base_b = 100, 101
    for i in range(512):
        assert _derive_rollout_seed(base_a, i) != _derive_rollout_seed(base_b, i)


def test_initial_latent_diverse_within_one_rollout_step():
    """Distinct rollout indices within one step must not share the same latent."""
    base = _per_step_base_seed(42, 1)
    latents = [_draw_initial_latent(_derive_rollout_seed(base, i)) for i in range(16)]
    for i in range(16):
        for j in range(i + 1, 16):
            assert not torch.equal(latents[i], latents[j])


def test_initial_latent_changes_across_training_steps():
    data_seed, i = 42, 3
    latents = [
        _draw_initial_latent(_derive_rollout_seed(_per_step_base_seed(data_seed, step), i)) for step in range(1, 17)
    ]
    for s in range(len(latents)):
        for t in range(s + 1, len(latents)):
            assert not torch.equal(latents[s], latents[t])


def test_validation_seed_is_independent_of_training_seed():
    val_latent_a = _draw_initial_latent(1234)
    val_latent_b = _draw_initial_latent(1234)
    assert torch.equal(val_latent_a, val_latent_b)

    train_seed_step_1 = _derive_rollout_seed(_per_step_base_seed(42, 1), 0)
    train_seed_step_2 = _derive_rollout_seed(_per_step_base_seed(42, 2), 0)
    assert train_seed_step_1 != train_seed_step_2
    assert not torch.equal(_draw_initial_latent(train_seed_step_1), _draw_initial_latent(train_seed_step_2))


# ---------------------------------------------------------------------------
# Enable / disable contract for ``_maybe_per_rollout_seeds``
# ---------------------------------------------------------------------------


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


def test_enable_seed_accepts_numpy_int_like_base():
    seeds = _maybe_per_rollout_seeds({"rollout_seed": "42"}, batch_size=4)
    assert seeds == [_derive_rollout_seed(42, i) for i in range(4)]


def test_initial_latent_different_across_dp_ranks():
    """DP ranks own disjoint global-index slices and must see distinct seeds."""
    base = _per_step_base_seed(42, 1)
    dp_size, local_batch = 4, 4
    rank_seeds = [
        _maybe_per_rollout_seeds({"rollout_seed": base}, batch_size=dp_size * local_batch)[
            r * local_batch : (r + 1) * local_batch
        ]
        for r in range(dp_size)
    ]
    flat_seeds = [s for seeds in rank_seeds for s in seeds]
    assert len(set(flat_seeds)) == len(flat_seeds)
