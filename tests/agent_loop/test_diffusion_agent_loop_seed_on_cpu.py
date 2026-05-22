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

"""CPU-only regression tests for per-step / per-rollout seed handling.

These tests guard against two previously-observed regressions:

1. The agent loop forwarded a single shared ``seed`` to all ``rollout.n``
   group members of the same prompt. The shared seed caused identical
   initial noise and SDE noise across the group, which collapsed
   group-relative advantages to zero and produced ``grad_norm`` values
   around 1e-7.
2. The base rollout seed was constant across training steps, which made
   every rollout step start from the same initial latent and produced
   correlated trajectories across steps.

The first set of tests exercises the pure ``_derive_rollout_seed`` helper.
The second set drives a ``torch.Generator``-based RNG with derived seeds to
verify the *actual* behaviour the trainer relies on:

- determinism of the starting latent noise (same seed -> same noise);
- diversity of the starting latent noise across rollouts within one step;
- variation of the starting latent noise across training steps for the
  same rollout index;
- diversity of the *full* SDE noise sequence (initial latent + per-step
  variance noise) across rollouts inside a single step.
"""

import torch

from verl_omni.agent_loop.diffusion_agent_loop import (
    _MAX_SEED,
    _PER_ROLLOUT_SEED_STRIDE,
    _derive_rollout_seed,
)

# ---------------------------------------------------------------------------
# Pure-helper tests: properties of ``_derive_rollout_seed``.
# ---------------------------------------------------------------------------


def test_derive_seed_unique_within_group():
    """Within a single step, every rollout in the batch must see a distinct seed."""
    base_seed = 42
    n_rollouts = 16
    prompts = 32
    seeds = [_derive_rollout_seed(base_seed, i) for i in range(prompts * n_rollouts)]
    assert len(set(seeds)) == len(seeds), "per-rollout seeds collided within the batch"


def test_derive_seed_deterministic_across_runs():
    """Same (base_seed, index) must produce the same seed across calls / runs."""
    for base in [0, 1, 42, 12345, 1 << 30]:
        for i in [0, 1, 7, 16, 511, 4096]:
            assert _derive_rollout_seed(base, i) == _derive_rollout_seed(base, i)


def test_derive_seed_changes_across_steps():
    """Different per-step base seeds (data.seed + global_step) must yield disjoint
    per-rollout seed streams in practice.

    We don't claim global injectivity, only that two different base seeds
    cannot make the WHOLE rollout-index range collide.  The check is:
    seeds for base=B differ at every index from seeds for base=B+1.
    """
    base_a, base_b = 100, 101
    for i in range(0, 512):
        assert _derive_rollout_seed(base_a, i) != _derive_rollout_seed(base_b, i)


def test_derive_seed_within_int64():
    """Derived seeds must fit in a signed 64-bit integer for safe forwarding
    through ``torch.Generator.manual_seed`` and pickling."""
    big_base = (1 << 60) - 7
    for i in [0, 1, 1024, _PER_ROLLOUT_SEED_STRIDE - 1]:
        s = _derive_rollout_seed(big_base, i)
        assert 0 <= s < _MAX_SEED


def test_derive_seed_first_index_close_to_base():
    """Sanity check: the first rollout (index 0) is offset from the base by
    only the ``base * stride`` factor, so the result is reproducible by a
    spot-check formula below."""
    assert _derive_rollout_seed(42, 0) == (42 * _PER_ROLLOUT_SEED_STRIDE) % _MAX_SEED
    assert _derive_rollout_seed(42, 5) == (42 * _PER_ROLLOUT_SEED_STRIDE + 5) % _MAX_SEED


def test_derive_seed_handles_zero_base():
    """``base_seed == 0`` is degenerate but should not crash; rollout indices
    must still be distinct."""
    seeds = [_derive_rollout_seed(0, i) for i in range(16)]
    assert seeds == list(range(16))
    assert len(set(seeds)) == 16


# ---------------------------------------------------------------------------
# RNG-driven tests: simulate the rollout adapter behaviour.
#
# The rollout adapter creates ``torch.Generator(...).manual_seed(seed)`` and
# uses it for both the initial latent draw and per-step SDE noise. These
# tests reproduce that flow on CPU to verify the seeding scheme exposes the
# right amount of determinism *and* diversity.
# ---------------------------------------------------------------------------


def _per_step_base_seed(data_seed: int, global_step: int) -> int:
    """Mirror trainer logic: ``data.seed + global_step - 1`` (1-indexed steps)."""
    return int(data_seed) + int(global_step) - 1


def _draw_initial_latent(seed: int, shape=(1, 4, 8, 8)) -> torch.Tensor:
    """Reproduce the per-rollout initial-latent draw used by the rollout
    adapter (`torch.Generator(...).manual_seed(seed)` then `randn`)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=gen)


def _draw_sde_trajectory(seed: int, num_steps: int, shape=(1, 4, 8, 8)) -> list[torch.Tensor]:
    """Reproduce a multi-step SDE noise sequence drawn from a single
    generator (initial latent + per-step variance noise)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return [torch.randn(*shape, generator=gen) for _ in range(num_steps)]


def test_initial_latent_deterministic_for_same_seed():
    """Same (data_seed, global_step, rollout_index) -> identical initial latent."""
    data_seed, step, i = 42, 7, 3
    seed = _derive_rollout_seed(_per_step_base_seed(data_seed, step), i)
    latent_a = _draw_initial_latent(seed)
    latent_b = _draw_initial_latent(seed)
    assert torch.equal(latent_a, latent_b), "initial latent must be deterministic for a fixed seed"


def test_initial_latent_diverse_within_one_rollout_step():
    """Within one step, distinct rollout indices must produce distinct latents.

    Mirrors the original bug: when every group member shared the same seed,
    all latents were identical and group-relative advantages collapsed.
    """
    data_seed, step = 42, 1
    base = _per_step_base_seed(data_seed, step)
    n_rollouts = 16
    latents = [_draw_initial_latent(_derive_rollout_seed(base, i)) for i in range(n_rollouts)]

    for i in range(n_rollouts):
        for j in range(i + 1, n_rollouts):
            assert not torch.equal(latents[i], latents[j]), (
                f"rollouts {i} and {j} have identical initial latent; group-relative advantage will collapse"
            )

    flat = torch.stack([latent.flatten() for latent in latents])
    pairwise_eq = (flat.unsqueeze(0) == flat.unsqueeze(1)).all(dim=-1)
    assert pairwise_eq.sum().item() == n_rollouts, "expected only the diagonal to match exactly"


def test_initial_latent_changes_across_training_steps():
    """For the same rollout index, the initial latent must change across
    training steps so the model sees a fresh starting point each step."""
    data_seed, i = 42, 3
    latents = [
        _draw_initial_latent(_derive_rollout_seed(_per_step_base_seed(data_seed, step), i)) for step in range(1, 17)
    ]
    for s in range(len(latents)):
        for t in range(s + 1, len(latents)):
            assert not torch.equal(latents[s], latents[t]), (
                f"initial latent identical for steps {s + 1} and {t + 1}; rollouts will not diversify across steps"
            )


def test_sde_trajectory_deterministic_and_diverse_within_step():
    """The full per-rollout SDE noise *trajectory* (initial latent + every
    per-step variance noise drawn from the same generator) must be:

    - deterministic for a fixed seed (so checkpoints + retries reproduce);
    - distinct across rollout indices at every step (so each group member
      explores a different SDE path, not just a different initial latent).
    """
    data_seed, step = 42, 5
    base = _per_step_base_seed(data_seed, step)
    n_rollouts, num_sde_steps = 8, 4

    trajectories = [
        _draw_sde_trajectory(_derive_rollout_seed(base, i), num_steps=num_sde_steps) for i in range(n_rollouts)
    ]

    for i in range(n_rollouts):
        seed_i = _derive_rollout_seed(base, i)
        replayed = _draw_sde_trajectory(seed_i, num_steps=num_sde_steps)
        for t, (orig, repl) in enumerate(zip(trajectories[i], replayed, strict=False)):
            assert torch.equal(orig, repl), f"rollout {i} step {t} not reproducible from the same seed"

    for t in range(num_sde_steps):
        for i in range(n_rollouts):
            for j in range(i + 1, n_rollouts):
                assert not torch.equal(trajectories[i][t], trajectories[j][t]), (
                    f"rollouts {i} and {j} produced identical noise at SDE step {t}"
                )


def test_validation_seed_is_independent_of_training_seed():
    """The agent loop uses ``config.val_kwargs.seed`` (a constant) for
    validation runs, *not* the per-step ``rollout_seed``. That makes
    validation reproducible without re-using the training-step RNG stream.

    This test pins the contract that the validation seed -> latent mapping
    is stable across training steps, while the training-side mapping is
    not.
    """
    val_seed = 1234
    val_latent_a = _draw_initial_latent(val_seed)
    val_latent_b = _draw_initial_latent(val_seed)
    assert torch.equal(val_latent_a, val_latent_b)

    data_seed = 42
    train_seed_step_1 = _derive_rollout_seed(_per_step_base_seed(data_seed, 1), 0)
    train_seed_step_2 = _derive_rollout_seed(_per_step_base_seed(data_seed, 2), 0)
    assert train_seed_step_1 != train_seed_step_2
    assert not torch.equal(_draw_initial_latent(train_seed_step_1), _draw_initial_latent(train_seed_step_2))
