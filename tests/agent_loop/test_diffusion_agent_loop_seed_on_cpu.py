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

"""CPU-only regression tests for the per-rollout seed derivation helper.

These tests guard against a previously-observed regression in which the
agent loop forwarded a single shared ``seed`` to all ``rollout.n`` group
members of the same prompt.  The shared seed caused identical initial
noise and SDE noise across the group, which collapsed group-relative
advantages to zero and produced ``grad_norm`` values around 1e-7.
"""

from verl_omni.agent_loop.diffusion_agent_loop import (
    _MAX_SEED,
    _PER_ROLLOUT_SEED_STRIDE,
    _derive_rollout_seed,
)


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
    must still be distinct.

    (In practice ``data.seed=0`` is unusual; we usually default to 42 when
    unset.  Test the corner anyway.)
    """
    seeds = [_derive_rollout_seed(0, i) for i in range(16)]
    assert seeds == list(range(16))
    assert len(set(seeds)) == 16
