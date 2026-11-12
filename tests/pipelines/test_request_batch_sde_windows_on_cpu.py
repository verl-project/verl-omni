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

"""CPU tests for per-sample SDE window sampling under request packing."""

import torch

from verl_omni.agent_loop.utils import derive_rollout_seed
from verl_omni.pipelines.request_batch import sample_per_sample_sde_windows, shared_sde_window_generator


def _advanced_generator(seed: int, latent_numel: int = 8) -> torch.Generator:
    generator = torch.Generator().manual_seed(seed)
    _ = torch.randn(latent_numel, generator=generator)
    return generator


def test_per_sample_sde_windows_match_serial_and_ignore_pack_order():
    seeds = [11, 22, 33]
    serial = [
        sample_per_sample_sde_windows(
            sde_window_size=2,
            sde_window_range=(0, 5),
            num_timesteps=10,
            batch_size=1,
            generator=_advanced_generator(seed),
            device="cpu",
        )[0]
        for seed in seeds
    ]

    packed_order = [2, 0, 1]
    packed = sample_per_sample_sde_windows(
        sde_window_size=2,
        sde_window_range=(0, 5),
        num_timesteps=10,
        batch_size=len(packed_order),
        generator=[_advanced_generator(seeds[i]) for i in packed_order],
        device="cpu",
    )
    assert packed == [serial[i] for i in packed_order]


def _window_for(generator: torch.Generator) -> tuple[int, int]:
    return sample_per_sample_sde_windows(
        sde_window_size=2,
        sde_window_range=(0, 5),
        num_timesteps=10,
        batch_size=1,
        generator=generator,
        device="cpu",
    )[0]


def test_shared_step_seed_collapses_grpo_windows_without_touching_sample_rng():
    # Wan2.2 recipe, step 1: rollout.seed=42, n=8, window size 2 inside [0, 5).
    base_seed = 42
    per_sample = [
        _window_for(torch.Generator().manual_seed(derive_rollout_seed(base_seed, session))) for session in range(8)
    ]
    assert len(set(per_sample)) > 1

    shared = []
    for session in range(8):
        fallback = torch.Generator().manual_seed(derive_rollout_seed(base_seed, session))
        state = fallback.get_state().clone()
        generator = shared_sde_window_generator({"global_steps": 1}, fallback, device="cpu")
        shared.append(_window_for(generator))
        assert torch.equal(fallback.get_state(), state)
    assert len(set(shared)) == 1

    explicit = _window_for(
        shared_sde_window_generator({"sde_window_seed": 99, "global_steps": 1}, torch.Generator(), device="cpu")
    )
    from_steps = _window_for(shared_sde_window_generator({"global_steps": 99}, torch.Generator(), device="cpu"))
    assert explicit == from_steps
    assert shared_sde_window_generator(None, "fallback", device="cpu") == "fallback"
