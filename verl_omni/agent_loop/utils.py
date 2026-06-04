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

from typing import Optional


def _derive_rollout_seed(base_seed: int, rollout_index: int) -> int:
    """Map per-step rollout base and expanded row index to a vLLM seed.
    Row index is 0 .. num_prompts * rollout.n - 1 after interleaved repeat."""
    max_seed = (1 << 63) - 1
    return (int(base_seed) * 1_000_003 + int(rollout_index)) % max_seed


def maybe_per_rollout_seeds(meta_info: dict, batch_size: int, global_indices=None) -> Optional[list[int]]:
    """Build one seed per post-repeat rollout row."""
    base = meta_info.get("rollout_seed")
    if base is None:
        return None
    base = int(base)

    if global_indices is None:
        return [_derive_rollout_seed(base, i) for i in range(batch_size)]

    if len(global_indices) != batch_size:
        raise ValueError(f"Expected {batch_size} global rollout indices, got {len(global_indices)}")

    return [_derive_rollout_seed(base, int(idx)) for idx in global_indices]
