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

import torch

from verl_omni.pipelines.utils import seeded_or_entropy_generator


def test_seeded_generator_matches_manual_seed() -> None:
    """A seeded generator draws exactly what a manual_seed generator draws."""
    assert torch.equal(
        torch.randn(4, generator=seeded_or_entropy_generator(7, "cpu")),
        torch.randn(4, generator=torch.Generator(device="cpu").manual_seed(7)),
    )


def test_unseeded_generators_draw_distinct_noise() -> None:
    """Unseeded generators derive OS entropy, so separate draws never collide.

    Device-default RNG state is identical across worker processes on torch_npu,
    which is what this helper exists to avoid.
    """
    first = seeded_or_entropy_generator(None, "cpu")
    second = seeded_or_entropy_generator(None, "cpu")
    assert not torch.equal(torch.randn(4, generator=first), torch.randn(4, generator=second))
