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
"""Score assemblers for verl-omni rollouts."""

import torch
from verl import DataProto


def visual_score_assembler(data: DataProto, scores: list) -> torch.Tensor:
    """Score assembler for image rollouts.

    Visual rollouts emit one reward per generated image, so the resulting
    ``rm_scores`` tensor has shape ``(batch_size, 1)``.
    """
    return torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)
