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
"""Lightweight deterministic reward for the Wan2.2 NPU smoke test."""

import numpy as np
import torch


def compute_score(solution_image, **kwargs) -> float:
    """Return a deterministic content-dependent score without model weights."""
    del kwargs
    if isinstance(solution_image, torch.Tensor):
        values = solution_image.detach().to(dtype=torch.float32)
    else:
        values = torch.as_tensor(np.asarray(solution_image), dtype=torch.float32)
    if values.numel() == 0:
        return 0.0
    return float(values.mean().item() / 255.0)
