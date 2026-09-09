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
"""Deterministic reward used only by the MiniMax-H3 special E2E smoke test.

MiniMax-H3 produces joint video/audio outputs. Existing lightweight image-only
rewards such as JPEG compressibility do not accept that output, while CLAP and
ImageBind introduce model-weight and service dependencies inappropriate for a
smoke test. This scorer provides stable, content-dependent values without
external I/O; it verifies the reward and update path, not output quality.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch


def _stable_score(solution: torch.Tensor | np.ndarray) -> float:
    """Return a deterministic pseudo-random score in ``[0, 1)``."""
    if isinstance(solution, torch.Tensor):
        flattened = solution.detach().to(torch.float32).flatten()
    else:
        flattened = torch.from_numpy(np.asarray(solution, dtype=np.float32)).flatten()
    if flattened.numel() == 0:
        return 0.0

    payload = f"{tuple(getattr(solution, 'shape', ()))}|{float(flattened[0]):.6f}|{float(flattened[-1]):.6f}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return float(int.from_bytes(digest[:4], "big") / 2**32)


def compute_score(solution_image, *_, **__) -> float:
    """Return a deterministic smoke-test score for an H3 rollout output."""
    return _stable_score(solution_image)
