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

"""Small, position-sensitive publication diagnostics (not a full-model hash)."""

import math

import torch


def weight_fingerprint(tensor: torch.Tensor) -> float:
    """Fingerprint finite weights with bounded temporary storage.

    Accumulate in float64 and weight positions to detect cancelling changes that a
    plain sum misses. The comparison tolerates normal fp16/bf16 transport rounding;
    changes below that tolerance are deliberately not a parity guarantee.
    """
    flat = tensor.detach().reshape(-1)
    if not bool(torch.isfinite(flat).all()):
        raise ValueError("weight fingerprint requires finite tensor values")
    total = torch.zeros((), device=flat.device, dtype=torch.float64)
    for start in range(0, flat.numel(), 65536):
        values = flat[start : start + 65536].to(torch.float64)
        indices = torch.arange(start, start + values.numel(), device=flat.device, dtype=torch.int64)
        weights = ((indices * 17 + 13) % 251).to(torch.float64) / 125 - 1
        total += (values * weights).sum()
    return total.item() / math.sqrt(max(1, flat.numel()))


def fingerprints_match(sent: float, received: float) -> bool:
    """Allow 0.5% relative / 1e-4 absolute fp16/bf16 rounding; reject NaN/Inf."""
    return math.isfinite(sent) and math.isfinite(received) and math.isclose(sent, received, rel_tol=0.005, abs_tol=1e-4)
