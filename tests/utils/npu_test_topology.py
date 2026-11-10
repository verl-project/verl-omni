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
"""Resource selection for TP=2 NPU inference smoke tests."""

import os


def inference_npu_count(available: int) -> int:
    """Use one TP=2 replica unless full-allocation coverage is requested."""
    requested = int(os.environ.get("NUM_NPUS", str(available)))
    if requested < 2 or requested > available or requested % 2:
        raise ValueError(f"NUM_NPUS must be even and between 2 and {available}, got {requested}.")
    full_topology = os.environ.get("NPU_SMOKE_FULL_TOPOLOGY", "0")
    if full_topology not in ("0", "1"):
        raise ValueError("NPU_SMOKE_FULL_TOPOLOGY must be 0 or 1.")
    return requested if full_topology == "1" else 2
