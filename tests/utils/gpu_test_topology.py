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
import os

from verl.utils.device import get_device_name, get_torch_device

# GPU types the GPU smoke tests are validated on; on any other device they skip as before.
ENABLED_GPU_DEVICES = ("cuda", "xpu")


def enabled_gpu_count() -> int:
    """Return the number of visible GPUs, or 0 if the current device is not an enabled GPU type."""
    if get_device_name() not in ENABLED_GPU_DEVICES:
        return 0
    return get_torch_device().device_count()


def empty_gpu_cache() -> None:
    """Release cached GPU memory on enabled GPU types; no-op on any other device."""
    if enabled_gpu_count():
        get_torch_device().empty_cache()


def resolve_requested_num_gpus(default_num_gpus: int = 4) -> int:
    """Resolve GPU count from NUM_GPUS env var, clamped by visible GPUs."""
    n = int(os.getenv("NUM_GPUS", str(default_num_gpus)))
    available = enabled_gpu_count()
    if available <= 0:
        return 0
    return max(1, min(n, available))


def resolve_diffusion_agent_loop_gpu_topology(default_num_gpus: int = 4) -> tuple[int, int, int]:
    """Return (requested_gpus, tensor_parallel_size, attention_heads)."""
    n = resolve_requested_num_gpus(default_num_gpus)
    tp = min(2, n)
    attention_heads = tp
    return n, tp, attention_heads


def resolve_reward_loop_gpu_topology(default_num_gpus: int = 4) -> tuple[int, int]:
    """Return (reward_model_gpus, tensor_parallel_size) for reward-model tests.

    TP is capped at 2 for broad compatibility in smoke runs; reward_model_gpus matches TP.
    """
    tp = min(2, resolve_requested_num_gpus(default_num_gpus))
    return tp, tp
