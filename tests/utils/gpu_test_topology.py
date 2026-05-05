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

import torch


def resolve_requested_num_gpus(default_num_gpus: int = 4) -> int:
    """Resolve GPU count from NUM_GPUS env var, clamped by visible CUDA devices."""
    n = int(os.getenv("NUM_GPUS", str(default_num_gpus)))
    available = torch.cuda.device_count()
    if available <= 0:
        return 0
    return max(1, min(n, available))


def resolve_diffusion_agent_loop_gpu_topology(default_num_gpus: int = 4) -> tuple[int, int, int]:
    """Return (requested_gpus, tensor_parallel_size, attention_heads)."""
    n = resolve_requested_num_gpus(default_num_gpus)
    tp = min(2, n)
    return n, tp, tp


def resolve_reward_loop_gpu_topology(default_num_gpus: int = 4) -> tuple[int, int]:
    """Return (reward_model_gpus, tensor_parallel_size) for reward-model tests.

    TP is capped at 2 for broad compatibility in smoke runs; reward_model_gpus matches TP.
    """
    tp = min(2, resolve_requested_num_gpus(default_num_gpus))
    return tp, tp
