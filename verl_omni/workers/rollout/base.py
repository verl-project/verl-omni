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
from verl.workers.rollout.base import _ROLLOUT_REGISTRY

_ROLLOUT_REGISTRY[("vllm_omni", "async")] = (
    "verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server.vLLMOmniServerAdapter"
)


def get_rollout_sequence_parallel_size(config) -> int:
    """Return the SP footprint, defaulting to one for AR rollout configs."""
    size = 1
    for name in ("ulysses_degree", "ring_degree"):
        degree = getattr(config, name, 1)
        if type(degree) is not int or degree < 1:
            raise ValueError(f"{name} must be a positive integer, got {degree!r}.")
        size *= degree
    return size


def get_rollout_world_size(config) -> int:
    """Count allocated rollout ranks; encoder and VAE parallelism reuse them."""
    return (
        config.tensor_model_parallel_size
        * config.data_parallel_size
        * config.pipeline_model_parallel_size
        * get_rollout_sequence_parallel_size(config)
    )
