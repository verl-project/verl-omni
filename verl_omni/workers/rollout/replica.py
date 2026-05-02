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
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict
from verl.workers.rollout.replica import RolloutReplicaRegistry


class DiffusionOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    diffusion_output: Any
    """generated image tensor (CHW format) / video tensor (TCHW format)"""
    log_probs: Optional[Any] = None
    """logprobs of generated image/video"""
    stop_reason: Optional[str] = None
    """stop reason: 'completed', 'aborted', or None for unknown"""
    num_preempted: Optional[int] = None
    """number of preempted times for metric calculation"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


def _load_vllm_omni():
    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniReplica

    return vLLMOmniReplica


def _load_vllm_omni_thinker():
    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniThinkerReplica

    return vLLMOmniThinkerReplica


RolloutReplicaRegistry.register("vllm_omni", _load_vllm_omni)
RolloutReplicaRegistry.register("vllm_omni_thinker", _load_vllm_omni_thinker)
