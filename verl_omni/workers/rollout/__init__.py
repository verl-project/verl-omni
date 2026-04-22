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
from verl.workers.rollout.replica import RolloutReplicaRegistry


def _load_vllm_omni():
    try:
        from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniReplica
    except ImportError as err:
        raise ImportError("vllm-omni rollout requires vllm-omni to be installed.") from err

    return vLLMOmniReplica


# TODO (mike): drop this once `verl` drops diffusion related config
# Override the upstream "vllm_omni" registration so that verl-omni's
# DiffusionModelConfigis used.
RolloutReplicaRegistry.register("vllm_omni", _load_vllm_omni)
