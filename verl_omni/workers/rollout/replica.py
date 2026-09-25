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
import asyncio
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.tracking import RLInsightLogger
from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.rollout.replica import RolloutReplicaRegistry
from verl.workers.rollout.utils import update_prometheus_config

from verl_omni.workers.config import DiffusionRolloutConfig
from verl_omni.workers.rollout.base import get_rollout_sequence_parallel_size, get_rollout_world_size


class DiffusionOutput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    diffusion_output: Any
    """Generated uint8 pixel tensor (CHW/TCHW) in [0, 255], or floating-point latents."""
    log_probs: Optional[Any] = None
    """logprobs of generated image/video"""
    stop_reason: Optional[str] = None
    """stop reason: 'completed', 'aborted', or None for unknown"""
    num_preempted: Optional[int] = None
    """number of preempted times for metric calculation"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class DiffusionLLMServerManager(LLMServerManager):
    """Include diffusion SP ranks without changing verl's rollout lifecycle."""

    async def _initialize_llm_servers(self, start_rank: int | None = None):
        if get_rollout_sequence_parallel_size(self.rollout_config) == 1:
            return await super()._initialize_llm_servers(start_rank=start_rank)

        # The pinned manager has no replica-size hook; only its SP allocation is overridden.
        config = omega_conf_to_dataclass(self.rollout_config, dataclass_type=DiffusionRolloutConfig)
        start_rank = self.start_rank if start_rank is None else start_rank
        replica_size = get_rollout_world_size(config)
        world_size = self.worker_group.world_size if self.worker_group else config.n_gpus_per_node * config.nnodes
        if world_size < 1 or world_size % replica_size:
            raise ValueError(f"GPU pool size {world_size} must be divisible by replica size {replica_size}")
        if config.disable_log_stats and (config.prometheus.enable or RLInsightLogger.enabled()):
            raise ValueError("Metrics monitoring requires disable_log_stats=False, but it is currently True.")

        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=start_rank + rank,
                config=config,
                model_config=self.model_config,
                gpus_per_node=config.n_gpus_per_node,
            )
            for rank in range(world_size // replica_size)
        ]
        if self.worker_group:
            await asyncio.gather(*[replica.init_hybrid(self.worker_group) for replica in self.rollout_replicas])
        else:
            await asyncio.gather(*[replica.init_standalone() for replica in self.rollout_replicas])

        self.server_handles = [replica._server_handle for replica in self.rollout_replicas]
        self.server_addresses = [replica._server_address for replica in self.rollout_replicas]
        print(f"LLMServerManager: {self.server_addresses}")
        if not config.disable_log_stats:
            if config.prometheus.enable:
                update_prometheus_config(config.prometheus, self.server_addresses, config.name)
            if RLInsightLogger.enabled():
                RLInsightLogger.register_rollout_metrics(
                    self.server_addresses,
                    config.name,
                    labels=[{"replica": replica.replica_rank} for replica in self.rollout_replicas],
                )


def _load_vllm_omni():
    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniReplica

    return vLLMOmniReplica


RolloutReplicaRegistry.register("vllm_omni", _load_vllm_omni)
