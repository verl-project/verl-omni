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

"""Agent loop manager for composite AR+DiT rollouts returning split batches."""

from __future__ import annotations

import asyncio

import ray
from omegaconf import DictConfig
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, auto_await
from verl.protocol import DataProto
from verl.trainer.ppo.utils import SkipManager
from verl.workers.rollout.llm_server import LLMServerClient

from verl_omni.agent_loop.composite_agent_loop import CompositeAgentLoopWorker


class CompositeAgentLoopManager(AgentLoopManager):
    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] | None = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] | None = None,
    ):
        self.agent_loop_workers_class = ray.remote(CompositeAgentLoopWorker)
        super().__init__(config, llm_client, teacher_client, reward_loop_worker_handles)

    @auto_await
    @SkipManager.annotate(role="rollout")
    async def generate_sequences(self, prompts: DataProto) -> tuple[DataProto, DataProto]:
        """Gather ``(ar_batch, diffusion_batch)`` tuples from composite workers."""

        chunks = prompts.chunk(len(self.agent_loop_workers))
        outputs = await asyncio.gather(
            *[
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunks, strict=True)
            ]
        )

        # concatenate outputs
        ar_outputs = [output[0] for output in outputs]
        diffusion_outputs = [output[1] for output in outputs]

        ar_batch = DataProto.concat(ar_outputs)
        diffusion_batch = DataProto.concat(diffusion_outputs)

        # calculate performance metrics
        metrics = [output.meta_info.pop("metrics") for output in ar_outputs]
        ar_timing = self._performance_metrics(metrics, ar_batch)
        for key in ar_timing.keys():
            new_key = key.replace("agent_loop/", "agent_loop/ar/")
            ar_timing[new_key] = ar_timing.pop(key)
        ar_batch.meta_info = {"timing": ar_timing, **ar_outputs[0].meta_info}

        metrics = [output.meta_info.pop("metrics") for output in diffusion_outputs]
        diffusion_timing = self._performance_metrics(metrics, ar_batch)
        diffusion_batch.meta_info = {"timing": diffusion_timing, **diffusion_outputs[0].meta_info}

        return ar_batch, diffusion_batch
