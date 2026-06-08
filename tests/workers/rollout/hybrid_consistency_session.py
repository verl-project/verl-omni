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
"""Hybrid actor+rollout session for LoRA weight-sync consistency tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import ray
import torch
from omegaconf import DictConfig
from verl.checkpoint_engine import CheckpointEngineManager
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.workers.rollout.llm_server import LLMServerManager

import verl_omni.workers.rollout.replica  # noqa: F401 — register vllm_omni replica
from verl_omni.workers.engine_workers import ActorRolloutRefWorker
from verl_omni.workers.rollout.replica import DiffusionOutput

from .train_rollout_consistency_helpers import (
    QwenImageConsistencyProfile,
    build_train_batch_from_rollout,
    compose_trainer_config,
    rollout_sampling_params,
)


@dataclass
class HybridConsistencySession:
    profile: QwenImageConsistencyProfile
    config: DictConfig
    worker_group: RayWorkerGroup
    llm_manager: LLMServerManager
    checkpoint_manager: CheckpointEngineManager

    @classmethod
    async def create(cls, profile: QwenImageConsistencyProfile) -> HybridConsistencySession:
        config = compose_trainer_config(profile)
        worker_group = _create_hybrid_actor_rollout_worker_group(config)
        resource_pool = RayResourcePool(process_on_nodes=[1], max_colocate_count=1)
        llm_manager = await LLMServerManager.create(
            config=config,
            worker_group=worker_group,
            rollout_resource_pool=resource_pool,
        )
        checkpoint_engine_config = omega_conf_to_dataclass(config.actor_rollout_ref.rollout.checkpoint_engine)
        checkpoint_manager = CheckpointEngineManager(
            checkpoint_engine_config,
            worker_group,
            llm_manager.get_replicas(),
        )
        await checkpoint_manager.sleep_replicas()
        return cls(profile, config, worker_group, llm_manager, checkpoint_manager)

    @property
    def server(self):
        return self.llm_manager.server_handles[0]

    async def sync_weights(self, global_steps: int = 0) -> None:
        await self.checkpoint_manager.update_weights(global_steps=global_steps)

    def generate(
        self,
        *,
        prompt_ids: list[int],
        negative_prompt_ids: list[int],
        seed: int = 42,
    ) -> DiffusionOutput:
        output: DiffusionOutput = ray.get(
            self.server.generate.remote(
                prompt_ids=prompt_ids,
                negative_prompt_ids=negative_prompt_ids,
                sampling_params=rollout_sampling_params(profile=self.profile, seed=seed),
                request_id=f"consistency_{uuid4().hex[:8]}",
            ),
            timeout=self.profile.generate_timeout_s,
        )
        return output

    async def recompute_log_probs(self, extra_fields: dict[str, Any]) -> torch.Tensor:
        await self.checkpoint_manager.sleep_replicas()
        train_batch = build_train_batch_from_rollout(extra_fields)
        output = self.worker_group.infer_actor_batch(train_batch)
        if isinstance(output, list):
            output = output[0]
        if hasattr(output, "get"):
            log_probs = output.get("log_probs")
        elif hasattr(output, "batch"):
            log_probs = output.batch["log_probs"]
        else:
            log_probs = output["log_probs"]
        return log_probs.detach().cpu().float()

    async def close(self) -> None:
        for handle in self.llm_manager.server_handles:
            ray.kill(handle)
        del self.worker_group
        torch.cuda.empty_cache()


def _create_hybrid_actor_rollout_worker_group(config: DictConfig) -> RayWorkerGroup:
    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(ActorRolloutRefWorker),
        config=config.actor_rollout_ref,
        role="actor_rollout",
    )
    resource_pool = RayResourcePool(process_on_nodes=[1], max_colocate_count=1)
    worker_group = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
    worker_group.init_model()
    return worker_group


def perturb_hybrid_actor_lora(worker_group: RayWorkerGroup, *, scale: float = 1e-3, seed: int = 12345) -> None:
    """Apply a small deterministic perturbation to LoRA weights on the hybrid actor."""

    def _perturb(worker: ActorRolloutRefWorker) -> None:
        engine = worker.actor.engine
        was_offload = engine._is_offload_param
        load_fsdp_model_to_gpu(engine.module)
        root = getattr(engine.module, "_fsdp_wrapped_module", engine.module)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for name, param in root.named_parameters():
                if "lora" in name.lower() and param.requires_grad:
                    noise = torch.randn(param.shape, generator=generator, dtype=torch.float32)
                    param.add_(scale * noise.to(device=param.device, dtype=param.dtype))
        if was_offload:
            offload_fsdp_model_to_cpu(engine.module)

    ray.get(worker_group._workers[0].__ray_call__.remote(lambda self: _perturb(self)))


def run_hybrid_session(coro):
    """Run async hybrid session setup/teardown from sync pytest."""
    return asyncio.run(coro)


async def _session_lifecycle(profile: QwenImageConsistencyProfile):
    session = await HybridConsistencySession.create(profile)
    try:
        return session
    except Exception:
        await session.close()
        raise
