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
# TODO: move this file to verl_omni.experimental.agent_loop after V1 is stable
"""TransferQueue adapter for DiffusionAgentLoopManager and DiffusionAgentLoopWorker.

This mirrors verl's ``verl.trainer.ppo.v1.agent_loop_tq`` but reuses the existing
diffusion agent loop instantiation and postprocess logic from
``verl_omni.agent_loop.diffusion_agent_loop``. The worker is fire-and-forget:
it dispatches background generation tasks and writes each finished diffusion
trajectory into TransferQueue instead of returning a ``DataProto`` batch.
"""

import asyncio
import logging
import os
from typing import Any

import ray
import torch
import transfer_queue as tq
from tensordict import NonTensorData, NonTensorStack, TensorDict

from verl.experimental.agent_loop import AgentLoopManager, get_trajectory_info
from verl.utils.ray_utils import auto_await
from verl.utils.tensordict_utils import list_of_dict_to_tensordict

from verl_omni.agent_loop.diffusion_agent_loop import (
    DiffusionAgentLoopWorker,
    _InternalDiffusionAgentLoopOutput,
    _config_to_sampling_dict,
)
from verl_omni.agent_loop.utils import _derive_rollout_seed

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@ray.remote
class DiffusionAgentLoopWorkerTQ(DiffusionAgentLoopWorker):
    """TransferQueue-backed diffusion agent loop worker.

    Unlike the blocking ``DiffusionAgentLoopWorker``, this worker accepts a
    ``TensorDict`` batch, spawns background generation tasks per prompt, and
    returns immediately. Each completed diffusion trajectory is written into
    TransferQueue with key ``{uid}_{session_id}_{index}``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tq.init()
        self.background_tasks = set()

    async def generate_sequences(self, batch: TensorDict) -> None:
        """Spawn diffusion agent loops for each prompt in the batch without waiting."""
        validate = batch["validate"] if "validate" in batch else False
        if isinstance(validate, NonTensorData):
            validate = validate.data
        batch.pop("validate", None)

        config = self.rollout_config
        sampling_params = {
            **_config_to_sampling_dict(config.pipeline),
            **_config_to_sampling_dict(config.algo),
            "logprobs": config.calculate_log_probs,
        }

        rollout_base_seed = None
        if validate:
            sampling_params.update(_config_to_sampling_dict(config.val_kwargs.pipeline))
            sampling_params.update(_config_to_sampling_dict(config.val_kwargs.algo))
            sampling_params["seed"] = config.val_kwargs.seed
            sampling_params["logprobs"] = False
        else:
            global_steps = batch["global_steps"]
            if isinstance(global_steps, NonTensorData):
                global_steps = global_steps.data
            sampling_params["global_steps"] = global_steps
            rollout_seed_meta = batch.get("rollout_seed")
            if rollout_seed_meta is not None:
                rollout_base_seed = int(rollout_seed_meta.data if isinstance(rollout_seed_meta, NonTensorData) else rollout_seed_meta)

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch:
            default_agent_loop = config.agent.default_agent_loop
            batch["agent_name"] = NonTensorData(default_agent_loop)

        index = batch["index"] if "index" in batch else list(range(len(batch)))
        trajectory_info = await get_trajectory_info(
            sampling_params.get("global_steps", -1), index, bool(validate)
        )

        for i in range(len(batch)):
            prompt = self._extract_prompt(batch, i)
            task = asyncio.create_task(
                self._run_prompt(
                    prompt,
                    sampling_params,
                    trajectory=trajectory_info[i],
                    sample_index=i,
                    rollout_base_seed=rollout_base_seed,
                )
            )
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)

    @staticmethod
    def _extract_prompt(batch: TensorDict, i: int) -> dict:
        """Extract per-sample fields from a TensorDict row."""
        prompt = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                prompt[k] = v[i]
            elif isinstance(v, NonTensorStack):
                prompt[k] = v[i].data
            elif isinstance(v, NonTensorData):
                prompt[k] = v.data
            else:
                logger.exception(f"Unsupported type {type(v)} for key {k}")
        return prompt

    async def _run_prompt(
        self,
        prompt: dict,
        sampling_params: dict,
        trajectory: dict,
        sample_index: int,
        rollout_base_seed: int | None = None,
    ) -> None:
        """Spawn ``rollout.n`` sessions per prompt and write trajectories to TQ."""
        uid = prompt["uid"]
        partition_id = "val" if trajectory["validate"] else "train"
        await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "running"})
        try:
            config = self.rollout_config
            n = config.val_kwargs.n if trajectory["validate"] else config.n
            tasks = []
            for session_id in range(n):
                run_sampling_params = dict(sampling_params)
                if rollout_base_seed is not None and not trajectory["validate"]:
                    run_sampling_params["seed"] = _derive_rollout_seed(rollout_base_seed, sample_index * n + session_id)
                task = asyncio.create_task(
                    self._run_agent_loop(
                        run_sampling_params,
                        session_id=session_id,
                        trajectory=trajectory,
                        **prompt,
                    )
                )
                tasks.append(task)
            await asyncio.gather(*tasks)
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "finished"})
        except Exception as e:
            logger.exception(f"Error in _run_prompt for uid={uid}: {e}")
            await tq.async_kv_put(key=uid, partition_id=partition_id, tag={"status": "failure"})

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        *,
        agent_name: str,
        session_id: int = 0,
        trajectory: dict | None = None,
        **kwargs,
    ) -> None:
        """Run one diffusion agent loop session and write its output to TransferQueue."""
        internal: _InternalDiffusionAgentLoopOutput = await super()._run_agent_loop(
            sampling_params, agent_name=agent_name, **kwargs
        )
        uid = kwargs["uid"]
        non_conflicting_kwargs = {
            k: v for k, v in kwargs.items() if k not in {"uid", "global_steps"}
        }
        await self._write_trajectory_to_tq(
            internal,
            uid=uid,
            session_id=session_id,
            trajectory=trajectory,
            validate=trajectory["validate"] if trajectory else False,
            global_steps=sampling_params.get("global_steps"),
            **non_conflicting_kwargs,
        )

    async def _write_trajectory_to_tq(
        self,
        internal: _InternalDiffusionAgentLoopOutput,
        *,
        uid: str,
        session_id: int,
        trajectory: dict | None,
        validate: bool,
        global_steps: int | None = None,
        **kwargs,
    ) -> None:
        """Convert a padded diffusion agent loop output into a TransferQueue row."""
        # Diffusion single-turn agent loops produce one output per session.
        index = 0
        key = f"{uid}_{session_id}_{index}"
        partition_id = "val" if validate else "train"

        field: dict[str, Any] = {
            "prompts": internal.prompt_ids.squeeze(0),
            "responses": internal.response_diffusion_output.squeeze(0),
            "__num_turns__": internal.num_turns,
        }
        if internal.response_logprobs is not None:
            field["rollout_log_probs"] = internal.response_logprobs.squeeze(0)
        if internal.reward_score is not None:
            field["rm_scores"] = torch.tensor([internal.reward_score], dtype=torch.float32)

        extra = internal.extra_fields
        for tensor_key in [
            "attention_mask",
            "prompt_embeds",
            "prompt_embeds_mask",
            "negative_prompt_embeds",
            "negative_prompt_embeds_mask",
            "all_latents",
            "all_timesteps",
        ]:
            value = extra.get(tensor_key)
            if isinstance(value, torch.Tensor):
                field[tensor_key] = value.squeeze(0) if value.dim() >= 1 and value.shape[0] == 1 else value

        # Non-tensor dataset fields forwarded as-is.
        for non_tensor_key in ["reward_model", "data_source", "extra_info", "raw_prompt"]:
            if non_tensor_key in kwargs:
                field[non_tensor_key] = kwargs[non_tensor_key]

        reward_extra_info = extra.get("reward_extra_info")
        extra_fields_out: dict[str, Any] = {}
        if reward_extra_info is not None:
            extra_fields_out["reward_extra_info"] = reward_extra_info
        # Track the rollout model version this trajectory was generated against.
        step = trajectory["step"] if trajectory else global_steps
        extra_fields_out["min_global_steps"] = step
        extra_fields_out["max_global_steps"] = step
        field["extra_fields"] = extra_fields_out

        fields_td = list_of_dict_to_tensordict([field])

        prompt_len = int(internal.prompt_ids.shape[-1])
        tags = [
            {
                "status": "success",
                "prompt_len": prompt_len,
                "response_len": 1,
                "seq_len": prompt_len + 1,
                "global_steps": step,
                "min_global_steps": step,
                "max_global_steps": step,
            }
        ]
        await tq.async_kv_batch_put(
            keys=[key],
            fields=fields_td,
            tags=tags,
            partition_id=partition_id,
        )


class DiffusionAgentLoopManagerTQ(AgentLoopManager):
    """TransferQueue-backed diffusion agent loop manager.

    Dispatches prompt batches to ``DiffusionAgentLoopWorkerTQ`` actors without
    waiting for generated trajectories. Trajectories are consumed through
    TransferQueue by the trainer's replay buffer.
    """

    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = DiffusionAgentLoopWorkerTQ
        super().__init__(*args, **kwargs)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        """Create the diffusion agent loop manager."""
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    def generate_sequences(self, prompts: TensorDict) -> None:
        """Dispatch input batch to diffusion agent loop workers without blocking."""
        chunks = prompts.chunk(len(self.agent_loop_workers))
        ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunks, strict=False)
            ]
        )
