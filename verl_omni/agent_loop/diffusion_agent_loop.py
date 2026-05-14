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
import random
from typing import Any, Optional
from uuid import uuid4

import hydra
import numpy as np
import ray
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from verl.base_config import BaseConfig
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    DictConfigWrap,
    _agent_loop_registry,
)
from verl.experimental.agent_loop.utils import resolve_config_path
from verl.protocol import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import get_dataset_class
from verl.utils.profiler import simple_timer
from verl.workers.rollout.llm_server import LLMServerClient

from verl_omni.workers.config import DiffusionModelConfig, DiffusionRolloutConfig
from verl_omni.workers.rollout.replica import DiffusionOutput


def _config_to_sampling_dict(config: Optional[BaseConfig]) -> dict:
    if config is None:
        return {}
    return {k: v for k, v in config.items() if not k.startswith("_")}


async def _server_generate_batched(
    server_manager: LLMServerClient,
    *,
    request_id: str,
    prompt_ids: list[int],
    sampling_params: dict[str, Any],
    num_outputs_per_prompt: int,
    image_data: Optional[list[Any]] = None,
    video_data: Optional[list[Any]] = None,
    negative_prompt_ids: Optional[list[int]] = None,
) -> list[DiffusionOutput]:
    """Invoke ``vLLMOmniHttpServer.generate_batched`` via the LLM server load balancer.

    ``LLMServerClient`` in upstream ``verl`` only exposes ``generate``, so this
    helper acquires a server handle via the existing load-balancer pair and
    calls the new ``generate_batched`` method directly on the Ray actor. The
    sticky-session behavior of ``LLMServerClient.generate`` is intentionally
    bypassed here: each diffusion request is independent and benefits more
    from least-loaded routing than prefix-cache stickiness.
    """
    server_id, server = await server_manager._acquire_server(request_id)
    try:
        return await server.generate_batched.remote(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            num_outputs_per_prompt=num_outputs_per_prompt,
            image_data=image_data,
            video_data=video_data,
            negative_prompt_ids=negative_prompt_ids,
        )
    finally:
        server_manager._release_server(server_id)


class DiffusionAgentLoopOutput(BaseModel):
    """Agent loop output."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: list[int]
    """Prompt token ids."""
    response_diffusion_output: Any
    """Response diffusion output (torch.Tensor): image tensor (CHW) / video tensor (TCHW)."""
    response_logprobs: Optional[Any] = None
    """Log probabilities for the response tokens. (torch.Tensor)"""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class _InternalDiffusionAgentLoopOutput(DiffusionAgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_diffusion_output: torch.Tensor
    """Response diffusion output: image (NCHW format) / video (NTCHW format)."""
    response_logprobs: Optional[torch.Tensor] = None
    """Log probabilities over denoising timesteps."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class DiffusionAgentLoopWorker:
    """Diffusion Agent loop worker takes a batch of messages and run each message in an agent loop.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server replicas, produced by
            ``LLMServerManager.get_client()`` in the trainer.
        teacher_client (dict[str, LLMServerClient]): Not used by diffusion training; accepted to
            keep the constructor signature compatible with verl's ``AgentLoopManager.create()``,
            which positionally forwards a teacher client argument to each worker.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming
            reward computation.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] | None = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        self.config = config
        rollout_config = config.actor_rollout_ref.rollout
        model_config = config.actor_rollout_ref.model
        self.rollout_config: DiffusionRolloutConfig = omega_conf_to_dataclass(rollout_config)
        self.model_config: DiffusionModelConfig = omega_conf_to_dataclass(model_config)

        if not hasattr(self, "server_manager"):
            self.server_manager = llm_client

        self.dataset_cls = get_dataset_class(config.data)
        self.reward_loop_worker_handles = reward_loop_worker_handles

        self.tokenizer = self.model_config.tokenizer
        self.processor = self.model_config.processor

        self.max_prompt_embed_length = self.rollout_config.pipeline.max_sequence_length

        agent_loop_config_path = self.rollout_config.agent.agent_loop_config_path
        if agent_loop_config_path:
            resolved_path = resolve_config_path(agent_loop_config_path)
            agent_loop_configs = OmegaConf.load(resolved_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.model_config.get("custom_chat_template", None) is not None:
            if self.model_config.processor is not None:
                self.model_config.processor.chat_template = self.model_config.custom_chat_template
            self.model_config.tokenizer.chat_template = self.model_config.custom_chat_template

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch with the following fields.

            - ``prompts``: ``[bsz, prompt_length]`` prompt token ids from dataset.
            - ``responses``: diffusion output, typically ``[bsz, C, H, W]`` (image)
              or ``[bsz, T, C, H, W]`` (video).
            - ``rm_scores`` (optional): ``[bsz, 1]`` reward model scores.
            - ``meta_info``:

              - ``metrics``: ``List[dict]``, per-sample agent loop metrics.
              - ``reward_extra_keys`` (optional): ``List[str]``, keys for reward
                extra info for logging/validation.
        """
        config = self.rollout_config

        sampling_params = {
            **_config_to_sampling_dict(config.pipeline),
            **_config_to_sampling_dict(config.algo),
            "logprobs": config.calculate_log_probs,
        }

        is_validate = batch.meta_info.get("validate", False)

        if is_validate:
            sampling_params.update(_config_to_sampling_dict(config.val_kwargs.pipeline))
            sampling_params.update(_config_to_sampling_dict(config.val_kwargs.algo))
            sampling_params["seed"] = config.val_kwargs.seed
            sampling_params["logprobs"] = False
        else:
            sampling_params["global_steps"] = batch.meta_info["global_steps"]

        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        rollout_n = self._effective_rollout_n(is_validate)
        if self._can_use_batched_path(batch, rollout_n):
            outputs = await self._run_agent_loops_batched(batch, sampling_params, rollout_n)
        else:
            tasks = []
            for i in range(len(batch)):
                kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
                tasks.append(asyncio.create_task(self._run_agent_loop(sampling_params, **kwargs)))
            outputs = await asyncio.gather(*tasks)

        output = self._postprocess(outputs, input_non_tensor_batch=batch.non_tensor_batch)

        return output

    def _effective_rollout_n(self, is_validate: bool) -> int:
        """Number of samples generated per prompt for the current call."""
        if is_validate:
            return int(getattr(self.rollout_config.val_kwargs, "n", 1) or 1)
        return int(getattr(self.rollout_config, "n", 1) or 1)

    def _can_use_batched_path(self, batch: DataProto, rollout_n: int) -> bool:
        """Decide whether ``generate_sequences`` can dispatch via ``generate_batched``.

        The trainer expands each prompt with ``DataProto.repeat(repeat_times=n,
        interleave=True)`` before calling us, so consecutive groups of
        ``rollout_n`` rows share the same prompt (and same ``agent_name``).
        We only collapse a group when (a) the rollout server is ``vllm_omni``,
        (b) the flag is enabled, (c) ``rollout_n > 1`` and divides the batch
        cleanly, and (d) every row in the group uses the same ``agent_name``.
        """
        if not bool(getattr(self.rollout_config, "enable_batched_diffusion", False)):
            return False
        if rollout_n <= 1:
            return False
        if getattr(self.rollout_config, "name", None) != "vllm_omni":
            return False
        if len(batch) == 0 or len(batch) % rollout_n != 0:
            return False
        agent_names = batch.non_tensor_batch.get("agent_name")
        if agent_names is None:
            return False
        for start in range(0, len(batch), rollout_n):
            group = agent_names[start : start + rollout_n]
            if not all(name == group[0] for name in group):
                return False
        return True

    async def _run_agent_loops_batched(
        self,
        batch: DataProto,
        sampling_params: dict[str, Any],
        rollout_n: int,
    ) -> list["_InternalDiffusionAgentLoopOutput"]:
        """Dispatch one ``generate_batched`` call per prompt group and flatten the results.

        Each group of ``rollout_n`` adjacent rows is collapsed into a single
        engine request with ``num_outputs_per_prompt = rollout_n``. The
        returned ``rollout_n`` per-sample :class:`DiffusionOutput` objects are
        then unpacked into ``rollout_n`` independent agent-loop outputs whose
        per-row ``kwargs`` (raw prompt, reward model spec, ...) are inherited
        from the original batch row at the matching position.
        """
        tasks: list[asyncio.Task] = []
        for group_start in range(0, len(batch), rollout_n):
            row_kwargs_list = [
                {k: v[group_start + offset] for k, v in batch.non_tensor_batch.items()} for offset in range(rollout_n)
            ]
            tasks.append(asyncio.create_task(self._run_agent_loop_batched(sampling_params, rollout_n, row_kwargs_list)))
        groups = await asyncio.gather(*tasks)
        return [out for group in groups for out in group]

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ) -> _InternalDiffusionAgentLoopOutput:
        assert agent_name in _agent_loop_registry, (
            f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
        )

        agent_loop_config = _agent_loop_registry[agent_name]
        agent_loop = hydra.utils.instantiate(
            config=agent_loop_config,
            trainer_config=DictConfigWrap(config=self.config),
            server_manager=self.server_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
            data_config=DictConfigWrap(self.config.data),
        )
        output: DiffusionAgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
        return await self._agent_loop_postprocess(output, **kwargs)

    async def _run_agent_loop_batched(
        self,
        sampling_params: dict[str, Any],
        num_outputs_per_prompt: int,
        row_kwargs_list: list[dict[str, Any]],
    ) -> list[_InternalDiffusionAgentLoopOutput]:
        """Run a single ``generate_batched`` engine call for one prompt group.

        The first row in ``row_kwargs_list`` is used to tokenize the prompt
        (all rows in the group are duplicates of the same prompt). The
        resulting :class:`list[DiffusionOutput]` of length ``num_outputs_per_prompt``
        is then post-processed per-row so reward / extra-field handling
        matches the per-row code path.
        """
        assert len(row_kwargs_list) == num_outputs_per_prompt, (
            f"row_kwargs_list size {len(row_kwargs_list)} != num_outputs_per_prompt {num_outputs_per_prompt}"
        )
        head_kwargs = row_kwargs_list[0]
        agent_name = head_kwargs["agent_name"]
        assert agent_name in _agent_loop_registry, (
            f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
        )

        # Reuse the registered agent loop just for tokenization / vision-info
        # extraction so the prompt-prep path stays identical to the per-row path.
        agent_loop_config = _agent_loop_registry[agent_name]
        agent_loop = hydra.utils.instantiate(
            config=agent_loop_config,
            trainer_config=DictConfigWrap(config=self.config),
            server_manager=self.server_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
            data_config=DictConfigWrap(self.config.data),
        )

        raw_prompt = head_kwargs["raw_prompt"]
        raw_negative_prompt = head_kwargs.get("raw_negative_prompt")
        multi_modal_data = await agent_loop.process_vision_info(raw_prompt)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        prompt_ids = await agent_loop.apply_chat_template(raw_prompt, images=images, videos=videos)
        if raw_negative_prompt is not None:
            negative_prompt_ids = await agent_loop.apply_chat_template(
                raw_negative_prompt, images=images, videos=videos
            )
        else:
            negative_prompt_ids = None

        metrics: dict[str, Any] = {}
        with simple_timer("generate_sequences", metrics):
            diffusion_outputs = await _server_generate_batched(
                self.server_manager,
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                num_outputs_per_prompt=num_outputs_per_prompt,
                image_data=images,
                video_data=videos,
                negative_prompt_ids=negative_prompt_ids,
            )
        if len(diffusion_outputs) != num_outputs_per_prompt:
            raise RuntimeError(
                f"generate_batched returned {len(diffusion_outputs)} samples, expected {num_outputs_per_prompt}"
            )

        results: list[_InternalDiffusionAgentLoopOutput] = []
        for diffusion_output, row_kwargs in zip(diffusion_outputs, row_kwargs_list, strict=True):
            # Each sample gets its own metrics dict so per-row timings don't alias
            # the same underlying dict and so num_preempted is per-sample.
            sample_metrics = dict(metrics)
            if sample_metrics.get("num_preempted") is None:
                sample_metrics["num_preempted"] = (
                    diffusion_output.num_preempted if diffusion_output.num_preempted is not None else -1
                )
            agent_output = DiffusionAgentLoopOutput(
                prompt_ids=prompt_ids,
                response_diffusion_output=diffusion_output.diffusion_output,
                response_logprobs=diffusion_output.log_probs,
                num_turns=2,
                metrics=sample_metrics,
                extra_fields=diffusion_output.extra_fields,
            )
            results.append(await self._agent_loop_postprocess(agent_output, **row_kwargs))
        return results

    async def _agent_loop_postprocess(self, output, **kwargs) -> _InternalDiffusionAgentLoopOutput:
        """Perform post-processing operations on the output of each individual agent loop."""
        # Pad extra tensor outputs from vllm-omni (e.g. prompt embeddings).
        extra_fields = {}
        for k, v in output.extra_fields.items():
            if isinstance(v, torch.Tensor):
                if k in ["prompt_embeds", "negative_prompt_embeds"]:
                    pad_tuple = (0, 0, 0, self.max_prompt_embed_length - v.shape[0])
                    v = F.pad(v, pad_tuple, value=0)
                elif k in ["prompt_embeds_mask", "negative_prompt_embeds_mask"]:
                    pad_tuple = (0, self.max_prompt_embed_length - v.shape[0])
                    v = F.pad(v, pad_tuple, value=0)
                extra_fields[k] = v.unsqueeze(0)
            else:
                extra_fields[k] = v

        extra_fields["raw_prompt"] = kwargs["raw_prompt"]

        prompt_output = self.tokenizer.pad(
            {"input_ids": output.prompt_ids},
            padding="max_length",
            max_length=self.rollout_config.prompt_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)

        response_diffusion_output = output.response_diffusion_output.unsqueeze(0)

        response_logprobs = None
        if output.response_logprobs is not None:
            response_logprobs = output.response_logprobs.unsqueeze(0)

        prompt_ids = prompt_output["input_ids"]

        await self._compute_score(
            output,
            prompts=prompt_ids,
            responses=response_diffusion_output,
            kwargs=kwargs,
        )

        if "reward_extra_info" in output.extra_fields:
            extra_fields["reward_extra_info"] = output.extra_fields["reward_extra_info"]

        return _InternalDiffusionAgentLoopOutput(
            prompt_ids=prompt_ids,
            response_diffusion_output=response_diffusion_output,
            response_logprobs=response_logprobs,
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=extra_fields,
        )

    async def _compute_score(self, output, prompts, responses, kwargs):
        """Compute reward score for single sample."""
        enable_async_reward = self.reward_loop_worker_handles is not None

        if output.reward_score is None and enable_async_reward:
            timing = {}
            with simple_timer("compute_score", timing):
                batch = TensorDict(
                    {
                        "prompts": prompts,  # [1, prompt_length]
                        "responses": responses,  # [1, C, H, W] or [1, T, C, H, W]
                    },
                    batch_size=1,
                )
                non_tensor_batch = {
                    **{k: np.array([v]) for k, v in kwargs.items()},
                    "__num_turns__": np.array([output.num_turns]),
                    "tool_extra_fields": np.array([output.extra_fields], dtype=object),
                }

                data = DataProto(
                    batch=batch,
                    non_tensor_batch=non_tensor_batch,
                )
                selected_reward_loop_worker_handle = random.choice(self.reward_loop_worker_handles)
                result = await selected_reward_loop_worker_handle.compute_score.remote(data)
                output.reward_score = result["reward_score"]
                output.extra_fields["reward_extra_info"] = result["reward_extra_info"]
            output.metrics.compute_score = timing["compute_score"]

    def _postprocess(
        self,
        inputs: list[_InternalDiffusionAgentLoopOutput],
        input_non_tensor_batch: dict | None = None,
    ) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_diffusion_output = torch.cat([input.response_diffusion_output for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)

        # Handle extra fields that are tensors
        extra_keys = [k for k, v in inputs[0].extra_fields.items() if isinstance(v, torch.Tensor)]
        for key in extra_keys:
            optional_outputs[key] = torch.cat([input.extra_fields[key] for input in inputs], dim=0)
            for input in inputs:
                del input.extra_fields[key]

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_diffusion_output,  # [bsz, C, H, W] or [bsz, T, C, H, W]
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            rm_scores = torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }
        if self.reward_loop_worker_handles is None and input_non_tensor_batch:
            non_tensor_batch.update(input_non_tensor_batch)

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        metrics = [input.metrics.model_dump() for input in inputs]
        # Collect extra fields from all inputs and convert them to np.ndarray
        extra_fields = {}
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields)
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)

        # Only include reward_extra_keys in meta_info if rm_scores is in batch
        # This avoids conflicts when reward_tensor is merged later in ray_trainer.py
        if "rm_scores" in batch.keys():
            meta_info = {"metrics": metrics, "reward_extra_keys": reward_extra_keys}
        else:
            meta_info = {"metrics": metrics}

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )
