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
import time
from typing import Any, Optional

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
from verl.workers.rollout.llm_server import LLMServerClient

from verl_omni.agent_loop.utils import maybe_per_rollout_seeds
from verl_omni.workers.config import DiffusionModelConfig, DiffusionRolloutConfig

_PENDING_REWARD_KEY = "_pending_reward_awaitable"
_PENDING_REWARD_START_KEY = "_pending_reward_start"
# Surfaced on DataProto.non_tensor_batch so the trainer can ray.get scores while
# sleep_replicas / old_log_prob run (reward GPU is a separate pool).
PENDING_REWARD_REF_KEY = "_pending_reward_ref"
PENDING_REWARD_START_KEY = "_pending_reward_start"


def pop_pending_reward_refs(data: DataProto) -> tuple[list[Any], list[Any]] | None:
    """Remove deferred GenRM ObjectRefs from ``data`` (gen output or batch)."""
    refs = data.non_tensor_batch.pop(PENDING_REWARD_REF_KEY, None)
    starts = data.non_tensor_batch.pop(PENDING_REWARD_START_KEY, None)
    if refs is None:
        return None
    refs_list = list(refs)
    starts_list = list(starts) if starts is not None else [None] * len(refs_list)
    if not any(ref is not None for ref in refs_list):
        return None
    return refs_list, starts_list


def fetch_pending_reward_results(refs: list[Any]) -> list[dict[str, Any]]:
    """``ray.get`` in-flight GenRM ObjectRefs, preserving row alignment."""
    results: list[dict[str, Any] | None] = [None] * len(refs)
    pending_idx = [i for i, ref in enumerate(refs) if ref is not None]
    fetched = ray.get([refs[i] for i in pending_idx])
    for i, result in zip(pending_idx, fetched, strict=True):
        results[i] = result
    for i, result in enumerate(results):
        if result is None:
            raise RuntimeError(f"Missing deferred reward result at row {i}")
    return results  # type: ignore[return-value]


def apply_pending_reward_results(
    batch: DataProto,
    results: list[dict[str, Any]],
    starts: list[Any] | None = None,
) -> None:
    """Attach ``rm_scores`` / reward extras from deferred GenRM results."""
    scores = [float(result["reward_score"]) for result in results]
    reward_extra_infos = [result.get("reward_extra_info") or {} for result in results]
    batch.batch["rm_scores"] = torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)

    reward_extra_keys = list(reward_extra_infos[0].keys()) if reward_extra_infos else []
    for key in reward_extra_keys:
        batch.non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos], dtype=object)
    if reward_extra_keys:
        batch.meta_info["reward_extra_keys"] = reward_extra_keys

    metrics = batch.meta_info.get("metrics")
    if starts is not None and metrics is not None and len(metrics) == len(starts):
        now = time.perf_counter()
        for i, start in enumerate(starts):
            if start is not None and isinstance(metrics[i], dict):
                metrics[i]["compute_score"] = now - float(start)


def materialize_pending_rewards(batch: DataProto) -> dict[str, float]:
    """Block on deferred GenRM ObjectRefs and attach ``rm_scores`` in-place."""
    pending = pop_pending_reward_refs(batch)
    if pending is None:
        return {}
    refs, starts = pending
    t0 = time.perf_counter()
    results = fetch_pending_reward_results(refs)
    apply_pending_reward_results(batch, results, starts)
    return {"pending_reward_resolve": time.perf_counter() - t0}


def _config_to_sampling_dict(config: Optional[BaseConfig]) -> dict:
    if config is None:
        return {}
    return {k: v for k, v in config.items() if not k.startswith("_")}


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
        # Cache Hydra-instantiated agent loops so per-sample fan-out does not
        # stagger request admission into vLLM-Omni's request-level batcher.
        self._agent_loop_cache: dict[str, Any] = {}

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

    def _get_or_create_agent_loop(self, agent_name: str):
        agent_loop = self._agent_loop_cache.get(agent_name)
        if agent_loop is not None:
            return agent_loop

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
            extra_tokenizer_map=self.model_config.extra_tokenizer_map,
        )
        self._agent_loop_cache[agent_name] = agent_loop
        return agent_loop

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
        per_rollout_seeds: Optional[list[int]] = None

        if is_validate:
            sampling_params.update(_config_to_sampling_dict(config.val_kwargs.pipeline))
            sampling_params.update(_config_to_sampling_dict(config.val_kwargs.algo))
            sampling_params["seed"] = config.val_kwargs.seed
            sampling_params["logprobs"] = False
        else:
            sampling_params["global_steps"] = batch.meta_info["global_steps"]
            # Prefer trainer-assigned global indices so chunked workers derive the
            # same per-row seed regardless of local batch position / pack order.
            global_indices = batch.non_tensor_batch.get("_rollout_seed_global_idx")
            if global_indices is not None:
                global_indices = np.asarray(global_indices, dtype=np.int64).reshape(-1)
            per_rollout_seeds = maybe_per_rollout_seeds(batch.meta_info, len(batch), global_indices)

        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        tasks = []
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            task_sampling_params = sampling_params.copy()
            if per_rollout_seeds is not None:
                task_sampling_params["seed"] = per_rollout_seeds[i]
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(
                        task_sampling_params,
                        validate=is_validate,
                        resolve_reward=False,
                        **kwargs,
                    )
                )
            )
        outputs = await asyncio.gather(*tasks)

        # Training: keep GenRM ObjectRefs in-flight and let the trainer await them
        # in parallel with sleep_replicas / old_log_prob (separate reward GPUs).
        # Validation / non-Ray fakes: resolve here so callers still see rm_scores.
        defer_reward_to_trainer = (not is_validate) and self._pending_rewards_are_object_refs(outputs)
        if not defer_reward_to_trainer:
            await self._resolve_pending_scores(outputs)
            return self._postprocess(outputs, input_non_tensor_batch=batch.non_tensor_batch)

        pending_refs: list[Any] = []
        pending_starts: list[Any] = []
        for internal in outputs:
            pending_refs.append(internal.extra_fields.pop(_PENDING_REWARD_KEY, None))
            pending_starts.append(internal.extra_fields.pop(_PENDING_REWARD_START_KEY, None))

        output = self._postprocess(outputs, input_non_tensor_batch=batch.non_tensor_batch)
        output.non_tensor_batch[PENDING_REWARD_REF_KEY] = np.array(pending_refs, dtype=object)
        output.non_tensor_batch[PENDING_REWARD_START_KEY] = np.array(pending_starts, dtype=object)
        return output

    @staticmethod
    def _pending_rewards_are_object_refs(outputs: list[_InternalDiffusionAgentLoopOutput]) -> bool:
        """True when every in-flight reward handle is a Ray ObjectRef (trainer-deferrable)."""
        saw_pending = False
        for internal in outputs:
            pending = internal.extra_fields.get(_PENDING_REWARD_KEY)
            if pending is None:
                continue
            saw_pending = True
            if not isinstance(pending, ray.ObjectRef):
                return False
        return saw_pending

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        *,
        agent_name: str,
        validate: bool = False,
        resolve_reward: bool = True,
        **kwargs,
    ) -> _InternalDiffusionAgentLoopOutput:
        agent_loop = self._get_or_create_agent_loop(agent_name)
        output: DiffusionAgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
        return await self._agent_loop_postprocess(
            output, validate=validate, resolve_reward=resolve_reward, **kwargs
        )

    async def _agent_loop_postprocess(
        self, output, validate: bool = False, resolve_reward: bool = True, **kwargs
    ) -> _InternalDiffusionAgentLoopOutput:
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
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        response_diffusion_output = output.response_diffusion_output.unsqueeze(0)

        response_logprobs = None
        if output.response_logprobs is not None:
            response_logprobs = output.response_logprobs.unsqueeze(0)

        prompt_ids = prompt_output["input_ids"]
        extra_fields["attention_mask"] = prompt_output["attention_mask"]

        await self._compute_score(
            output,
            prompts=prompt_ids,
            responses=response_diffusion_output,
            kwargs=kwargs,
            validate=validate,
        )
        if resolve_reward:
            await self._resolve_pending_score(output)

        if "reward_extra_info" in output.extra_fields:
            extra_fields["reward_extra_info"] = output.extra_fields["reward_extra_info"]
        # Carry unresolved reward futures through to batch resolve.
        if _PENDING_REWARD_KEY in output.extra_fields:
            extra_fields[_PENDING_REWARD_KEY] = output.extra_fields[_PENDING_REWARD_KEY]
            extra_fields[_PENDING_REWARD_START_KEY] = output.extra_fields[_PENDING_REWARD_START_KEY]

        return _InternalDiffusionAgentLoopOutput(
            prompt_ids=prompt_ids,
            response_diffusion_output=response_diffusion_output,
            response_logprobs=response_logprobs,
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=extra_fields,
        )

    async def _compute_score(self, output, prompts, responses, kwargs, validate: bool = False):
        """Launch reward scoring for a single sample without awaiting completion.

        Streaming launch preserves overlap of early GenRM work with later denoising.
        Callers must ``_resolve_pending_score`` / ``_resolve_pending_scores`` before
        reading ``reward_score``.
        """
        enable_async_reward = self.reward_loop_worker_handles is not None

        if output.reward_score is None and enable_async_reward:
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
                meta_info={"validate": validate},
            )
            selected_reward_loop_worker_handle = random.choice(self.reward_loop_worker_handles)
            pending = selected_reward_loop_worker_handle.compute_score.remote(data)
            if asyncio.iscoroutine(pending):
                pending = asyncio.create_task(pending)
            output.extra_fields[_PENDING_REWARD_KEY] = pending
            output.extra_fields[_PENDING_REWARD_START_KEY] = time.perf_counter()

    async def _resolve_pending_score(self, output) -> None:
        """Await one sample's in-flight reward RPC and populate score fields."""
        pending = output.extra_fields.pop(_PENDING_REWARD_KEY, None)
        start = output.extra_fields.pop(_PENDING_REWARD_START_KEY, None)
        if pending is None:
            return

        result = await pending
        output.reward_score = result["reward_score"]
        output.extra_fields["reward_extra_info"] = result["reward_extra_info"]
        if start is not None:
            output.metrics.compute_score = time.perf_counter() - start

    async def _resolve_pending_scores(self, outputs: list[_InternalDiffusionAgentLoopOutput]) -> None:
        """Resolve all in-flight reward RPCs after denoise gather completes."""

        async def _resolve_one(internal: _InternalDiffusionAgentLoopOutput) -> None:
            pending = internal.extra_fields.pop(_PENDING_REWARD_KEY, None)
            start = internal.extra_fields.pop(_PENDING_REWARD_START_KEY, None)
            if pending is None:
                return
            result = await pending
            internal.reward_score = result["reward_score"]
            internal.extra_fields["reward_extra_info"] = result["reward_extra_info"]
            if start is not None:
                internal.metrics.compute_score = time.perf_counter() - start

        await asyncio.gather(*[_resolve_one(o) for o in outputs])

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
        if input_non_tensor_batch:
            non_tensor_batch.update(input_non_tensor_batch)

        # add reward_extra_info to non_tensor_batch
        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys()) if reward_extra_infos else []
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
