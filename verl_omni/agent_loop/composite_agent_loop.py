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
"""

Composite agent framework for multi-stage visual generation by
AR (LLM/MLLM) + DiT composite architecture, as well as for agentic RL.

- CompositeAgentLoopWorker extends DiffusionAgentLoopWorker with:
  - a reward handle for AR part
  - extra returns from AR generation, e.g., reward score and token-level log-probs.

"""

import asyncio
import random
from typing import Any, Optional

import numpy as np
import ray
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pydantic import ConfigDict
from tensordict import TensorDict
from verl.base_config import BaseConfig
from verl.protocol import DataProto
from verl.utils.profiler import simple_timer
from verl.workers.rollout.llm_server import LLMServerClient

from verl_omni.agent_loop.diffusion_agent_loop import DiffusionAgentLoopOutput, DiffusionAgentLoopWorker
from verl_omni.agent_loop.utils import maybe_per_rollout_seeds


def _config_to_sampling_dict(config: Optional[BaseConfig]) -> dict:
    if config is None:
        return {}
    return {k: v for k, v in config.items() if not k.startswith("_")}


class CompositeAgentLoopOutput(DiffusionAgentLoopOutput):
    """Agent loop output. Supplement additional fields for AR part."""

    llm_response_logprobs: Optional[Any] = None
    """Log probabilities for the response tokens."""
    llm_reward_score: Optional[float] = None
    """Reward score for the semantic reward."""


class _InternalCompositeAgentLoopOutput(CompositeAgentLoopOutput):
    """Internal agent loop output with padded sequences.
    Supplement additional fields for AR part.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_diffusion_output: torch.Tensor
    """Response diffusion output: image (NCHW format) / video (NTCHW format)."""
    llm_response_logprobs: Optional[torch.Tensor] = None
    """Log probabilities for the response tokens."""
    response_logprobs: Optional[torch.Tensor] = None
    """Log probabilities over denoising timesteps."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class CompositeAgentLoopWorker(DiffusionAgentLoopWorker):
    """Composite Agent loop worker takes a batch of messages and run each message in an agent loop.
    Two-stage rollout: LLM CoT+rewrite → prompt embeds → DiT sampling.

    Args:
        config (DictConfig): whole config for main entrypoint.
        llm_client (LLMServerClient): Client for the LLM server replicas, produced by
            ``LLMServerManager.get_client()`` in the trainer.
        teacher_client (dict[str, LLMServerClient]): Not used by diffusion training; accepted to
            keep the constructor signature compatible with verl's ``AgentLoopManager.create()``,
            which positionally forwards a teacher client argument to each worker.
        reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming
            reward computation.
        ar_reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming
            reward computation. Optional if reward_loop_worker_handles provides reward computation for both LLM and DiT.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client: dict[str, LLMServerClient] | None = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        assert reward_loop_worker_handles is None or len(reward_loop_worker_handles) == 2
        if reward_loop_worker_handles is None:
            self.ar_reward_loop_worker_handles = None
        else:
            self.ar_reward_loop_worker_handles = reward_loop_worker_handles[1:]  # second for llm
            reward_loop_worker_handles = reward_loop_worker_handles[:1]  # first for dit

        super().__init__(config, llm_client, teacher_client, reward_loop_worker_handles)

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch with the following fields.

            - ``prompts``: ``[bsz, prompt_length]`` original prompt token ids from dataset.
            - ``responses``: diffusion output, typically ``[bsz, C, H, W]`` (image)
              or ``[bsz, T, C, H, W]`` (video).
            - ``rm_scores`` (optional): ``[bsz, 2]`` reward model scores for ar and dit part.
            - ``text_encoder_responses``: ``List[str]``, refined prompts with CoT.
            - ``meta_info``:
              - ``metrics``: ``List[dict]``, per-sample agent loop metrics.
              - ``reward_extra_keys`` (optional): ``List[str]``, keys for reward
                extra info for logging/validation.
        """
        config = self.rollout_config

        # composite sampling configs for LLM and DiT
        sampling_params = {
            **_config_to_sampling_dict(config.pipeline),
            **_config_to_sampling_dict(config.algo),
            "logprobs": config.calculate_log_probs,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "repetition_penalty": config.repetition_penalty,
            "llm_logprobs": config.llm_calculate_log_probs,
            "max_new_tokens": config.max_new_tokens,
        }

        is_validate = batch.meta_info.get("validate", False)
        per_rollout_seeds: Optional[list[int]] = None

        if is_validate:
            sampling_params.update(_config_to_sampling_dict(config.val_kwargs.pipeline))
            sampling_params.update(_config_to_sampling_dict(config.val_kwargs.algo))
            sampling_params["seed"] = config.val_kwargs.seed
            sampling_params["logprobs"] = False

            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature
            sampling_params["llm_logprobs"] = False
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
            tasks.append(asyncio.create_task(self._run_agent_loop(task_sampling_params, **kwargs)))
        outputs = await asyncio.gather(*tasks)

        output = self._postprocess(outputs, input_non_tensor_batch=batch.non_tensor_batch)

        return output

    async def _agent_loop_postprocess(
        self, output: DiffusionAgentLoopOutput, **kwargs
    ) -> _InternalCompositeAgentLoopOutput:
        """Perform post-processing operations on the output of each individual agent loop."""
        output = CompositeAgentLoopOutput(**dict(output))

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

        # It is not used in training for now, but we keep it for future use.
        # For example, we can compose common LLM and T2I models for this rollout pipeline and training,
        # where LLM generated refined pompts used for T2I model image generation.
        extra_fields["text_encoder_responses"] = output.extra_fields["text_encoder_responses"]

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
        llm_response_logprobs = None
        if output.extra_fields.get("llm_all_log_probs", None) is not None:
            llm_response_logprobs = output.extra_fields["llm_all_log_probs"].unsqueeze(0)

        prompt_ids = prompt_output["input_ids"]
        extra_fields["attention_mask"] = prompt_output["attention_mask"]

        await self._compute_score(
            output,
            prompts=prompt_ids,
            responses=response_diffusion_output,
            kwargs=kwargs,
        )

        if "reward_extra_info" in output.extra_fields:
            extra_fields["reward_extra_info"] = output.extra_fields["reward_extra_info"]

        return _InternalCompositeAgentLoopOutput(
            prompt_ids=prompt_ids,
            response_diffusion_output=response_diffusion_output,
            response_logprobs=response_logprobs,
            llm_response_logprobs=llm_response_logprobs,
            reward_score=output.reward_score,
            llm_reward_score=output.llm_reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=extra_fields,
        )

    async def _compute_score(self, output, prompts, responses, kwargs):
        """Compute dual reward scores for single sample.
        Note that either AR or DiT part computes image-grounded rewards
        """

        enable_async_reward = self.reward_loop_worker_handles is not None
        ar_enable_async_reward = self.ar_reward_loop_worker_handles is not None

        if (output.reward_score is None and enable_async_reward) or (
            output.llm_reward_score is None and ar_enable_async_reward
        ):
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
                if output.reward_score is None and enable_async_reward:
                    selected_reward_loop_worker_handle = random.choice(self.reward_loop_worker_handles)
                    result = await selected_reward_loop_worker_handle.compute_score.remote(data)
                    output.reward_score = result["reward_score"]
                    output.extra_fields["reward_extra_info"] = result["reward_extra_info"]
                if output.llm_reward_score is None and ar_enable_async_reward:
                    selected_ar_reward_loop_worker_handle = random.choice(self.ar_reward_loop_worker_handles)
                    result = await selected_ar_reward_loop_worker_handle.compute_score.remote(data)
                    output.llm_reward_score = result["reward_score"]
                    if output.extra_fields["reward_extra_info"] is not None:
                        output.extra_fields["reward_extra_info"].update(result["reward_extra_info"])
                    else:
                        output.extra_fields["reward_extra_info"] = result["reward_extra_info"]

            output.metrics.compute_score = timing["compute_score"]

    def _postprocess(
        self,
        inputs: list[_InternalCompositeAgentLoopOutput],
        input_non_tensor_batch: dict | None = None,
    ) -> DataProto:
        # pack other outputs
        outputs = super()._postprocess(inputs, input_non_tensor_batch)

        # LLM log-probs and scores
        if inputs[0].llm_response_logprobs is not None:
            outputs.batch["rollout_llm_log_probs"] = torch.cat([input.llm_response_logprobs for input in inputs], dim=0)

        llm_scores = [input.llm_reward_score for input in inputs]
        if all(score is not None for score in llm_scores):
            rm_scores = torch.tensor(llm_scores, dtype=torch.float32).unsqueeze(-1)
            outputs.batch["llm_rm_scores"] = rm_scores

        return outputs
