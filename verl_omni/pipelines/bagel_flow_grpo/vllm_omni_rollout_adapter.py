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
"""Custom vllm-omni pipeline for BAGEL RL rollouts with verl-omni.

Extends :class:`BagelPipeline` to:
* Replace the scheduler with an SDE scheduler for stochastic denoising
  with log-probability recording.
* Always enable trajectory recording.
* Read SDE kwargs from ``sampling_params.extra_args``.
* Return RL artifacts in ``DiffusionOutput.custom_output``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.bagel.pipeline_bagel import BagelPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

logger = logging.getLogger(__name__)


_CHAT_MARKERS = (
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
)


def _to_token_list(token_ids: Any) -> list[int] | None:
    if token_ids is None:
        return None
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token_id) for token_id in token_ids]


def _extract_prompt_text(decoded: str) -> str:
    if "<|im_start|>" in decoded:
        user_chunks = []
        for segment in decoded.split("<|im_start|>"):
            if not segment.startswith("user"):
                continue
            content = segment[len("user") :].lstrip("\n")
            content = content.split("<|im_end|>", 1)[0]
            user_chunks.append(content)
        if user_chunks:
            decoded = user_chunks[-1]

    for marker in _CHAT_MARKERS:
        decoded = decoded.replace(marker, "")
    return decoded.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()


def _to_cpu_tensor(v):
    """Convert to a single CPU tensor, stacking a list of tensors if needed."""
    if isinstance(v, torch.Tensor):
        return v.detach().cpu()
    if isinstance(v, list):
        tensors = [x.detach().cpu() if isinstance(x, torch.Tensor) else torch.tensor(x) for x in v]
        return torch.stack(tensors) if tensors else None
    return v


@dataclass
class _AdapterStepOutput:
    """Adapter output matching what bagel_transformer.generate_image expects."""

    prev_sample: torch.Tensor
    log_prob: torch.Tensor | None


class _BagelSchedulerAdapter:
    """Wraps the diffusers-based FlowMatchSDEDiscreteScheduler to match
    BAGEL's calling convention: ``step(v_t, sigma, x_t, dt, **kwargs)``.

    BAGEL's transformer calls ``scheduler.step(model_output, timesteps[i],
    sample, dts[i], **scheduler_kwargs)`` with 4 positional args, while the
    diffusers scheduler takes ``step(model_output, timestep, sample, **kwargs)``
    and computes dt internally.  This adapter bridges the gap.
    """

    def __init__(self, inner: FlowMatchSDEDiscreteScheduler):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def step(
        self,
        model_output: torch.Tensor,
        sigma: float | torch.Tensor,
        sample: torch.Tensor,
        dt: float | torch.Tensor,  # noqa: ARG002 — not used, inner computes from timestep schedule
        **kwargs,
    ) -> _AdapterStepOutput:
        out = self._inner.step(
            model_output=model_output,
            timestep=sigma,
            sample=sample,
            return_dict=False,
            **kwargs,
        )
        # step() with return_dict=False returns (prev_sample, log_prob, prev_sample_mean, std_dev_t)
        prev_sample, log_prob = out[0], out[1]
        return _AdapterStepOutput(prev_sample=prev_sample, log_prob=log_prob)


@VllmOmniPipelineBase.register("OmniBagelForConditionalGeneration", algorithm="flow_grpo")
class BagelPipelineWithLogProb(BagelPipeline):
    """BAGEL pipeline variant for RL rollouts with verl-omni."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        inner = FlowMatchSDEDiscreteScheduler()
        self.scheduler = _BagelSchedulerAdapter(inner)
        logger.info("BagelPipelineWithLogProb: SDE scheduler enabled for RL rollouts")

    def _decode_token_prompt(self, token_ids: Any) -> str | None:
        token_list = _to_token_list(token_ids)
        if not token_list:
            return None
        decoded = self.tokenizer.decode(token_list, skip_special_tokens=False)
        return _extract_prompt_text(decoded)

    def _ensure_bagel_prompt_text(self, req: OmniDiffusionRequest) -> None:
        if not req.prompts or not isinstance(req.prompts[0], dict):
            return

        custom_prompt = req.prompts[0]
        if not custom_prompt.get("prompt"):
            prompt = self._decode_token_prompt(custom_prompt.get("prompt_token_ids"))
            if prompt is not None:
                custom_prompt["prompt"] = prompt

        extra_args = req.sampling_params.extra_args
        if "negative_prompt" not in extra_args:
            negative_prompt = self._decode_token_prompt(custom_prompt.get("negative_prompt_ids"))
            if negative_prompt is not None:
                extra_args["negative_prompt"] = negative_prompt

        prompt_extra_args = custom_prompt.get("extra_args")
        if isinstance(prompt_extra_args, dict):
            multi_modal_data = prompt_extra_args.get("multi_modal_data")
            if multi_modal_data is not None and "multi_modal_data" not in custom_prompt:
                custom_prompt["multi_modal_data"] = multi_modal_data

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        self._ensure_bagel_prompt_text(req)

        # Force trajectory recording on for RL
        req.sampling_params.return_trajectory_latents = True

        # Read SDE scheduler kwargs from extra_args
        extra_args = req.sampling_params.extra_args
        logprobs = extra_args.get("logprobs", True)
        self.scheduler_kwargs = {k: extra_args[k] for k in ("noise_level", "sde_type", "generator") if k in extra_args}
        self.scheduler_kwargs["return_logprobs"] = logprobs

        # Per-request scheduler setup: compute BAGEL's shifted sigmas so
        # the inner SDE scheduler's sigma schedule matches what
        # generate_image() computes internally.
        assert req.sampling_params.num_inference_steps is not None, "num_inference_steps must be set for RL rollouts"
        num_timesteps = req.sampling_params.num_inference_steps
        timestep_shift = 3.0  # must match BagelPipeline.forward() hardcoded value

        t = np.linspace(1, 0, num_timesteps)
        t_shifted = timestep_shift * t / (1 + (timestep_shift - 1) * t)
        sigmas = t_shifted[:-1].tolist()  # drop terminal 0; set_timesteps appends it

        inner = self.scheduler._inner
        inner.set_shift(1.0)  # identity — sigmas already shifted
        inner.set_timesteps(sigmas=sigmas)
        inner.set_begin_index(0)

        output = super().forward(req)

        # Enrich custom_output with RL-specific fields (must be tensors for batch stacking)
        custom = output.custom_output or {}
        if output.trajectory_latents is not None:
            custom["all_latents"] = _to_cpu_tensor(output.trajectory_latents)
        if output.trajectory_timesteps is not None:
            custom["all_timesteps"] = _to_cpu_tensor(output.trajectory_timesteps)
        if output.trajectory_log_probs is not None:
            custom["all_log_probs"] = _to_cpu_tensor(output.trajectory_log_probs)
        output.custom_output = custom

        return output
