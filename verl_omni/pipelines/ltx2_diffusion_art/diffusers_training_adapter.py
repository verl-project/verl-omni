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
"""LTX-2.3 joint audio-video training adapter for DiffusionART."""

from typing import Optional

import torch
from diffusers import ModelMixin
from tensordict import TensorDict

from verl_omni.pipelines.ltx2_flow_grpo.common import apply_x0_cfg
from verl_omni.pipelines.ltx2_flow_grpo.diffusers_training_adapter import LTX23FlowGRPO
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config import DiffusionModelConfig

from .config import ltx_guidance_scales

__all__ = ["LTX23DiffusionART"]


def _shared_int(value: torch.Tensor, name: str) -> int:
    values = value.reshape(-1)
    if values.numel() == 0 or not torch.all(values == values[0]):
        raise ValueError(f"LTX-2.3 DiffusionART requires one shared {name} per micro-batch, got {values.tolist()}.")
    return int(values[0].item())


@DiffusionModelBase.register("LTX2Pipeline", algorithm="diffusion_art")
class LTX23DiffusionART(LTX23FlowGRPO):
    """Joint video/audio velocity at one cached ODE state (text-to-audio-video only).

    ``latents`` is the unified ``[video | audio]`` token sequence the rollout visited, split at
    ``video_seq_len``. The forward returns the same unified layout so ``DiffusionARTLoss`` can
    apply per-modality guidance. CFG is applied in clean-sample space per modality exactly as the
    vLLM-Omni rollout does, with ``video_cfg_scale`` / ``audio_cfg_scale`` falling back to
    ``guidance_scale``; the trainer checks that these equal the rollout's.
    """

    @classmethod
    def prepare_model_inputs(
        cls,
        module: ModelMixin,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_embeds_mask: Optional[torch.Tensor],
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, Optional[dict]]:
        """Split one unified state and build positive (and CFG negative) transformer inputs."""
        del step
        for key in ("audio_prompt_embeds", "video_seq_len"):
            if key not in micro_batch:
                raise KeyError(f"LTX-2.3 DiffusionART rollout is missing required field `{key}`.")
        condition = micro_batch.get("condition_image_latents", None)
        if condition is not None and condition.shape[1] > 0:
            raise NotImplementedError(
                "LTX-2.3 DiffusionART supports text-to-audio-video only (no TI2VA condition rows)."
            )

        video_seq_len = _shared_int(micro_batch["video_seq_len"], "video_seq_len")
        video_latents = latents[:, :video_seq_len]
        audio_latents = latents[:, video_seq_len:]
        pipeline = model_config.pipeline
        common = {
            "hidden_states": video_latents,
            "audio_hidden_states": audio_latents,
            "timestep": timesteps[:, None].expand(-1, video_latents.shape[1]),
            "audio_timestep": timesteps[:, None].expand(-1, audio_latents.shape[1]),
            "sigma": timesteps,
            "audio_sigma": timesteps,
            "num_frames": (pipeline.num_frames - 1) // 8 + 1,
            "height": pipeline.height // 32,
            "width": pipeline.width // 32,
            "fps": pipeline.frame_rate,
            "audio_num_frames": audio_latents.shape[1],
            "return_dict": False,
        }
        # LTX connectors replace padding with learned registers: all context rows attend (as in vLLM-Omni).
        model_inputs = {
            **common,
            "encoder_hidden_states": prompt_embeds,
            "audio_encoder_hidden_states": micro_batch["audio_prompt_embeds"],
            "encoder_attention_mask": None,
            "audio_encoder_attention_mask": None,
        }
        video_scale, audio_scale = ltx_guidance_scales(pipeline)
        if video_scale <= 1.0 and audio_scale <= 1.0:
            return model_inputs, None
        if negative_prompt_embeds is None or "negative_audio_prompt_embeds" not in micro_batch:
            raise ValueError(
                "LTX-2.3 DiffusionART CFG requires negative video and audio prompt embeddings from rollout."
            )
        negative_model_inputs = {
            **common,
            "encoder_hidden_states": negative_prompt_embeds,
            "audio_encoder_hidden_states": micro_batch["negative_audio_prompt_embeds"],
            "encoder_attention_mask": None,
            "audio_encoder_attention_mask": None,
        }
        return model_inputs, negative_model_inputs

    @classmethod
    def forward(
        cls,
        module: ModelMixin,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Unified ``[video | audio]`` FP32 velocity with per-modality clean-space CFG, gradients preserved."""
        video_pred, audio_pred = cls._predict(module, model_inputs)
        video_scale, audio_scale = ltx_guidance_scales(model_config.pipeline)
        if video_scale > 1.0 or audio_scale > 1.0:
            if negative_model_inputs is None:
                raise ValueError("LTX-2.3 DiffusionART CFG requires negative model inputs.")
            negative_video_pred, negative_audio_pred = cls._predict(module, negative_model_inputs)
            sigma = (model_inputs["sigma"].float() / 1000.0).view(-1, 1, 1)
            if video_scale > 1.0:
                video_pred = apply_x0_cfg(
                    model_inputs["hidden_states"].float(), video_pred, negative_video_pred, sigma, video_scale
                )
            if audio_scale > 1.0:
                audio_pred = apply_x0_cfg(
                    model_inputs["audio_hidden_states"].float(), audio_pred, negative_audio_pred, sigma, audio_scale
                )
        return torch.cat([video_pred, audio_pred], dim=1)

    @classmethod
    def forward_and_sample_previous_step(cls, *args, **kwargs):
        """Reject the reverse-transition API; DiffusionART regresses velocities, it does not replay transitions."""
        raise NotImplementedError("LTX-2.3 DiffusionART does not sample reverse transitions or compute log-probs.")
