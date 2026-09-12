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

"""Diffusers + FSDP2 training adapter for LTX-2.3 FlowGRPO."""

from pathlib import Path
from typing import Optional

import torch
from diffusers import ModelMixin
from tensordict import TensorDict
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionI2IModelBase, DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

from .common import apply_x0_cfg, set_ltx23_timesteps

__all__ = ["LTX23FlowGRPO"]


def _single_int(value: torch.Tensor, name: str) -> int:
    values = value.reshape(-1)
    if values.numel() == 0 or not torch.all(values == values[0]):
        raise ValueError(f"LTX-2.3 requires one shared {name} per micro-batch, got {values.tolist()}.")
    return int(values[0].item())


@DiffusionModelBase.register("LTX2Pipeline", algorithm="flow_grpo")
class LTX23FlowGRPO(DiffusionI2IModelBase):
    """Recompute text- or first-frame-conditioned audio-video transitions."""

    @classmethod
    def prepare_processor_files(cls, model_path: str) -> str:
        """Use the text tokenizer path because LTX transports images outside its prompt encoder."""
        tokenizer_dir = Path(model_path) / "tokenizer"
        if not tokenizer_dir.is_dir():
            raise FileNotFoundError(f"LTX-2.3 tokenizer directory not found: {tokenizer_dir}")
        return str(tokenizer_dir)

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig) -> FlowMatchSDEDiscreteScheduler:
        """Load and configure the LTX flow-matching SDE scheduler."""
        scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(model_config.local_path, subfolder="scheduler")
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(
        cls,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        device: str,
    ) -> None:
        """Match the LTX-2.3 schedule used by the pinned vLLM-Omni runtime."""
        set_ltx23_timesteps(scheduler, model_config.pipeline.num_inference_steps, device)

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
        """Split the unified trajectory and build the joint transformer inputs."""
        required = ["audio_prompt_embeds", "video_seq_len", "all_next_latents"]
        missing = [key for key in required if key not in micro_batch]
        if missing:
            raise KeyError(f"LTX-2.3 FlowGRPO rollout is missing required fields: {missing}.")

        current = latents[:, step]
        timestep = timesteps[:, step]
        video_seq_len = _single_int(micro_batch["video_seq_len"], "video_seq_len")
        video_latents = current[:, :video_seq_len]
        audio_latents = current[:, video_seq_len:]

        latent_frames = (model_config.pipeline.num_frames - 1) // 8 + 1
        latent_height = model_config.pipeline.height // 32
        latent_width = model_config.pipeline.width // 32
        frame_rate = model_config.pipeline.frame_rate

        common = {
            "hidden_states": video_latents,
            "audio_hidden_states": audio_latents,
            "timestep": timestep[:, None].expand(-1, video_latents.shape[1]),
            "audio_timestep": timestep[:, None].expand(-1, audio_latents.shape[1]),
            "sigma": timestep,
            "audio_sigma": timestep,
            "num_frames": latent_frames,
            "height": latent_height,
            "width": latent_width,
            "fps": frame_rate,
            "audio_num_frames": audio_latents.shape[1],
            "return_dict": False,
            "_require_image_condition": getattr(model_config.pipeline, "task", None) == "ti2va",
        }
        model_inputs = {
            **common,
            "encoder_hidden_states": prompt_embeds,
            "audio_encoder_hidden_states": micro_batch["audio_prompt_embeds"],
            "encoder_attention_mask": None,
            "audio_encoder_attention_mask": None,
        }

        guidance_scale = model_config.pipeline.guidance_scale or 1.0
        if guidance_scale <= 1.0:
            return model_inputs, None
        if negative_prompt_embeds is None or negative_prompt_embeds_mask is None:
            raise ValueError("LTX-2.3 CFG requires negative prompt embeddings and attention masks.")
        if "negative_audio_prompt_embeds" not in micro_batch:
            raise KeyError("LTX-2.3 CFG requires `negative_audio_prompt_embeds` from rollout.")
        negative_model_inputs = {
            **common,
            "encoder_hidden_states": negative_prompt_embeds,
            "audio_encoder_hidden_states": micro_batch["negative_audio_prompt_embeds"],
            "encoder_attention_mask": None,
            "audio_encoder_attention_mask": None,
        }
        return model_inputs, negative_model_inputs

    @classmethod
    def prepare_condition(
        cls,
        micro_batch: TensorDict,
        latents: torch.Tensor,
        step: int,
    ) -> dict[str, torch.Tensor] | None:
        """Read the fixed first-frame latent captured by rollout."""
        del latents, step
        image_latents = micro_batch.get("condition_image_latents")
        if image_latents is None:
            return None
        return {"image_latents": image_latents}

    @classmethod
    def inject_condition(
        cls,
        model_inputs: dict,
        negative_model_inputs: Optional[dict],
        condition: Optional[dict],
    ) -> tuple[dict, Optional[dict]]:
        """Prepend the fixed first-frame rows and assign them timestep zero."""
        if not condition:
            return model_inputs, negative_model_inputs
        image_latents = condition.get("image_latents")
        if not isinstance(image_latents, torch.Tensor) or image_latents.ndim != 3:
            raise ValueError("LTX-2.3 condition_image_latents must have shape (batch, rows, width).")

        for inputs in (model_inputs, negative_model_inputs):
            if inputs is None:
                continue
            target = inputs["hidden_states"]
            if image_latents.shape[0] != target.shape[0] or image_latents.shape[2] != target.shape[2]:
                raise ValueError("LTX-2.3 condition_image_latents must match the target video batch and width.")
            condition_rows = image_latents.to(device=target.device, dtype=target.dtype)
            if inputs.get("_require_image_condition", False) and condition_rows.shape[1] == 0:
                raise ValueError("LTX-2.3 TI2VA requires condition_image_latents from rollout.")
            expected_condition_rows = int(inputs["height"]) * int(inputs["width"])
            if condition_rows.shape[1] not in (0, expected_condition_rows):
                raise ValueError(
                    "LTX-2.3 TI2VA requires exactly one first-frame latent; "
                    f"expected {expected_condition_rows} rows, got {condition_rows.shape[1]}."
                )
            expected_video_rows = int(inputs["num_frames"]) * expected_condition_rows
            if condition_rows.shape[1] + target.shape[1] != expected_video_rows:
                raise ValueError("LTX-2.3 condition and target rows do not reconstruct the configured video geometry.")
            inputs["hidden_states"] = torch.cat([condition_rows, target], dim=1)
            inputs["timestep"] = torch.cat(
                [inputs["timestep"].new_zeros(target.shape[0], condition_rows.shape[1]), inputs["timestep"]],
                dim=1,
            )
            inputs["_condition_video_seq_len"] = condition_rows.shape[1]
        return model_inputs, negative_model_inputs

    @staticmethod
    def _predict(module: ModelMixin, model_inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the LTX transformer and return target-only float32 velocities."""
        model_inputs = dict(model_inputs)
        condition_rows = int(model_inputs.pop("_condition_video_seq_len", 0))
        model_inputs.pop("_require_image_condition", None)
        video_pred, audio_pred = module(**model_inputs)
        return video_pred[:, condition_rows:].float(), audio_pred.float()

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module: ModelMixin,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ):
        """Recompute one selected CPS/SDE transition and its joint log-probability."""
        if scheduler_inputs is None:
            raise ValueError("LTX-2.3 FlowGRPO requires rollout scheduler inputs.")

        condition_rows = int(model_inputs.get("_condition_video_seq_len", 0))
        video_latents = model_inputs["hidden_states"][:, condition_rows:].float()
        audio_latents = model_inputs["audio_hidden_states"].float()
        video_pred, audio_pred = cls._predict(module, model_inputs)

        guidance_scale = model_config.pipeline.guidance_scale or 1.0
        if guidance_scale > 1.0:
            if negative_model_inputs is None:
                raise ValueError("LTX-2.3 CFG requires negative model inputs.")
            negative_video_pred, negative_audio_pred = cls._predict(module, negative_model_inputs)
            sigma = (model_inputs["sigma"].float() / 1000.0).view(-1, 1, 1)
            video_pred = apply_x0_cfg(video_latents, video_pred, negative_video_pred, sigma, guidance_scale)
            audio_pred = apply_x0_cfg(audio_latents, audio_pred, negative_audio_pred, sigma, guidance_scale)

        current = torch.cat([video_latents, audio_latents], dim=1)
        model_output = torch.cat([video_pred, audio_pred], dim=1)
        next_sample = scheduler_inputs["all_next_latents"][:, step].float()
        timestep = scheduler_inputs["all_timesteps"][:, step]
        _, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = scheduler.sample_previous_step(
            sample=current,
            model_output=model_output,
            timestep=timestep,
            noise_level=model_config.algo.noise_level,
            prev_sample=next_sample,
            sde_type=model_config.algo.sde_type,
            return_logprobs=True,
            return_sqrt_dt=True,
        )
        return log_prob, prev_sample_mean, std_dev_t, sqrt_dt
