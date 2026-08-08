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

"""VeOmni training adapter for LTX-2.3 FlowGRPO.

Registered with ``backend="veomni"`` so that ``DiffusionModelBase.get_class``
picks it automatically when ``model_config.backend == "veomni"``. It targets
VeOmni's ``LTX2VideoTransformer3DModel`` (built by the VeOmni engine via
``veomni.trainer.dit_trainer.DiTTrainer``) whose ``forward()`` signature differs
materially from the diffusers ``LTXVideoTransformerModel``:

* inputs are **per-sample lists** (length == batch size, each element with
  batch dim == 1) rather than a single batched tensor;
* ``hidden_states`` elements are 5-D ``(1, C, F, H, W)`` so H/W/F are inferred
  from the tensor shape — ``height``/``width``/``num_frames`` are not accepted;
* the audio timestep is passed via ``audio_timestep`` (there is no ``sigma``
  argument);
* the output is a dataclass with ``.predictions`` / ``.audio_predictions``
  lists rather than a ``(video, audio)`` tuple.

The diffusers/FSDP counterpart lives in :mod:`diffusers_training_adapter` and
is the ``backend=None`` default.
"""

from typing import Optional

import numpy as np
import torch
from diffusers import ModelMixin
from tensordict import TensorDict
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

from .common import apply_x0_cfg, calculate_shift, remap_veomni_to_diffusers_key

__all__ = ["LTX23FlowGRPOVeOmni"]


def _single_int(value: torch.Tensor, name: str) -> int:
    values = value.reshape(-1)
    if values.numel() == 0 or not torch.all(values == values[0]):
        raise ValueError(f"LTX-2.3 requires one shared {name} per micro-batch, got {values.tolist()}.")
    return int(values[0].item())


@DiffusionModelBase.register("LTX2Pipeline", algorithm="flow_grpo", backend="veomni")
class LTX23FlowGRPOVeOmni(DiffusionModelBase):
    """Recompute joint audio-video transition probabilities with VeOmni's LTX2 transformer."""

    @classmethod
    def convert_export_key(cls, name: str) -> str:
        """Remap VeOmni parameter names to diffusers naming for the rollout loader.

        VeOmni's ``LTX2VideoTransformer3DModel`` uses different parameter names
        (e.g. ``adaln_single`` → ``time_embed``, ``q_norm`` → ``norm_q``) than the
        vLLM-Omni rollout model (diffusers ``LTXVideoTransformerModel``).  This
        ensures the actor's state-dict keys match what the rollout expects.
        """
        return remap_veomni_to_diffusers_key(name)

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
        """Match the LTX-2.3 diffusers/vLLM-Omni sigma schedule."""
        num_steps = model_config.pipeline.num_inference_steps
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        latent_frames = (model_config.pipeline.num_frames - 1) // 8 + 1
        latent_height = model_config.pipeline.height // 32
        latent_width = model_config.pipeline.width // 32
        video_seq_len = latent_frames * latent_height * latent_width
        mu = calculate_shift(
            video_seq_len,
            scheduler.config.get("base_image_seq_len", 1024),
            scheduler.config.get("max_image_seq_len", 4096),
            scheduler.config.get("base_shift", 0.95),
            scheduler.config.get("max_shift", 2.05),
        )
        scheduler.set_timesteps(num_steps, device=device, sigmas=sigmas, mu=mu)

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

        B = video_latents.shape[0]
        C = video_latents.shape[2]
        video_latents_5d = video_latents.reshape(B, latent_frames, latent_height, latent_width, C).permute(0, 4, 1, 2, 3)

        audio_C = 8
        audio_mel_bins = 16
        audio_latents_4d = audio_latents.reshape(B, -1, audio_C, audio_mel_bins).permute(0, 2, 1, 3)

        # VeOmni's LTX2VideoTransformer3DModel.forward expects per-sample lists
        # (list length == batch size, each element with batch dim == 1). Passing a
        # bare tensor makes ``zip`` iterate the batch dimension, yielding 4D
        # tensors and triggering ``IndexError: tuple index out of range``.
        timestep_list = [timestep[i : i + 1] for i in range(B)]
        common = {
            "hidden_states": [video_latents_5d[i : i + 1] for i in range(B)],
            "audio_hidden_states": [audio_latents_4d[i : i + 1] for i in range(B)],
            "timestep": timestep_list,
            "audio_timestep": timestep_list,
            "fps": [frame_rate] * B,
        }
        model_inputs = {
            **common,
            "encoder_hidden_states": [prompt_embeds[i : i + 1] for i in range(B)],
            "audio_encoder_hidden_states": [micro_batch["audio_prompt_embeds"][i : i + 1] for i in range(B)],
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
            "encoder_hidden_states": [negative_prompt_embeds[i : i + 1] for i in range(B)],
            "audio_encoder_hidden_states": [
                micro_batch["negative_audio_prompt_embeds"][i : i + 1] for i in range(B)
            ],
        }
        return model_inputs, negative_model_inputs

    @staticmethod
    def _predict(module: ModelMixin, model_inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the LTX transformer and return float32 video/audio velocities in 3D format."""
        output = module(**model_inputs)
        if isinstance(output, tuple):
            video_pred, audio_pred = output
        else:
            preds = output.predictions or []
            video_pred = torch.cat(preds, dim=0) if preds else preds
            audio_preds = output.audio_predictions or []
            audio_pred = torch.cat(audio_preds, dim=0) if audio_preds else None

        video_pred = video_pred.float()
        audio_pred = audio_pred.float() if audio_pred is not None else None

        if video_pred.ndim == 5:
            B, C, F, H, W = video_pred.shape
            video_pred = video_pred.permute(0, 2, 3, 4, 1).reshape(B, F * H * W, C)

        if audio_pred is not None and audio_pred.ndim == 4:
            B, C, F, M = audio_pred.shape
            audio_pred = audio_pred.permute(0, 2, 1, 3).reshape(B, F, C * M)

        return video_pred, audio_pred

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

        # ``model_inputs["hidden_states"]`` (and friends) are per-sample lists
        # built in ``prepare_model_inputs`` because VeOmni's transformer
        # expects that format. Rebatch them into single tensors here.
        video_latents = torch.cat(model_inputs["hidden_states"], dim=0).float()
        audio_latents = torch.cat(model_inputs["audio_hidden_states"], dim=0).float()
        video_pred, audio_pred = cls._predict(module, model_inputs)

        if video_latents.ndim == 5:
            B, C, F, H, W = video_latents.shape
            video_latents = video_latents.permute(0, 2, 3, 4, 1).reshape(B, F * H * W, C)

        if audio_latents.ndim == 4:
            B, C, F, M = audio_latents.shape
            audio_latents = audio_latents.permute(0, 2, 1, 3).reshape(B, F, C * M)

        guidance_scale = model_config.pipeline.guidance_scale or 1.0
        if guidance_scale > 1.0:
            if negative_model_inputs is None:
                raise ValueError("LTX-2.3 CFG requires negative model inputs.")
            negative_video_pred, negative_audio_pred = cls._predict(module, negative_model_inputs)
            sigma = (torch.cat(model_inputs["timestep"], dim=0).float() / 1000.0).view(-1, 1, 1)
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
