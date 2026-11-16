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

"""LTX-2.3 joint audio-video training adapter for OmniNFT."""

from pathlib import Path
from typing import Any, Callable, Optional

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, ModelMixin
from tensordict import TensorDict
from torch.utils._pytree import tree_map
from torch.utils.checkpoint import checkpoint
from verl.utils.device import get_device_name

from verl_omni.pipelines.ltx2_flow_grpo.common import apply_x0_cfg, set_ltx23_timesteps
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config import DiffusionModelConfig

__all__ = ["LTX23OmniNFT"]


@DiffusionModelBase.register("LTX2Pipeline", algorithm="omni_nft")
class LTX23OmniNFT(DiffusionModelBase):
    """Return separate video/audio velocities from one joint LTX forward."""

    @staticmethod
    def validate_engine_config(engine_config) -> None:
        """Reject unsupported sharding and sequence parallelism before initialization."""
        if engine_config.strategy != "fsdp2":
            raise NotImplementedError(
                f"OmniNFT currently supports only actor.strategy=fsdp2, got {engine_config.strategy!r}."
            )
        if engine_config.ulysses_sequence_parallel_size != 1:
            raise NotImplementedError(
                "OmniNFT FSDP2 does not implement Ulysses/context parallelism yet; "
                "set actor.fsdp_config.ulysses_sequence_parallel_size=1."
            )

    @classmethod
    def configure_gradient_checkpointing(cls, module, engine_config) -> None:
        """Configure the FSDP2 checkpoint wrapper only for OmniNFT training."""
        from verl.utils.torch_dtypes import PrecisionType

        precision = engine_config.mixed_precision
        param_dtype = PrecisionType.to_dtype(precision.get("param_dtype", "bf16") if precision is not None else "bf16")
        if getattr(module, "_keep_in_fp32_modules", None) and cls.preserve_fp32_modules():
            param_dtype = None
        module.enable_gradient_checkpointing(gradient_checkpointing_func=_gradient_checkpointing_with_cast(param_dtype))

    @classmethod
    def prepare_training_inputs(
        cls, micro_batch: TensorDict, step: int, scheduler
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Noise modalities independently and retain their context separately."""
        timestep, time = cls._flow_time(micro_batch, step, scheduler)
        context = {
            modality: cls._noise_latents(
                micro_batch, f"{modality}_latents_clean", f"{modality}_forward_noise", step, time
            )
            for modality in ("video", "audio")
        }
        latents = torch.cat((context["video"]["xt"], context["audio"]["xt"]), dim=1)
        return latents, timestep, context

    @staticmethod
    def prepare_model_outputs(predictions: dict, loss_context: dict) -> dict:
        """Map joint model predictions and modality context to OmniNFT loss fields."""
        outputs = {
            f"{modality}_{name}": value for modality, context in loss_context.items() for name, value in context.items()
        }
        for name, values in predictions.items():
            for modality, value in zip(("video", "audio"), values, strict=True):
                outputs[f"{modality}_{name}"] = value
        return outputs

    @classmethod
    def prepare_processor_files(cls, model_path: str) -> str:
        """Return the tokenizer directory used to construct the LTX processor."""
        tokenizer_dir = Path(model_path) / "tokenizer"
        if not tokenizer_dir.is_dir():
            raise FileNotFoundError(f"LTX-2.3 tokenizer directory not found: {tokenizer_dir}")
        return str(tokenizer_dir)

    @classmethod
    def set_timesteps(
        cls,
        scheduler: FlowMatchEulerDiscreteScheduler,
        model_config: DiffusionModelConfig,
        device: str,
    ) -> None:
        """Match the LTX-2.3 schedule used by the pinned vLLM-Omni runtime."""
        set_ltx23_timesteps(scheduler, model_config.pipeline.num_inference_steps, device)

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig) -> FlowMatchEulerDiscreteScheduler:
        """Load LTX's native deterministic flow-matching scheduler."""
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_config.local_path,
            subfolder="scheduler",
        )
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @staticmethod
    def _guidance_scales(model_config: DiffusionModelConfig) -> tuple[float, float]:
        """Use common CFG for both modalities, replacing None/zero by 1."""
        scale = float(model_config.pipeline.guidance_scale or 1.0)
        return scale, scale

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
        """Split packed noised tokens and build positive/optional CFG transformer inputs.

        Both modalities use the supplied model-scale timesteps. The clean video
        tensor supplies the split position for the dense actor micro-batch.
        """
        video_seq_len = micro_batch["video_latents_clean"].shape[1]
        video_latents = latents[:, :video_seq_len]
        audio_latents = latents[:, video_seq_len:]
        video_guidance_scale, audio_guidance_scale = cls._guidance_scales(model_config)
        common = {
            "hidden_states": video_latents,
            "audio_hidden_states": audio_latents,
            "timestep": timesteps,
            "sigma": timesteps,
            "num_frames": (model_config.pipeline.num_frames - 1) // 8 + 1,
            "height": model_config.pipeline.height // 32,
            "width": model_config.pipeline.width // 32,
            "fps": model_config.pipeline.frame_rate,
            "audio_num_frames": audio_latents.shape[1],
            "return_dict": False,
        }
        model_inputs = {
            **common,
            "encoder_hidden_states": prompt_embeds,
            "audio_encoder_hidden_states": micro_batch["audio_prompt_embeds"],
            "encoder_attention_mask": prompt_embeds_mask,
            "audio_encoder_attention_mask": prompt_embeds_mask,
        }
        if video_guidance_scale <= 1.0 and audio_guidance_scale <= 1.0:
            return model_inputs, None
        if negative_prompt_embeds is None or negative_prompt_embeds_mask is None:
            raise ValueError("LTX-2.3 OmniNFT CFG requires negative prompt embeddings and attention masks.")
        negative_model_inputs = {
            **common,
            "encoder_hidden_states": negative_prompt_embeds,
            "audio_encoder_hidden_states": micro_batch["negative_audio_prompt_embeds"],
            "encoder_attention_mask": negative_prompt_embeds_mask,
            "audio_encoder_attention_mask": negative_prompt_embeds_mask,
        }
        return model_inputs, negative_model_inputs

    @classmethod
    def forward(
        cls,
        module: ModelMixin,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict paired FP32 velocities with optional modality CFG, preserving gradients."""
        video_prediction, audio_prediction = cls._predict(module, model_inputs)

        video_guidance_scale, audio_guidance_scale = cls._guidance_scales(model_config)
        if video_guidance_scale > 1.0 or audio_guidance_scale > 1.0:
            if negative_model_inputs is None:
                raise ValueError("LTX-2.3 OmniNFT CFG requires negative model inputs.")
            negative_video_prediction, negative_audio_prediction = cls._predict(module, negative_model_inputs)
            sigma = (model_inputs["timestep"].float() / 1000.0).view(-1, 1, 1)
            if video_guidance_scale > 1.0:
                video_prediction = apply_x0_cfg(
                    model_inputs["hidden_states"].float(),
                    video_prediction,
                    negative_video_prediction,
                    sigma,
                    video_guidance_scale,
                )
            if audio_guidance_scale > 1.0:
                audio_prediction = apply_x0_cfg(
                    model_inputs["audio_hidden_states"].float(),
                    audio_prediction,
                    negative_audio_prediction,
                    sigma,
                    audio_guidance_scale,
                )

        return video_prediction, audio_prediction

    @staticmethod
    def _predict(module: ModelMixin, model_inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(video_prediction, audio_prediction)`` in FP32 without detaching."""
        video_prediction, audio_prediction = module(**model_inputs)
        return video_prediction.float(), audio_prediction.float()

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module: ModelMixin,
        scheduler: FlowMatchEulerDiscreteScheduler,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ):
        """Reject the reverse-transition API, which is not part of OmniNFT."""
        raise NotImplementedError("LTX-2.3 OmniNFT does not sample reverse transitions or compute their log-probs.")


def _gradient_checkpointing_with_cast(param_dtype: Optional[torch.dtype]) -> Callable:
    """Build a non-reentrant checkpoint wrapper using the FSDP parameter dtype."""

    def cast_fp_tensor(value):
        if (
            param_dtype is None
            or not isinstance(value, torch.Tensor)
            or not torch.is_floating_point(value)
            or value.dtype == param_dtype
        ):
            return value
        return value.to(param_dtype)

    def gradient_checkpointing_func(module, *args, **kwargs):
        def checkpointed_forward(*inner_args, **inner_kwargs):
            return module.__call__(*tree_map(cast_fp_tensor, inner_args), **tree_map(cast_fp_tensor, inner_kwargs))

        return checkpoint(checkpointed_forward, *args, use_reentrant=False, **kwargs)

    return gradient_checkpointing_func
