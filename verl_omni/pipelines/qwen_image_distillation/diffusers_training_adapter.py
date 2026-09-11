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
"""Qwen-Image T2I adapter for offline DMD2 distribution matching."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.qwen_image_flow_grpo.common import QwenImageTokenIdPromptMixin
from verl_omni.pipelines.qwen_image_flow_grpo.diffusers_training_adapter import QwenImage

__all__ = ["QwenImageDMD2"]


def build_qwen_dmd_sigmas(num_inference_steps, shift, device):
    """Build the fixed, once-shifted Euler grid shared by training and inference."""
    from verl_omni.trainer.diffusion.distillation.utils import timestep_shift

    if isinstance(num_inference_steps, bool) or not isinstance(num_inference_steps, int) or num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be a positive integer.")
    return timestep_shift(torch.linspace(1, 0, num_inference_steps + 1, device=device), 1, shift)


@DiffusionModelBase.register("QwenImagePipeline", algorithm="dmd2")
class QwenImageDMD2(QwenImage):
    """Base Qwen T2I flow adapter; optimization stays in the DMD engine."""

    @classmethod
    def sampling_sigmas(cls, model_config, dmd_config, device):
        """Use Qwen DMD2's fixed shift rather than the native resolution-dependent shift."""
        return build_qwen_dmd_sigmas(
            model_config.pipeline.num_inference_steps, dmd_config.rollout_timestep_shift, device
        )

    @classmethod
    def configure_train_mode(cls, module):
        """Keep sampling and checkpoint recomputation in evaluation mode with autograd enabled."""
        module.eval()

    @classmethod
    def build_conditioning_provider(cls, model_config, dmd_config):
        """Build the frozen/local-or-cached prompt provider once per engine."""
        return QwenImageConditionProvider(
            model_config.local_path or model_config.path,
            model_config.pipeline.max_sequence_length,
            dmd_config.negative_prompt,
        )

    @staticmethod
    def batch_dimension(batch, key, default):
        """Require same-resolution integer geometry within a physical batch."""
        value = tu.get(batch, key, default)
        values = torch.as_tensor(value).reshape(-1)
        if values.numel() == 0 or not torch.all(values == values[0]) or not torch.all(values == values.long()):
            raise ValueError(f"Qwen DMD2 requires homogeneous integer {key} values.")
        return int(values[0])

    @classmethod
    def latent_geometry(cls, module, model_config, batch):
        """Read VAE geometry and return normalized latent shape plus forward metadata."""
        if len(batch.batch_size) != 1 or batch.batch_size[0] <= 0:
            raise ValueError("Qwen DMD2 requires a nonempty leading batch dimension.")
        with open(Path(model_config.local_path or model_config.path) / "vae" / "config.json") as file:
            vae_config = json.load(file)
        scale = 2 ** len(vae_config["temperal_downsample"])
        channels = vae_config["z_dim"]
        model = getattr(module, "_fsdp_wrapped_module", module)
        if model.config.in_channels != channels * 4:
            raise ValueError("Qwen transformer packing and VAE channel counts do not match.")
        height = cls.batch_dimension(batch, "height", model_config.pipeline.height)
        width = cls.batch_dimension(batch, "width", model_config.pipeline.width)
        if height <= 0 or width <= 0 or height % (scale * 2) or width % (scale * 2):
            raise ValueError(f"Qwen image dimensions must be positive multiples of {scale * 2}.")
        shape = (batch.batch_size[0], channels, 1, height // scale, width // scale)
        return shape, {"height": height, "width": width, "vae_scale_factor": scale}

    @staticmethod
    def pack_latents(latents):
        """Pack declared normalized image latents using the native Qwen helper."""
        from diffusers import QwenImagePipeline

        batch, channels, _, height, width = latents.shape
        return QwenImagePipeline._pack_latents(latents, batch, channels, height, width)

    @classmethod
    def prepare_dmd_inputs(cls, module, model_config, latents, sigma, condition, geometry):
        """Reuse Qwen's input builder without its policy-gradient CFG or SDE step."""
        metadata = TensorDict({}, batch_size=[latents.shape[0]])
        tu.assign_non_tensor(metadata, **geometry)
        sigma = sigma.reshape(-1).expand(latents.shape[0])
        inputs, _ = super().prepare_model_inputs(
            module,
            model_config,
            latents.unsqueeze(1),
            (sigma * 1000).unsqueeze(1),
            condition["prompt_embeds"],
            condition["prompt_embeds_mask"],
            None,
            None,
            metadata,
            0,
        )
        dtype = getattr(module, "dtype", latents.dtype)
        inputs["hidden_states"] = inputs["hidden_states"].to(dtype=dtype)
        inputs["encoder_hidden_states"] = inputs["encoder_hidden_states"].to(dtype=dtype)
        return inputs

    @staticmethod
    def prediction_to_x0(noisy, prediction, sigma):
        """Translate packed Qwen flow velocity to canonical fp32 clean latents."""
        from verl_omni.trainer.diffusion.distillation.utils import velocity_to_x0

        return velocity_to_x0(noisy, prediction, sigma)


class QwenImageConditionProvider:
    """Encode frozen local or precomputed Qwen prompt conditioning."""

    def __init__(
        self,
        model_path: str,
        max_sequence_length: int,
        negative_prompt: str,
    ) -> None:
        self.model_path = model_path
        self.max_sequence_length = max_sequence_length
        self.negative_prompt = negative_prompt
        self.pipeline = None
        self.negative_condition: Optional[dict[str, torch.Tensor]] = None

    @staticmethod
    def make_condition(prompt_embeds: torch.Tensor, prompt_mask: Optional[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Build a detached [B, L, D] condition with a matching [B, L] mask."""
        if prompt_embeds.ndim != 3 or any(size == 0 for size in prompt_embeds.shape):
            raise ValueError(f"Qwen prompt embeddings must have shape [B, L, D], got {tuple(prompt_embeds.shape)}.")
        prompt_embeds = prompt_embeds.detach()
        if prompt_mask is None:
            prompt_mask = torch.ones(prompt_embeds.shape[:2], device=prompt_embeds.device, dtype=torch.long)
        elif prompt_mask.shape != prompt_embeds.shape[:2]:
            raise ValueError(
                f"Qwen prompt mask shape {tuple(prompt_mask.shape)} does not match {tuple(prompt_embeds.shape[:2])}."
            )
        return {"prompt_embeds": prompt_embeds, "prompt_embeds_mask": prompt_mask.detach()}

    @staticmethod
    def require_tensor(batch: TensorDict, key: str) -> torch.Tensor:
        """Require a tensor-valued precomputed conditioning field."""
        value = tu.get(batch, key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Precomputed Qwen conditioning requires tensor batch field {key!r}.")
        return value

    def encode_precomputed(
        self,
        batch: TensorDict,
        *,
        require_negative: bool,
    ) -> tuple[dict[str, torch.Tensor], Optional[dict[str, torch.Tensor]]]:
        """Validate and truncate cached positive and negative conditioning."""
        prompt_embeds = self.require_tensor(batch, "prompt_embeds")[:, : self.max_sequence_length]
        prompt_mask = tu.get(batch, "prompt_embeds_mask")
        if isinstance(prompt_mask, torch.Tensor):
            prompt_mask = prompt_mask[:, : self.max_sequence_length]
        negative_embeds = (
            self.require_tensor(batch, "negative_prompt_embeds")[:, : self.max_sequence_length]
            if require_negative
            else None
        )
        negative_mask = tu.get(batch, "negative_prompt_embeds_mask") if require_negative else None
        if isinstance(negative_mask, torch.Tensor):
            negative_mask = negative_mask[:, : self.max_sequence_length]
        if prompt_mask is not None and not isinstance(prompt_mask, torch.Tensor):
            raise TypeError("prompt_embeds_mask must be a tensor when supplied.")
        if negative_mask is not None and not isinstance(negative_mask, torch.Tensor):
            raise TypeError("negative_prompt_embeds_mask must be a tensor when supplied.")
        positive = self.make_condition(prompt_embeds, prompt_mask)
        negative = self.make_condition(negative_embeds, negative_mask) if negative_embeds is not None else None
        if positive["prompt_embeds"].shape[0] != batch.batch_size[0]:
            raise ValueError("Precomputed Qwen conditioning batch size does not match the input batch.")
        if negative is not None and negative["prompt_embeds"].shape[0] != batch.batch_size[0]:
            raise ValueError("Precomputed Qwen negative conditioning batch size does not match the input batch.")
        return positive, negative

    def ensure_pipeline(self, device: torch.device, dtype: torch.dtype):
        """Load the frozen checkpoint text encoder once on the execution device."""
        if self.pipeline is not None:
            return self.pipeline

        from diffusers import QwenImagePipeline

        pipeline = QwenImagePipeline.from_pretrained(
            self.model_path,
            transformer=None,
            vae=None,
            torch_dtype=dtype,
            local_files_only=os.path.isdir(self.model_path),
        ).to(device)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.text_encoder.eval()
        self.pipeline = pipeline
        return pipeline

    @staticmethod
    def prompt_rows(value: Any, batch_size: int, key: str) -> list[Any]:
        """Unwrap one text or chat-message row per batch sample."""
        if hasattr(value, "tolist") and not isinstance(value, torch.Tensor):
            value = value.tolist()
        if batch_size == 1 and (
            isinstance(value, str) or (isinstance(value, list) and value and isinstance(value[0], dict))
        ):
            return [value]
        if not isinstance(value, list) or len(value) != batch_size:
            raise ValueError(f"{key} must contain exactly {batch_size} prompt row(s).")
        return value

    def tokenize_rows(self, pipeline, rows: list[Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the fixed Qwen template before tokenization and prefix removal."""
        rendered = []
        for row in rows:
            if isinstance(row, list):
                if len(row) != 1 or not isinstance(row[0], dict) or row[0].get("role") != "user":
                    raise ValueError(
                        "Qwen DMD raw prompts require a single user message; use precomputed conditioning otherwise."
                    )
                row = row[0].get("content")
            if not isinstance(row, str):
                raise TypeError("Qwen DMD prompts must be strings or single text-only user messages.")
            # The encoder removes this template's fixed prefix, not a generic chat prefix.
            rendered.append(pipeline.prompt_template_encode.format(row))
        tokens = pipeline.tokenizer(
            rendered,
            max_length=self.max_sequence_length + pipeline.prompt_template_encode_start_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return tokens.input_ids.to(device), tokens.attention_mask.to(device)

    def encode_ids(
        self,
        pipeline,
        prompt_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Reuse Qwen token-ID encoding under no-grad and truncate the result."""
        with torch.no_grad():
            prompt_embeds, prompt_mask = QwenImageTokenIdPromptMixin._get_qwen_prompt_embeds(
                pipeline, prompt_ids, attention_mask=attention_mask
            )
        prompt_embeds = prompt_embeds[:, : self.max_sequence_length]
        if prompt_mask is not None:
            prompt_mask = prompt_mask[:, : self.max_sequence_length]
        if prompt_embeds.shape[1] == 0:
            raise ValueError("Qwen prompt encoding produced no tokens after removing the template prefix.")
        return self.make_condition(prompt_embeds.detach(), prompt_mask.detach() if prompt_mask is not None else None)

    @torch.profiler.record_function("distillation/condition_encode")
    def encode(
        self,
        batch: TensorDict,
        *,
        device: torch.device,
        dtype: torch.dtype,
        require_negative: bool,
    ) -> tuple[dict[str, torch.Tensor], Optional[dict[str, torch.Tensor]]]:
        """Encode the positive prompt and optional teacher negative condition."""
        if "prompt_embeds" in batch:
            return self.encode_precomputed(batch, require_negative=require_negative)

        pipeline = self.ensure_pipeline(device, dtype)
        prompt_ids = tu.get(batch, "prompt_ids", tu.get(batch, "prompts"))
        prompt_mask = tu.get(
            batch,
            "prompt_attention_mask",
            tu.get(batch, "prompt_mask", tu.get(batch, "attention_mask")),
        )
        if prompt_ids is None:
            raw_prompt = tu.get(batch, "raw_prompt")
            if raw_prompt is None:
                raise ValueError("Qwen DMD batches require prompt_ids, prompts, prompt_embeds, or raw_prompt.")
            rows = self.prompt_rows(raw_prompt, batch.batch_size[0], "raw_prompt")
            prompt_ids, prompt_mask = self.tokenize_rows(pipeline, rows, device)
        else:
            if prompt_mask is None:
                raise ValueError("Pre-tokenized Qwen prompts require prompt_attention_mask or attention_mask.")
            if not isinstance(prompt_ids, torch.Tensor):
                prompt_ids = torch.as_tensor(prompt_ids, device=device, dtype=torch.long)
            else:
                prompt_ids = prompt_ids.to(device=device, dtype=torch.long)
        if prompt_mask is not None:
            prompt_mask = torch.as_tensor(prompt_mask, device=device, dtype=torch.long)
        positive = self.encode_ids(pipeline, prompt_ids, prompt_mask)

        if not require_negative:
            return positive, None

        negative_ids = tu.get(batch, "negative_prompt_ids")
        negative_mask = tu.get(batch, "negative_prompt_attention_mask", tu.get(batch, "negative_prompt_mask"))
        if negative_ids is None:
            raw_negative = tu.get(batch, "raw_negative_prompt", tu.get(batch, "negative_prompt"))
            if raw_negative is not None:
                negative_rows = self.prompt_rows(raw_negative, batch.batch_size[0], "negative_prompt")
                negative_ids, negative_mask = self.tokenize_rows(pipeline, negative_rows, device)
                negative = self.encode_ids(pipeline, negative_ids, negative_mask)
            else:
                if self.negative_condition is None:
                    negative_ids, negative_mask = self.tokenize_rows(pipeline, [self.negative_prompt], device)
                    self.negative_condition = self.encode_ids(pipeline, negative_ids, negative_mask)
                negative = self.make_condition(
                    self.negative_condition["prompt_embeds"].expand(batch.batch_size[0], -1, -1),
                    self.negative_condition["prompt_embeds_mask"].expand(batch.batch_size[0], -1),
                )
        else:
            if negative_mask is None:
                raise ValueError("Pre-tokenized negative Qwen prompts require negative_prompt_attention_mask.")
            negative_ids = torch.as_tensor(negative_ids, device=device, dtype=torch.long)
            negative_mask = torch.as_tensor(negative_mask, device=device, dtype=torch.long)
            negative = self.encode_ids(pipeline, negative_ids, negative_mask)
        return positive, negative
