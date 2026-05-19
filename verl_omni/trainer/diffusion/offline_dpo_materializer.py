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

"""Materialize offline DPO image/prompt rows into diffusion training tensors."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from PIL import Image
from verl import DataProto


def _to_list(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return "" if value is None else str(value)


class OfflineDPOMaterializer:
    """Lazy SD3 VAE/text-encoder materializer for offline DPO batches."""

    def __init__(self, config):
        self.config = config
        self.pipe = None
        self.device = config.get("materialize_device", None)
        if self.device is None:
            self.device = config.trainer.device if torch.cuda.is_available() else "cpu"
        self.torch_dtype = self._resolve_dtype(config.actor_rollout_ref.rollout.get("dtype", "bfloat16"))

    @staticmethod
    def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
        if isinstance(dtype, torch.dtype):
            return dtype
        return {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(str(dtype).lower(), torch.bfloat16)

    def _load_pipe(self):
        if self.pipe is not None:
            return self.pipe

        from diffusers import StableDiffusion3Pipeline

        model_path = self.config.actor_rollout_ref.model.get("local_path", None)
        if model_path is None:
            model_path = self.config.actor_rollout_ref.model.path
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_path,
            transformer=None,
            torch_dtype=self.torch_dtype,
        )
        self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        for component_name in (
            "text_encoder",
            "text_encoder_2",
            "text_encoder_3",
            "vae",
        ):
            component = getattr(self.pipe, component_name, None)
            if component is not None:
                component.requires_grad_(False)
                component.eval()
        return self.pipe

    def _encode_images(self, image_paths: list[str]) -> torch.Tensor:
        pipe = self._load_pipe()
        height = self.config.actor_rollout_ref.rollout.pipeline.height
        width = self.config.actor_rollout_ref.rollout.pipeline.width
        images = []
        for image_path in image_paths:
            path = os.path.expanduser(_as_text(image_path))
            image = Image.open(path).convert("RGB")
            images.append(image)

        pixel_values = pipe.image_processor.preprocess(images, height=height, width=width)
        pixel_values = pixel_values.to(device=self.device, dtype=pipe.vae.dtype)
        latents = pipe.vae.encode(pixel_values).latent_dist.sample()
        scaling_factor = getattr(pipe.vae.config, "scaling_factor", 1.0)
        shift_factor = getattr(pipe.vae.config, "shift_factor", 0.0)
        latents = (latents - shift_factor) * scaling_factor
        return latents.detach().cpu()

    def _encode_prompts(
        self, prompts: list[str], negative_prompts: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        pipe = self._load_pipe()
        max_sequence_length = self.config.actor_rollout_ref.rollout.pipeline.max_sequence_length
        guidance_scale = self.config.actor_rollout_ref.rollout.pipeline.get("guidance_scale", 1.0)
        do_cfg = guidance_scale is not None and guidance_scale > 1.0

        with torch.no_grad():
            encoded = pipe.encode_prompt(
                prompt=prompts,
                prompt_2=None,
                prompt_3=None,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_cfg,
                negative_prompt=negative_prompts if do_cfg else None,
                negative_prompt_2=None,
                negative_prompt_3=None,
                max_sequence_length=max_sequence_length,
            )

        if len(encoded) == 4:
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = encoded
        elif len(encoded) == 2:
            prompt_embeds, pooled_prompt_embeds = encoded
            negative_prompt_embeds = None
            negative_pooled_prompt_embeds = None
        else:
            raise ValueError(f"Unexpected SD3 encode_prompt output length: {len(encoded)}")

        prompt_embeds_mask = torch.ones(
            prompt_embeds.shape[0],
            prompt_embeds.shape[1],
            dtype=torch.int32,
            device=prompt_embeds.device,
        )
        negative_prompt_embeds_mask = None
        if negative_prompt_embeds is not None:
            negative_prompt_embeds_mask = torch.ones(
                negative_prompt_embeds.shape[0],
                negative_prompt_embeds.shape[1],
                dtype=torch.int32,
                device=negative_prompt_embeds.device,
            )

        return (
            prompt_embeds.detach().cpu(),
            prompt_embeds_mask.detach().cpu(),
            pooled_prompt_embeds.detach().cpu(),
            negative_prompt_embeds.detach().cpu() if negative_prompt_embeds is not None else None,
            negative_prompt_embeds_mask.detach().cpu() if negative_prompt_embeds_mask is not None else None,
            negative_pooled_prompt_embeds.detach().cpu() if negative_pooled_prompt_embeds is not None else None,
        )

    def materialize(self, batch: DataProto) -> DataProto:
        image_paths = batch.non_tensor_batch.get("image_path", None)
        if image_paths is None:
            raise KeyError("Offline DPO materialization requires `image_path` in non_tensor_batch.")

        prompts = [_as_text(item) for item in _to_list(batch.non_tensor_batch.get("raw_prompt", ""))]
        negative_prompts = [_as_text(item) for item in _to_list(batch.non_tensor_batch.get("raw_negative_prompt", ""))]
        if len(negative_prompts) == 1 and len(prompts) > 1:
            negative_prompts = negative_prompts * len(prompts)

        (
            prompt_embeds,
            prompt_embeds_mask,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            negative_pooled_prompt_embeds,
        ) = self._encode_prompts(prompts, negative_prompts)

        tensor_dict = {
            "image_latents": self._encode_images(_to_list(image_paths)),
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }
        if negative_prompt_embeds is not None:
            tensor_dict["negative_prompt_embeds"] = negative_prompt_embeds
            tensor_dict["negative_prompt_embeds_mask"] = negative_prompt_embeds_mask
            tensor_dict["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds

        return batch.union(DataProto.from_single_dict(tensor_dict))
