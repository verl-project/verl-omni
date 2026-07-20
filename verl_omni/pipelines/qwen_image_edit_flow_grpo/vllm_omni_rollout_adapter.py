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

"""Qwen-Image-Edit-Plus rollout adapter with SDE log-prob collection."""

import os
from typing import Any, Literal

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit_plus import (
    VAE_IMAGE_SIZE,
    QwenImageEditPlusPipeline,
    calculate_dimensions,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.qwen_image_flow_grpo.common import (
    QwenImageTokenIdPromptMixin,
    apply_true_cfg,
    coalesce_not_none,
)
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.pipelines.utils import ImageGenerationRequest

__all__ = ["QwenImageEditPlusPipelineWithLogProb"]


def _maybe_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _use_true_cfg(
    true_cfg_scale: float,
    negative_prompt_ids,
    negative_prompt_embeds,
    negative_prompt_embeds_mask,
) -> bool:
    enabled = true_cfg_scale > 1
    has_negative_prompt = negative_prompt_ids is not None or (
        negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
    )
    if enabled and not has_negative_prompt:
        raise ValueError(
            "Qwen-Image-Edit true_cfg_scale > 1 requires negative_prompt_ids or negative prompt embeddings."
        )
    return enabled


def _validate_condition_image_sizes(condition_images, vae_image_sizes, target_size=None) -> None:
    if not condition_images or len(condition_images) != 1:
        count = 0 if not condition_images else len(condition_images)
        raise ValueError(f"Qwen-Image-Edit training requires exactly one condition image; got {count}")
    if condition_images and vae_image_sizes and len(vae_image_sizes) != len(condition_images):
        raise ValueError(
            f"got {len(condition_images)} condition images but {len(vae_image_sizes)} vae_image_sizes entries"
        )
    if not vae_image_sizes:
        raise ValueError("Qwen-Image-Edit requires non-empty additional_information['vae_image_sizes']")

    # Condition and target aspect ratios must match so batched latent lengths stay constant.
    if target_size is not None and all(target_size):
        target_height, target_width = target_size
        expected = calculate_dimensions(VAE_IMAGE_SIZE, target_width / target_height)
        if any((width, height) != expected for width, height in vae_image_sizes):
            raise ValueError(
                "Qwen-Image-Edit training requires every condition image to match the target "
                f"aspect ratio: expected VAE size {expected} for target {target_width}x{target_height}, "
                f"got {vae_image_sizes}. Letterbox or resize all source images to the target aspect "
                "ratio so condition latent lengths stay constant across a batch."
            )
        return

    if any(width != height for width, height in vae_image_sizes):
        raise ValueError(
            "Qwen-Image-Edit training requires square condition images when the pipeline target "
            "size is unspecified so condition latent lengths stay constant across a batch. Set the "
            "pipeline target height/width to your aspect ratio to train on non-square (e.g. 16:9, "
            "4:3) images, or pad source images to a square."
        )


@VllmOmniPipelineBase.register("QwenImageEditPlusPipeline", algorithm="flow_grpo")
class QwenImageEditPlusPipelineWithLogProb(QwenImageTokenIdPromptMixin, QwenImageEditPlusPipeline):
    """Qwen-Image-Edit-Plus rollout pipeline for FlowGRPO."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)

        # Replace the upstream scheduler with our SDE scheduler
        self.scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

    def _get_qwen_prompt_embeds(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        condition_images: list | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Encode prompt IDs and their condition-image features."""
        assert condition_images, "Qwen-Image-Edit prompt encoding requires condition images"
        dtype = dtype or self.text_encoder.dtype

        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)

        prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        attention_mask = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask
        attention_mask = attention_mask.to(self.device)
        drop_idx = self.prompt_template_encode_start_idx

        image_inputs = self.processor.image_processor(images=condition_images, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"].to(device=self.device, dtype=self.text_encoder.dtype)
        image_grid_thw = image_inputs["image_grid_thw"].to(self.device)

        encoder_hidden_states = self.text_encoder(
            input_ids=prompt_ids.to(self.device),
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype)

        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
        condition_images: list | None = None,
    ):
        """Encode text prompt token IDs into dense embeddings.

        Overrides the upstream :meth:`QwenImageEditPlusPipeline.encode_prompt`
        (which accepts raw text strings) to work with pre-tokenized prompt IDs
        as required by the verl-omni rollout loop.  ``condition_images`` is
        forwarded so the Qwen2.5-VL vision tower can replace
        ``<|image_pad|>`` placeholders with real image features instead of
        empty word embeddings.
        """
        if prompt_embeds is None:
            if prompt_ids is None:
                raise ValueError("prompt_ids is required when prompt_embeds is not provided")
            prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
            attention_mask = (
                attention_mask.unsqueeze(0)
                if attention_mask is not None and attention_mask.ndim == 1
                else attention_mask
            )
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                prompt_ids,
                attention_mask=attention_mask,
                condition_images=condition_images,
            )
        elif prompt_embeds_mask is None:
            prompt_embeds_mask = torch.ones(prompt_embeds.shape[:2], device=prompt_embeds.device, dtype=torch.long)

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            prompt_embeds_mask = prompt_embeds_mask.repeat_interleave(num_images_per_prompt, dim=0)

        return prompt_embeds, prompt_embeds_mask

    def diffuse(
        self,
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
        latents,
        condition_image_latents,
        img_shapes,
        txt_seq_lens,
        negative_txt_seq_lens,
        timesteps,
        do_true_cfg,
        true_cfg_scale,
        noise_level,
        sde_window,
        sde_type,
        generator,
        logprobs,
    ):
        """Run the full SDE diffusion loop for image editing with rollout data collection.

        Similar to :meth:`QwenImagePipelineWithLogProb.diffuse` but handles
        condition image latents concatenation and the edit-plus CFG variant.
        """
        all_latents = []
        all_log_probs = []
        all_timesteps = []
        self.scheduler.set_begin_index(0)
        self.transformer.do_true_cfg = do_true_cfg

        for i, timestep_value in enumerate(timesteps):
            if self.interrupt:
                continue

            if i < sde_window[0]:
                cur_noise_level = 0.0
            elif i == sde_window[0]:
                cur_noise_level = noise_level
                all_latents.append(latents.float())
            elif i > sde_window[0] and i < sde_window[1]:
                cur_noise_level = noise_level
            else:
                cur_noise_level = 0.0

            self._current_timestep = timestep_value
            timestep = timestep_value.expand(latents.shape[0]).to(device=latents.device, dtype=torch.float32)

            # Concatenate condition image latents and cast to model dtype for transformer forward.
            latent_model_input = torch.cat([latents, condition_image_latents], dim=1)
            latent_model_input = latent_model_input.to(self.transformer.img_in.weight.dtype)

            # Forward pass for positive prompt
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=None,  # QwenImageEditPlus doesn't use guidance embeds
                encoder_hidden_states_mask=prompt_embeds_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                attention_kwargs=self.attention_kwargs,
                return_dict=False,
            )[0]
            # Slice to target latent tokens only
            noise_pred = noise_pred[:, : latents.shape[1]]

            # CFG with negative prompt
            if do_true_cfg:
                neg_noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=negative_prompt_embeds_mask,
                    encoder_hidden_states=negative_prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=negative_txt_seq_lens,
                    attention_kwargs=self.attention_kwargs,
                    return_dict=False,
                )[0]
                neg_noise_pred = neg_noise_pred[:, : latents.shape[1]]
                noise_pred = apply_true_cfg(noise_pred, neg_noise_pred, true_cfg_scale)

            # Scheduler step
            latents, log_prob, _, _ = self.scheduler.step(
                noise_pred.float(),
                timestep_value,
                latents,
                generator=generator,
                noise_level=cur_noise_level,
                sde_type=sde_type,
                return_logprobs=logprobs,
                return_dict=False,
            )

            if i >= sde_window[0] and i < sde_window[1]:
                all_latents.append(latents.to(torch.float32))
                all_log_probs.append(log_prob)
                all_timesteps.append(timestep_value)

        all_latents = torch.stack(all_latents, dim=1)
        all_log_probs = torch.stack(all_log_probs, dim=1) if all_log_probs and all_log_probs[0] is not None else None
        all_timesteps = torch.stack(all_timesteps).unsqueeze(0).expand(latents.shape[0], -1)
        return latents, all_latents, all_log_probs, all_timesteps

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt_ids: torch.Tensor | list[int] | None = None,
        prompt_mask: torch.Tensor | None = None,
        negative_prompt_ids: torch.Tensor | list[int] | None = None,
        negative_prompt_mask: torch.Tensor | None = None,
        true_cfg_scale: float = 4.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int = 512,
        noise_level: float = 0.7,
        sde_window_size: int | None = None,
        sde_window_range: tuple[int, int] = (0, 5),
        sde_type: Literal["sde", "cps"] = "sde",
        logprobs: bool = True,
    ) -> DiffusionOutput:
        """End-to-end image editing with rollout data collection.

        Encodes the prompt (with condition images), prepares latents (with
        condition image latents), runs the SDE diffusion loop, and decodes the
        final output.
        """
        custom_prompt = req.prompts[0] if req.prompts else {}

        # Parse the condition images via the shared ImageGenerationRequest interface.
        # NOTE: only this image-edit pipeline consumes ImageGenerationRequest for now;
        # migrating the existing T2I pipelines onto it is left to a follow-up PR to keep
        # this change focused.
        request_payload = custom_prompt
        if (
            isinstance(custom_prompt, dict)
            and prompt_embeds is not None
            and custom_prompt.get("prompt") is None
            and custom_prompt.get("prompt_token_ids") is None
        ):
            request_payload = {**custom_prompt, "prompt": ""}
        gen_request = ImageGenerationRequest.from_request_payload(request_payload) if request_payload else None
        condition_images = gen_request.images if gen_request else None
        if not condition_images:
            raise ValueError("Qwen-Image-Edit requires at least one condition image")

        if isinstance(custom_prompt, dict):
            prompt_ids = custom_prompt.get("prompt_token_ids", prompt_ids)
            prompt_mask = custom_prompt.get("prompt_mask", prompt_mask)
            negative_prompt_ids = custom_prompt.get("negative_prompt_ids", negative_prompt_ids)
            negative_prompt_mask = custom_prompt.get("negative_prompt_mask", negative_prompt_mask)
            additional_information = custom_prompt.get("additional_information", {})
            vae_images = additional_information.get("vae_images")
            vae_image_sizes = additional_information.get("vae_image_sizes")
        else:
            vae_images = None
            vae_image_sizes = None

        sampling_params = req.sampling_params
        height = sampling_params.height or self.default_sample_size * self.vae_scale_factor
        width = sampling_params.width or self.default_sample_size * self.vae_scale_factor
        _validate_condition_image_sizes(condition_images, vae_image_sizes, target_size=(height, width))
        num_inference_steps = sampling_params.num_inference_steps or num_inference_steps
        sigmas = sampling_params.sigmas or sigmas
        max_sequence_length = sampling_params.max_sequence_length or max_sequence_length
        if sampling_params.guidance_scale_provided:
            guidance_scale = sampling_params.guidance_scale
        if sampling_params.num_outputs_per_prompt > 0:
            num_images_per_prompt = sampling_params.num_outputs_per_prompt

        noise_level = coalesce_not_none(sampling_params.extra_args.get("noise_level", None), noise_level)
        sde_window_size = coalesce_not_none(sampling_params.extra_args.get("sde_window_size", None), sde_window_size)
        sde_window_range = coalesce_not_none(sampling_params.extra_args.get("sde_window_range", None), sde_window_range)
        sde_type = coalesce_not_none(sampling_params.extra_args.get("sde_type", None), sde_type)
        logprobs = coalesce_not_none(sampling_params.extra_args.get("logprobs", None), logprobs)

        generator = sampling_params.generator or generator
        if generator is None and sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling_params.seed)
        true_cfg_scale = coalesce_not_none(sampling_params.true_cfg_scale, true_cfg_scale)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt_ids is not None:
            if isinstance(prompt_ids, list):
                prompt_ids = torch.tensor(prompt_ids, device=self.device)
            batch_size = prompt_ids.shape[0] if prompt_ids.ndim == 2 else 1
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            return DiffusionOutput(output=None, custom_output={})

        if isinstance(negative_prompt_ids, list):
            negative_prompt_ids = torch.tensor(negative_prompt_ids, device=self.device)

        do_true_cfg = _use_true_cfg(
            true_cfg_scale,
            negative_prompt_ids,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
        )

        # Encode prompts
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt_ids=prompt_ids,
            attention_mask=prompt_mask,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            condition_images=condition_images,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt_ids=negative_prompt_ids,
                attention_mask=negative_prompt_mask,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                condition_images=condition_images,
            )

        # Prepare target latents
        num_channels_latents = self.transformer.in_channels // 4
        latents, condition_image_latents = self.prepare_latents(
            vae_images,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
        )

        if condition_image_latents is None:
            raise ValueError("Qwen-Image-Edit requires preprocessed condition images")
        condition_image_latents = condition_image_latents.to(device=self.device, dtype=prompt_embeds.dtype)

        # Build img_shapes (includes both target and condition image shapes).
        target_shape = (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)
        condition_shapes = [
            (1, vae_height // self.vae_scale_factor // 2, vae_width // self.vae_scale_factor // 2)
            for vae_width, vae_height in vae_image_sizes
        ]
        img_shapes = [[target_shape, *condition_shapes] for _ in range(batch_size)]

        # Prepare timesteps
        timesteps, num_inference_steps = self.prepare_timesteps(num_inference_steps, sigmas, latents.shape[1])
        self._num_timesteps = len(timesteps)

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )

        if sde_window_size is not None:
            window_generator = generator[0] if isinstance(generator, list) else generator
            start = torch.randint(
                sde_window_range[0],
                sde_window_range[1] - sde_window_size + 1,
                (1,),
                generator=window_generator,
                device=self.device,
            ).item()
            end = start + sde_window_size
            sde_window = (start, end)
        else:
            sde_window = (0, len(timesteps) - 1)

        latents, all_latents, all_log_probs, all_timesteps = self.diffuse(
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            latents,
            condition_image_latents,
            img_shapes,
            txt_seq_lens,
            negative_txt_seq_lens,
            timesteps,
            do_true_cfg,
            true_cfg_scale,
            noise_level,
            sde_window,
            sde_type,
            generator,
            logprobs,
        )

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]

        return DiffusionOutput(
            output=_maybe_to_cpu(image),
            custom_output={
                "all_latents": _maybe_to_cpu(all_latents),
                "all_log_probs": _maybe_to_cpu(all_log_probs),
                "all_timesteps": _maybe_to_cpu(all_timesteps),
                "prompt_embeds": _maybe_to_cpu(prompt_embeds),
                "prompt_embeds_mask": _maybe_to_cpu(prompt_embeds_mask),
                "negative_prompt_embeds": _maybe_to_cpu(negative_prompt_embeds),
                "negative_prompt_embeds_mask": _maybe_to_cpu(negative_prompt_embeds_mask),
                "condition_image_latents": _maybe_to_cpu(condition_image_latents),
                "img_shapes": img_shapes,
            },
        )
