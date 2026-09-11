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

"""Boogu-Image rollout adapter for DiffusionNFT (T2I and Edit)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.boogu_image_flow_grpo.common import (
    apply_boogu_text_cfg,
    boogu_timestep_from_scheduler,
    configure_boogu_sde_timesteps,
    get_boogu_freqs_cis,
)
from verl_omni.pipelines.boogu_image_flow_grpo.vllm_omni_rollout_adapter import BooguImagePipelineWithLogProb
from verl_omni.pipelines.diffusion_rollout_output import rollout_output
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.request_batch import split_diffusion_output_by_request as _split_diffusion_output_by_request
from verl_omni.pipelines.utils import ImageGenerationRequest

__all__ = ["BooguImageDiffusionNFTPipeline"]


@VllmOmniPipelineBase.register("BooguImagePipeline", algorithm="diffusion_nft")
class BooguImageDiffusionNFTPipeline(BooguImagePipelineWithLogProb):
    """Generate clean latents with deterministic denoising for DiffusionNFT."""

    supports_request_batch = True

    def forward(
        self,
        req: OmniDiffusionRequest | DiffusionRequestBatch,
    ) -> DiffusionOutput | list[DiffusionOutput]:
        """Generate T2I / Edit images and clean latents for DiffusionNFT."""
        request_batch = req if isinstance(req, DiffusionRequestBatch) else DiffusionRequestBatch(requests=[req])
        return_batch = isinstance(req, DiffusionRequestBatch)
        prompts = request_batch.prompts

        # Parent preprocessing supplies VAE tensors; Qwen3VL still needs the raw
        # image whose pixel grid matches the pre-tokenised placeholders.
        _, preprocessed_images = self._extract_reference_images(prompts)
        has_reference = any(image is not None for image in preprocessed_images)

        # Edit (TI2I) sizes its output from each request's own reference image,
        # so packed Edit requests cannot share one latent grid. Serve them one
        # at a time; only T2I takes the packed path below.
        if has_reference and request_batch.num_reqs > 1:
            return [self.forward(request) for request in request_batch.requests]

        # Only the Edit path reads images off the prompt, and it is unpacked by
        # the branch above, so the first prompt is the only one that carries them.
        custom_prompt = prompts[0] if prompts else {}
        condition_images: list = []
        if isinstance(custom_prompt, dict):
            generation_request = ImageGenerationRequest.from_request_payload(custom_prompt)
            condition_images = list(generation_request.images or [])
        if len(condition_images) > 1:
            raise ValueError(
                f"Boogu-Image editing supports a single reference image; received {len(condition_images)}."
            )
        condition_images = [image.convert("RGB") for image in condition_images]
        if has_reference != bool(condition_images):
            raise ValueError("Boogu-Image Edit requires both raw and parent-preprocessed reference images.")

        prompt_ids, prompt_mask, negative_prompt_ids, negative_prompt_mask = self._collate_prompt_batch(prompts)

        if isinstance(prompt_ids, list):
            prompt_ids = torch.tensor(prompt_ids, device=self.device)
        if isinstance(negative_prompt_ids, list):
            negative_prompt_ids = torch.tensor(negative_prompt_ids, device=self.device)

        if prompt_ids is None:
            # Engine warm-up / dummy run without a usable prompt.
            outputs = [DiffusionOutput(output=None) for _ in range(request_batch.num_reqs)]
            return outputs if return_batch else outputs[0]

        sampling_params = request_batch.sampling_params_list[0]
        height = sampling_params.height or self.default_sample_size * self.vae_scale_factor
        width = sampling_params.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = sampling_params.num_inference_steps or 50
        max_sequence_length = sampling_params.max_sequence_length or 1280
        # Upstream default text guidance is 4.0; the engine coerces an unset
        # guidance_scale to 1.0, so only honor a caller-provided value.
        guidance_scale = sampling_params.guidance_scale if sampling_params.guidance_scale_provided else 4.0
        num_images_per_prompt = (
            sampling_params.num_outputs_per_prompt if sampling_params.num_outputs_per_prompt > 0 else 1
        )

        if request_batch.num_reqs > 1:
            # Preserve each packed request's seeded initial latents.
            for request in request_batch.requests:
                request_params = request.sampling_params
                if request_params.generator is None and request_params.seed is not None:
                    request_params.generator = torch.Generator(device=self.device).manual_seed(request_params.seed)
            generator = request_batch.collate_request_generators(num_images_per_prompt, None)
        else:
            generator = sampling_params.generator
            if generator is None and sampling_params.seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(sampling_params.seed)

        batch_size = prompt_ids.shape[0] if prompt_ids.ndim == 2 else 1
        if has_reference and batch_size != 1:
            raise ValueError(
                "Boogu-Image Edit rollouts support one prompt per request "
                f"(a single reference image); got a prompt batch of {batch_size}."
            )

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt_ids=prompt_ids,
            attention_mask=prompt_mask,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            condition_images=condition_images or None,
        )
        do_cfg = guidance_scale > 1.0 and negative_prompt_ids is not None
        if do_cfg:
            # Upstream default use_input_images_4_neg_instruct=False: the
            # negative instruction is encoded text-only.
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt_ids=negative_prompt_ids,
                attention_mask=negative_prompt_mask,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None

        # Edit path: reuse the parent's near-native VAE preprocessing and let
        # the output resolution follow the reference dims (align_res).
        ref_image_hidden_states = None
        condition_image_latents = None
        if has_reference:
            ref_image_hidden_states = self._build_ref_latents(
                preprocessed_images,
                num_images_per_prompt,
                self.device,
                generator,
            )
            # Transport shape (B, C, H, W): one reference latent per output.
            condition_image_latents = torch.stack([sample_latents[0] for sample_latents in ref_image_hidden_states])

        # Working resolution (upstream clamps to 2048^2, multiples of vsf*2).
        height, width, ori_height, ori_width = self._resolve_output_size(height, width)

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.transformer.in_channels,
            height,
            width,
            torch.float32,
            self.device,
            generator,
        )

        num_tokens = latents.shape[-2] * latents.shape[-1]
        configure_boogu_sde_timesteps(
            self.scheduler,
            native_scheduler=self._boogu_scheduler,
            num_inference_steps=num_inference_steps,
            num_tokens=num_tokens,
            device=self.device,
        )
        timesteps = self.scheduler.timesteps

        freqs_cis = get_boogu_freqs_cis(self.transformer.axes_dim_rope, self.transformer.axes_lens)
        self.scheduler.set_begin_index(0)
        num_train_timesteps = self.scheduler.config.num_train_timesteps

        for timestep_value in timesteps:
            boogu_t = boogu_timestep_from_scheduler(timestep_value, num_train_timesteps)
            x = latents.to(prompt_embeds.dtype)
            noise_pred = self.predict(boogu_t, x, prompt_embeds, freqs_cis, prompt_embeds_mask, ref_image_hidden_states)
            if do_cfg:
                negative_noise_pred = self.predict(
                    boogu_t, x, negative_prompt_embeds, freqs_cis, negative_prompt_embeds_mask, ref_image_hidden_states
                )
                noise_pred = apply_boogu_text_cfg(noise_pred, negative_noise_pred, guidance_scale)

            latents, _, _, _ = self.scheduler.step(
                noise_pred.to(torch.float32).neg(),
                timestep_value,
                latents.to(torch.float32),
                generator=generator,
                noise_level=0.0,
                sde_type="sde",
                return_logprobs=False,
                return_dict=False,
            )

        # Decode the way upstream does: undo the VAE scaling/shift, resize back.
        output_type = sampling_params.output_type or "pil"
        if output_type == "latent":
            image = latents
        else:
            decode_latents = latents.to(dtype=self.vae.dtype)
            if self.vae.config.scaling_factor is not None:
                decode_latents = decode_latents / self.vae.config.scaling_factor
            if self.vae.config.shift_factor is not None:
                decode_latents = decode_latents + self.vae.config.shift_factor
            image = self.vae.decode(decode_latents, return_dict=False)[0]
            if (ori_height, ori_width) != (height, width):
                image = F.interpolate(image, size=(ori_height, ori_width), mode="bilinear")

        result = rollout_output(
            media=image,
            prompt_embeddings={
                "prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": prompt_embeds_mask,
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
            },
            rl={
                "latents_clean": latents.float(),
                "train_timesteps": timesteps.unsqueeze(0).expand(latents.shape[0], -1),
                **({"condition_image_latents": condition_image_latents} if condition_image_latents is not None else {}),
            },
            to_cpu=True,
        )
        outputs = _split_diffusion_output_by_request(
            result,
            request_batch,
            num_outputs_per_prompt=num_images_per_prompt,
        )
        return outputs if return_batch else outputs[0]
