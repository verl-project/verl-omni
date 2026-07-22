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

"""Boogu-Image vLLM-Omni rollout adapter for FlowGRPO (T2I and Edit/TI2I)."""

import os
from typing import Literal

import torch
import torch.nn.functional as F
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.boogu_image.image_processor import BooguImageProcessor
from vllm_omni.diffusion.models.boogu_image.pipeline_boogu_image import BooguImagePipeline
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.pipelines.utils import ImageGenerationRequest

from .common import (
    apply_boogu_text_cfg,
    boogu_timestep_from_scheduler,
    configure_boogu_sde_timesteps,
    get_boogu_freqs_cis,
)

__all__ = ["BooguImagePipelineWithLogProb"]


def _coalesce_not_none(value, default):
    return default if value is None else value


@VllmOmniPipelineBase.register("BooguImagePipeline", algorithm="flow_grpo")
class BooguImagePipelineWithLogProb(BooguImagePipeline):
    """Rollout pipeline for Boogu-Image that captures per-step log-probabilities.

    Extends the vllm-omni ``BooguImagePipeline`` with:

    - the :class:`FlowMatchSDEDiscreteScheduler` (diffusers sigma convention;
      the Boogu 0->1 ascending timesteps map onto it via ``sigma = 1 - t`` and
      a **negated** velocity — see ``common.py``);
    - ``encode_prompt`` accepting pre-tokenised ``prompt_ids`` from the agent
      loop instead of raw text;
    - an SDE ``diffuse`` loop collecting ``all_latents`` / ``all_log_probs`` /
      ``all_timesteps`` within the SDE window.

    The Base (T2I) and Edit (TI2I) checkpoints share this architecture string;
    one adapter serves both. Edit requests carry a single reference image,
    which is VAE-encoded at near-native resolution (align_res: the output
    resolution follows the reference) and threaded through the transformer's
    ``ref_image_hidden_states`` refiner path; the VLM copy uses the
    checkpoint processor's own sizing so the image-placeholder count matches
    the pre-tokenised prompt.
    """

    supports_request_batch = False

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self.device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)

        num_layers = self.transformer.instruction_feature_configs.get("num_instruction_feature_layers", 1)
        if num_layers > 1:
            raise NotImplementedError(
                "Boogu-Image checkpoints with num_instruction_feature_layers > 1 return "
                "a list of per-layer prompt embeddings, which the FlowGRPO agent loop "
                "cannot ferry as a single padded tensor."
            )

        self.scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

    # ------------------------------------------------------------------
    # Prompt encoding from pre-tokenised IDs
    # ------------------------------------------------------------------

    def _get_boogu_prompt_embeds(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        condition_images: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode token IDs with the Qwen3VL ``mllm`` encoder.

        For Edit (TI2I) requests, ``condition_images`` are processed with the
        checkpoint processor's own image sizing — the same rule the agent
        loop's tokenization applied — so the image-placeholder token count in
        ``prompt_ids`` matches the pixel grid. (This intentionally skips
        upstream inference's 384^2 VLM downscale cap, which would desync the
        placeholder count from the pre-tokenised prompt.)
        """
        prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)
        attention_mask = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask

        encoder_kwargs = {}
        if condition_images:
            image_inputs = self.processor.image_processor(images=condition_images, return_tensors="pt")
            encoder_kwargs["pixel_values"] = image_inputs["pixel_values"].to(device=self.device, dtype=self.mllm.dtype)
            encoder_kwargs["image_grid_thw"] = image_inputs["image_grid_thw"].to(self.device)

        with torch.no_grad():
            outputs = self.mllm(
                input_ids=prompt_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                output_hidden_states=True,
                return_dict=True,
                **encoder_kwargs,
            )
        prompt_embeds = outputs.hidden_states[-1].to(dtype=self.mllm.dtype)
        return prompt_embeds, attention_mask.to(self.device)

    def encode_prompt(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1280,
        condition_images: list | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode pre-tokenised prompt IDs into padded ``(B, L, D)`` embeddings.

        Replaces the upstream text-based ``encode_prompt``; the agent loop ships
        token IDs produced by the data preprocessor, whose chat template must
        match the upstream Boogu system prompts exactly.
        """
        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_boogu_prompt_embeds(
                prompt_ids, attention_mask, condition_images=condition_images
            )

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            prompt_embeds_mask = prompt_embeds_mask.repeat_interleave(num_images_per_prompt, dim=0)

        return prompt_embeds, prompt_embeds_mask

    def _tokenize_text_prompt(self, text: str | list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize raw text with the upstream chat template (dummy-run fallback)."""
        prompts = [text] if isinstance(text, str) else list(text)
        messages = [self._apply_chat_template(item, None) for item in prompts]
        vlm_inputs = self.processor.apply_chat_template(
            messages,
            padding="longest",
            padding_side="right",
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        )
        return vlm_inputs["input_ids"], vlm_inputs["attention_mask"]

    def _extract_prompt_ids(self, prompts):
        """Extract positive/negative prompt IDs and masks from the request prompts."""
        prompt_ids = None
        prompt_mask = None
        negative_prompt_ids = None
        negative_prompt_mask = None
        if prompts:
            p0 = prompts[0]
            if isinstance(p0, dict):
                prompt_ids = p0.get("prompt_token_ids", None)
                prompt_mask = p0.get("prompt_mask", None)
                negative_prompt_ids = p0.get("negative_prompt_ids", None)
                negative_prompt_mask = p0.get("negative_prompt_mask", None)

                # Fallback: tokenize raw text prompt (covers the dummy-run path).
                if prompt_ids is None and p0.get("prompt"):
                    prompt_ids, prompt_mask = self._tokenize_text_prompt(p0["prompt"])
                if negative_prompt_ids is None and p0.get("negative_prompt"):
                    negative_prompt_ids, negative_prompt_mask = self._tokenize_text_prompt(p0["negative_prompt"])
            elif isinstance(p0, str):
                prompt_ids, prompt_mask = self._tokenize_text_prompt(p0)
        return prompt_ids, prompt_mask, negative_prompt_ids, negative_prompt_mask

    # ------------------------------------------------------------------
    # SDE denoising loop
    # ------------------------------------------------------------------

    def diffuse(
        self,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds_mask: torch.Tensor | None,
        latents: torch.Tensor,
        freqs_cis,
        timesteps: torch.Tensor,
        guidance_scale: float,
        noise_level: float,
        sde_window: tuple[int, int],
        sde_type: str,
        generator: torch.Generator | None,
        logprobs: bool,
        ref_image_hidden_states: list | None = None,
    ):
        """Run the SDE loop and collect per-step rollout data.

        The scheduler operates in the diffusers sigma convention while the
        transformer consumes Boogu time ``t = 1 - sigma`` and predicts the
        Boogu velocity, which is negated before every ``scheduler.step``.
        Latents stay fp32 in storage; casts to model dtype happen only for the
        transformer forward (see common_pitfalls: float32 trajectory rule).
        """
        all_latents: list[torch.Tensor] = []
        all_log_probs: list[torch.Tensor] = []
        all_timesteps: list[torch.Tensor] = []
        self.scheduler.set_begin_index(0)

        do_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None
        num_train_timesteps = self.scheduler.config.num_train_timesteps

        for i, timestep_value in enumerate(timesteps):
            if i < sde_window[0]:
                cur_noise_level = 0.0
            elif i == sde_window[0]:
                cur_noise_level = noise_level
                all_latents.append(latents.float())
            elif i < sde_window[1]:
                cur_noise_level = noise_level
            else:
                cur_noise_level = 0.0

            boogu_t = boogu_timestep_from_scheduler(timestep_value, num_train_timesteps)
            x = latents.to(prompt_embeds.dtype)

            noise_pred = self.predict(boogu_t, x, prompt_embeds, freqs_cis, prompt_embeds_mask, ref_image_hidden_states)
            if do_cfg:
                # Upstream text-only TI2I guidance keeps the reference latents
                # in the unconditional forward as well.
                negative_noise_pred = self.predict(
                    boogu_t, x, negative_prompt_embeds, freqs_cis, negative_prompt_embeds_mask, ref_image_hidden_states
                )
                noise_pred = apply_boogu_text_cfg(noise_pred, negative_noise_pred, guidance_scale)

            latents, log_prob, _, _ = self.scheduler.step(
                noise_pred.to(torch.float32).neg(),
                timestep_value,
                latents.to(torch.float32),
                generator=generator,
                noise_level=cur_noise_level,
                sde_type=sde_type,
                return_logprobs=logprobs,
                return_dict=False,
            )

            if sde_window[0] <= i < sde_window[1]:
                all_latents.append(latents.to(torch.float32))
                all_log_probs.append(log_prob)
                all_timesteps.append(timestep_value)

        stacked_latents = torch.stack(all_latents, dim=1)
        stacked_log_probs = (
            torch.stack(all_log_probs, dim=1) if all_log_probs and all_log_probs[0] is not None else None
        )
        stacked_timesteps = torch.stack(all_timesteps).unsqueeze(0).expand(latents.shape[0], -1)
        return latents, stacked_latents, stacked_log_probs, stacked_timesteps

    # ------------------------------------------------------------------
    # End-to-end rollout forward
    # ------------------------------------------------------------------

    def forward(
        self,
        req: DiffusionRequestBatch,
        noise_level: float = 0.7,
        sde_window_size: int | None = None,
        sde_window_range: tuple[int, int] = (0, 5),
        sde_type: Literal["sde", "cps", "dance_sde"] = "sde",
        logprobs: bool = True,
    ) -> DiffusionOutput:
        """End-to-end T2I / Edit (TI2I) generation with rollout-trajectory collection."""
        prompts = req.prompts

        # Edit (TI2I) requests carry a single reference image via the shared
        # multimodal request payload. Boogu editing supports one image.
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
        has_reference = bool(condition_images)

        prompt_ids, prompt_mask, negative_prompt_ids, negative_prompt_mask = self._extract_prompt_ids(prompts)

        if isinstance(prompt_ids, list):
            prompt_ids = torch.tensor(prompt_ids, device=self.device)
        if isinstance(negative_prompt_ids, list):
            negative_prompt_ids = torch.tensor(negative_prompt_ids, device=self.device)

        if prompt_ids is None:
            # Engine warm-up / dummy run without a usable prompt.
            return DiffusionOutput(output=None)

        sampling_params = req.sampling_params
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

        extra = sampling_params.extra_args or {}
        noise_level = _coalesce_not_none(extra.get("noise_level", None), noise_level)
        sde_window_size = _coalesce_not_none(extra.get("sde_window_size", None), sde_window_size)
        sde_window_range = _coalesce_not_none(extra.get("sde_window_range", None), sde_window_range)
        sde_type = _coalesce_not_none(extra.get("sde_type", None), sde_type)
        logprobs = _coalesce_not_none(extra.get("logprobs", None), logprobs)

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

        # Edit path: VAE-preprocess the reference at near-native resolution and
        # let the output resolution follow the reference dims (align_res).
        ref_image_hidden_states = None
        condition_image_latents = None
        if has_reference:
            image_processor = BooguImageProcessor(vae_scale_factor=self.vae_scale_factor * 2, do_resize=True)
            preprocessed_image = image_processor.preprocess(
                condition_images[0], max_pixels=2048 * 2048, max_side_length=2048 * 2
            )
            height = int(preprocessed_image.shape[-2])
            width = int(preprocessed_image.shape[-1])
            ref_image_hidden_states = self._build_ref_latents(
                [preprocessed_image],
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
            num_inference_steps=num_inference_steps,
            num_tokens=num_tokens,
            device=self.device,
        )
        timesteps = self.scheduler.timesteps

        if sde_window_size is not None:
            start = torch.randint(
                sde_window_range[0],
                sde_window_range[1] - sde_window_size + 1,
                (1,),
                generator=generator,
                device=self.device,
            ).item()
            sde_window = (start, start + sde_window_size)
        else:
            sde_window = (0, len(timesteps) - 1)

        freqs_cis = get_boogu_freqs_cis(self.transformer.axes_dim_rope, self.transformer.axes_lens)

        latents, all_latents, all_log_probs, all_timesteps = self.diffuse(
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            latents,
            freqs_cis,
            timesteps,
            guidance_scale,
            noise_level,
            sde_window,
            sde_type,
            generator,
            logprobs,
            ref_image_hidden_states=ref_image_hidden_states,
        )

        # Decode (upstream post-VAE normalisation and optional resize-back).
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

        metadata = {
            "prompt_embeddings": {
                "prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": prompt_embeds_mask,
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
            },
        }
        if condition_image_latents is not None:
            metadata["condition_image_latents"] = condition_image_latents

        return DiffusionOutput(
            output={
                "payload": {"image": image},
                "metadata": metadata,
            },
            trajectory_latents=all_latents,
            trajectory_log_probs=all_log_probs,
            trajectory_timesteps=all_timesteps,
            to_cpu=True,
        )
