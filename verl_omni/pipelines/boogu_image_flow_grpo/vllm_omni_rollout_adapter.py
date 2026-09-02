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

from __future__ import annotations

import os
from typing import Any, Literal

import torch
import torch.nn.functional as F
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.boogu_image import pipeline_boogu_image
from vllm_omni.diffusion.models.boogu_image.pipeline_boogu_image import BooguImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.diffusion_rollout_output import rollout_output, wrap_rollout_postprocessor
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.qwen_image_flow_grpo.common import QwenImageTokenIdPromptMixin, coalesce_not_none
from verl_omni.pipelines.request_batch import (
    collate_prompt_mask as _collate_prompt_mask,
)
from verl_omni.pipelines.request_batch import (
    collate_prompt_rows as _collate_prompt_rows,
)
from verl_omni.pipelines.request_batch import (
    sample_per_sample_sde_windows as _sample_per_sample_sde_windows,
)
from verl_omni.pipelines.request_batch import (
    split_diffusion_output_by_request as _split_diffusion_output_by_request,
)
from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.pipelines.utils import ImageGenerationRequest

from .common import (
    apply_boogu_text_cfg,
    boogu_timestep_from_scheduler,
    configure_boogu_sde_timesteps,
    get_boogu_freqs_cis,
)

__all__ = ["BooguImagePipelineWithLogProb"]

_BOOGU_POST_PROCESS_FACTORY = pipeline_boogu_image.get_boogu_image_post_process_func


def get_rollout_post_process_func(od_config):
    """Postprocess Boogu-Image media while preserving rollout metadata."""
    return wrap_rollout_postprocessor(_BOOGU_POST_PROCESS_FACTORY(od_config))


# vllm-omni resolves the built-in architecture's factory in the engine process.
pipeline_boogu_image.get_boogu_image_post_process_func = get_rollout_post_process_func


@VllmOmniPipelineBase.register("BooguImagePipeline", algorithm="flow_grpo")
class BooguImagePipelineWithLogProb(QwenImageTokenIdPromptMixin, BooguImagePipeline):
    """Rollout pipeline for Boogu-Image that captures per-step log-probabilities.

    Extends the vllm-omni ``BooguImagePipeline`` with:

    - the :class:`FlowMatchSDEDiscreteScheduler` (diffusers sigma convention;
      see ``common.py`` for the sigma/velocity mapping);
    - ``encode_prompt`` accepting pre-tokenised ``prompt_ids`` from the agent
      loop instead of raw text;
    - an SDE ``diffuse`` loop collecting ``all_latents`` / ``all_log_probs`` /
      ``all_timesteps`` within the SDE window.

    Base (T2I) and Edit (TI2I) checkpoints share this architecture string, so
    one adapter serves both; Edit requests simply carry a reference image.

    T2I requests are served packed: without that the engine clamps
    ``max_num_seqs`` to 1 and every rollout image costs a separate forward.
    Edit falls back to one request at a time because its output size follows
    each request's own reference image.
    """

    supports_request_batch = True

    #: Declares the primary rollout media stream so downstream consumers read
    #: the modality from the adapter instead of inferring it from tensor rank.
    diffusion_io_spec = DiffusionIOSpec(primary=MediaSpec("image"))

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self._boogu_scheduler = self.scheduler
        self.device = self._execution_device
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

        # Skip the vision tower unless the prompt carries image placeholder
        # tokens: the engine warm-up injects a synthetic image without any,
        # which would crash on unmatched features ("tokens: 0, features: N").
        encoder_kwargs = {}
        image_token_id = getattr(self.mllm.config, "image_token_id", None)
        has_image_tokens = image_token_id is not None and bool((prompt_ids == image_token_id).any())
        if condition_images and has_image_tokens:
            image_inputs = self.processor.image_processor(images=condition_images, return_tensors="pt")
            encoder_kwargs["pixel_values"] = image_inputs["pixel_values"].to(device=self.device, dtype=self.mllm.dtype)
            encoder_kwargs["image_grid_thw"] = image_inputs["image_grid_thw"].to(self.device)
            # transformers >= 5.x Qwen3VL wants per-token modality ids for
            # M-RoPE (0 = text, 1 = image); rebuild them from the pre-tokenised
            # prompt, since we bypass the processor call that returns them.
            encoder_kwargs["mm_token_type_ids"] = (prompt_ids == image_token_id).long().to(self.device)

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
        return super().encode_prompt(
            prompt_ids,
            attention_mask,
            num_images_per_prompt,
            prompt_embeds,
            prompt_embeds_mask,
            max_sequence_length,
        )

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

    def _collate_prompt_batch(self, prompts):
        """Pack every request's pre-tokenised prompt into one padded ``(N, L)`` grid.

        Falls back to :meth:`_extract_prompt_ids` for raw-text prompts, which
        only the engine warm-up sends and which never arrive packed.
        """
        prompt_ids, token_lengths = _collate_prompt_rows(
            prompts,
            ("prompt_token_ids", "prompt_ids"),
            None,
            device=self.device,
            field_name="prompt_token_ids",
        )
        if prompt_ids is None:
            return self._extract_prompt_ids(prompts)

        prompt_mask = _collate_prompt_mask(
            prompts,
            ("prompt_mask",),
            None,
            device=self.device,
            field_name="prompt_mask",
            token_lengths=token_lengths,
            target_seq_len=int(prompt_ids.shape[1]),
        )
        negative_prompt_ids, negative_token_lengths = _collate_prompt_rows(
            prompts,
            ("negative_prompt_ids",),
            None,
            device=self.device,
            field_name="negative_prompt_ids",
        )
        negative_prompt_mask = _collate_prompt_mask(
            prompts,
            ("negative_prompt_mask",),
            None,
            device=self.device,
            field_name="negative_prompt_mask",
            token_lengths=negative_token_lengths,
            target_seq_len=None if negative_prompt_ids is None else int(negative_prompt_ids.shape[1]),
        )
        # Both the Qwen3VL encoder and the Boogu transformer were fed a long
        # mask before packing; the collate helper hands back bool.
        if prompt_mask is not None:
            prompt_mask = prompt_mask.long()
        if negative_prompt_mask is not None:
            negative_prompt_mask = negative_prompt_mask.long()
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
        sde_window: tuple[int, int] | list[tuple[int, int]],
        sde_type: str,
        generator: torch.Generator | list[torch.Generator] | None,
        logprobs: bool,
        ref_image_hidden_states: list | None = None,
    ):
        """Run the SDE loop and collect per-step rollout data.

        The scheduler operates in the diffusers sigma convention while the
        transformer consumes Boogu time ``t = 1 - sigma`` and predicts the
        Boogu velocity, which is negated before every ``scheduler.step``.
        Latents stay fp32 in storage; casts to model dtype happen only for the
        transformer forward (see common_pitfalls: float32 trajectory rule).

        ``sde_window`` is either one window shared by the batch or one per
        packed row. Per-row windows are what let packed requests keep the
        trajectory each would have drawn alone: a row outside its own window
        sees ``noise_level`` 0 while its neighbours are still diffusing.
        """
        batch_size = latents.shape[0]
        windows = [sde_window] * batch_size if isinstance(sde_window, tuple) else list(sde_window)
        if len(windows) != batch_size:
            raise ValueError(f"Expected {batch_size} SDE windows, got {len(windows)}.")
        if len({end - start for start, end in windows}) != 1:
            raise ValueError("Packed SDE windows must share the same size.")

        all_latents: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        all_log_probs: list[list[Any]] = [[] for _ in range(batch_size)]
        all_timesteps: list[list[Any]] = [[] for _ in range(batch_size)]
        self.scheduler.set_begin_index(0)

        do_cfg = guidance_scale > 1.0 and negative_prompt_embeds is not None
        num_train_timesteps = self.scheduler.config.num_train_timesteps

        for i, timestep_value in enumerate(timesteps):
            for batch_idx, (start, _end) in enumerate(windows):
                if i == start:
                    all_latents[batch_idx].append(latents[batch_idx].detach().float().clone())

            levels = [float(noise_level) if start <= i < end else 0.0 for start, end in windows]
            cur_noise_level: float | torch.Tensor = (
                levels[0]
                if all(level == levels[0] for level in levels)
                else torch.tensor(levels, device=latents.device, dtype=torch.float32).view(
                    batch_size, *([1] * (latents.ndim - 1))
                )
            )

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

            for batch_idx, (start, end) in enumerate(windows):
                if start <= i < end:
                    all_latents[batch_idx].append(latents[batch_idx].detach().to(torch.float32).clone())
                    all_log_probs[batch_idx].append(None if log_prob is None else log_prob[batch_idx])
                    all_timesteps[batch_idx].append(timestep_value)

        if not any(all_timesteps):
            # Degenerate window (e.g. the engine's single-step warm-up run):
            # nothing was collected, and the caller ships None-safe fields.
            return latents, None, None, None

        stacked_latents = torch.stack([torch.stack(row, dim=0) for row in all_latents], dim=0)
        stacked_log_probs = (
            torch.stack([torch.stack(row, dim=0) for row in all_log_probs], dim=0)
            if all_log_probs[0] and all_log_probs[0][0] is not None
            else None
        )
        stacked_timesteps = torch.stack([torch.stack(row, dim=0) for row in all_timesteps], dim=0)
        return latents, stacked_latents, stacked_log_probs, stacked_timesteps

    # ------------------------------------------------------------------
    # End-to-end rollout forward
    # ------------------------------------------------------------------

    def forward(
        self,
        req: OmniDiffusionRequest | DiffusionRequestBatch,
        noise_level: float = 0.7,
        sde_window_size: int | None = None,
        sde_window_range: tuple[int, int] = (0, 5),
        sde_type: Literal["sde", "cps", "dance_sde"] = "sde",
        logprobs: bool = True,
    ) -> DiffusionOutput | list[DiffusionOutput]:
        """End-to-end T2I / Edit (TI2I) generation with rollout-trajectory collection."""
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
            return [
                self.forward(
                    request,
                    noise_level=noise_level,
                    sde_window_size=sde_window_size,
                    sde_window_range=sde_window_range,
                    sde_type=sde_type,
                    logprobs=logprobs,
                )
                for request in request_batch.requests
            ]

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

        extra = sampling_params.extra_args or {}
        noise_level = coalesce_not_none(extra.get("noise_level", None), noise_level)
        sde_window_size = coalesce_not_none(extra.get("sde_window_size", None), sde_window_size)
        sde_window_range = coalesce_not_none(extra.get("sde_window_range", None), sde_window_range)
        sde_type = coalesce_not_none(extra.get("sde_type", None), sde_type)
        logprobs = coalesce_not_none(extra.get("logprobs", None), logprobs)

        if request_batch.num_reqs > 1:
            # One RNG per packed request, so a row draws the same latents and
            # SDE window it would have drawn running alone.
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

        sde_window = _sample_per_sample_sde_windows(
            sde_window_size=sde_window_size,
            sde_window_range=sde_window_range if sde_window_range is not None else (0, 5),
            num_timesteps=len(timesteps),
            batch_size=latents.shape[0],
            generator=generator,
            device=self.device,
        )

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
            trajectory_latents=all_latents,
            trajectory_log_probs=all_log_probs,
            trajectory_timesteps=all_timesteps,
            prompt_embeddings={
                "prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": prompt_embeds_mask,
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
            },
            # T2I rollouts carry no reference image; omit the group so the
            # training side's ``condition_image_latents`` stays absent there.
            rl=None if condition_image_latents is None else {"condition_image_latents": condition_image_latents},
            to_cpu=True,
        )
        outputs = _split_diffusion_output_by_request(
            result,
            request_batch,
            num_outputs_per_prompt=num_images_per_prompt,
        )
        return outputs if return_batch else outputs[0]
