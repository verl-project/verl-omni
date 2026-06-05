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

"""
Qwen-Image-Edit-Plus rollout adapter for vllm-omni with SDE log-prob collection.

Extends the upstream QwenImageEditPlusPipeline to:
1. Replace the scheduler with FlowMatchSDEDiscreteScheduler.
2. Collect per-step latents, log-probs, and timesteps during the SDE window.
3. Return prompt embeddings in DiffusionOutput.custom_output for the agent loop.
"""

import os
from typing import Any, Literal

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit_plus import QwenImageEditPlusPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

__all__ = ["QwenImageEditPlusPipelineWithLogProb"]


def _maybe_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _coalesce_not_none(value, default):
    return default if value is None else value


def _pick_condition_images(custom_prompt: dict) -> list | None:
    """Choose the condition image list whose patch-grid count matches the
    ``<|image_pad|>`` placeholder count baked into the prompt tokens.

    The agent loop calls ``processor(text=..., images=raw_images, ...)`` on
    the **raw** PIL list to expand each ``<image>`` placeholder into a run
    of ``<|vision_start|><|image_pad|>...<|vision_end|>`` tokens whose count
    is governed by ``image_grid_thw`` from THAT call. Any condition image
    list that produces a different grid (e.g. the resized 384x384 list that
    vllm-omni's ``pre_process_func`` writes to
    ``additional_information["condition_images"]``) will surface as::

        ValueError: Image features and image tokens do not match,
            tokens: <N>, features: <M>

    Resolution order (most-likely-correct first):

    1. ``custom_prompt["multi_modal_data"]["image"]`` — the raw PIL list the
       agent loop produced. ``pre_process_func`` reads it but does not
       modify it, so the tokens were derived from this exact list.
    2. ``additional_information["condition_images"]`` — populated by
       upstream ``pre_process_func`` after resize. Used only as a last-
       resort fallback when the rollout request was constructed without
       ``multi_modal_data`` (e.g. e2e smoke tests).
    3. ``None`` — text-only path; the encoder gets no image features.

    Centralising the selection makes it unit-testable without standing up
    the full pipeline.
    """
    if not isinstance(custom_prompt, dict):
        return None
    multi_modal_data = custom_prompt.get("multi_modal_data") or {}
    raw_images = multi_modal_data.get("image")
    if isinstance(raw_images, list) and len(raw_images) > 0:
        return raw_images
    additional_information = custom_prompt.get("additional_information") or {}
    return additional_information.get("condition_images")


@VllmOmniPipelineBase.register("QwenImageEditPlusPipeline", algorithm="flow_grpo")
class QwenImageEditPlusPipelineWithLogProb(QwenImageEditPlusPipeline):
    """Rollout pipeline for Qwen-Image-Edit-Plus that captures per-step log-probabilities.

    Extends the base QwenImageEditPlusPipeline with a custom SDE-based scheduler
    and additional output fields for RL training (FlowGRPO).
    """

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
        dtype = dtype or self.text_encoder.dtype

        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)

        prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        attention_mask = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask
        drop_idx = self.prompt_template_encode_start_idx

        # The agent loop expanded every ``<image>`` placeholder in the prompt
        # text into a long run of ``<|vision_start|><|image_pad|>...<|vision_end|>``
        # tokens by calling ``processor(text=..., images=raw_images, ...)`` on
        # the same raw PIL list. The placeholder count is governed by
        # ``image_grid_thw`` from THAT call.
        #
        # Without ``pixel_values`` / ``image_grid_thw``, the Qwen2.5-VL
        # text_encoder treats every ``<|image_pad|>`` as an empty word
        # embedding, prompt features become garbage, and the diffusion
        # transformer denoises noise into noise (visible as garbled rollout
        # images). With them, the encoder fuses real image features in.
        #
        # IMPORTANT: ``condition_images`` here MUST be the raw PIL images
        # forwarded by the agent loop (i.e. ``multi_modal_data["image"]`` in
        # ``forward()``), NOT the resized list that vllm-omni's
        # ``pre_process_func`` writes to ``additional_information["condition_images"]``.
        # The latter is already resized to CONDITION_IMAGE_SIZE (384x384) and
        # would produce a 14x14 (=196) patch grid — mismatched against the
        # placeholder count derived from the raw image, surfacing as::
        #
        #   ValueError: Image features and image tokens do not match,
        #     tokens: <N>, features: <M>
        pixel_values = None
        image_grid_thw = None
        if condition_images is not None and len(condition_images) > 0:
            image_inputs = self.processor.image_processor(images=condition_images, return_tensors="pt")
            pixel_values = image_inputs["pixel_values"].to(device=self.device, dtype=self.text_encoder.dtype)
            image_grid_thw = image_inputs["image_grid_thw"].to(self.device)

        encoder_hidden_states = self.text_encoder(
            input_ids=prompt_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
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
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 1024,
        condition_images: list | None = None,
    ):
        """Encode text prompt token IDs into dense embeddings.

        Overrides the upstream QwenImageEditPlusPipeline.encode_prompt (which
        accepts raw text strings) to work with pre-tokenized prompt IDs as
        required by the verl-omni rollout loop. ``condition_images`` is forwarded
        so the Qwen2.5-VL vision tower can replace ``<|image_pad|>`` placeholders
        with real image features instead of empty word embeddings.
        """
        prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        attention_mask = (
            attention_mask.unsqueeze(0) if attention_mask is not None and attention_mask.ndim == 1 else attention_mask
        )

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                prompt_ids,
                attention_mask=attention_mask,
                condition_images=condition_images,
            )

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
        image_latents,
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

        Similar to QwenImagePipelineWithLogProb.diffuse but handles condition
        image latents concatenation.
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
            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1)
            else:
                latent_model_input = latents
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

                # Rescaled CFG (norm-preserving) specific to QwenImageEditPlus.
                # Clamp the denominator: when comb_pred collapses to near-zero
                # under bf16 + few-inference-steps the unclamped form produces
                # inf/NaN that downstream get cached as the rollout's
                # ``old_log_prob`` and silently NaN out the FSDP grad.
                comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True).clamp_min(1e-6)
                noise_pred = comb_pred * (cond_norm / noise_norm)

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
                all_latents.append(latents)
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
        image_latents: torch.Tensor | None = None,
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

        Encodes the prompt, prepares latents (with condition image latents),
        runs the SDE diffusion loop, and decodes the final output.
        """
        custom_prompt = req.prompts[0] if req.prompts else {}
        condition_images = _pick_condition_images(custom_prompt)
        if isinstance(custom_prompt, dict):
            prompt_ids = custom_prompt.get("prompt_ids", prompt_ids)
            prompt_mask = custom_prompt.get("prompt_mask", prompt_mask)
            negative_prompt_ids = custom_prompt.get("negative_prompt_ids", negative_prompt_ids)
            negative_prompt_mask = custom_prompt.get("negative_prompt_mask", negative_prompt_mask)
            image_latents = custom_prompt.get("image_latents", image_latents)
            additional_information = custom_prompt.get("additional_information", {})
            vae_images = additional_information.get("vae_images")
            vae_image_sizes = additional_information.get("vae_image_sizes")
        else:
            vae_images = None
            vae_image_sizes = None

        sampling_params = req.sampling_params
        height = sampling_params.height or self.default_sample_size * self.vae_scale_factor
        width = sampling_params.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = sampling_params.num_inference_steps or num_inference_steps
        max_sequence_length = sampling_params.max_sequence_length or max_sequence_length

        noise_level = _coalesce_not_none(sampling_params.extra_args.get("noise_level", None), noise_level)
        sde_window_size = _coalesce_not_none(sampling_params.extra_args.get("sde_window_size", None), sde_window_size)
        sde_window_range = _coalesce_not_none(
            sampling_params.extra_args.get("sde_window_range", None), sde_window_range
        )
        sde_type = _coalesce_not_none(sampling_params.extra_args.get("sde_type", None), sde_type)
        logprobs = _coalesce_not_none(sampling_params.extra_args.get("logprobs", None), logprobs)

        generator = sampling_params.generator or generator
        if generator is None and sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling_params.seed)
        true_cfg_scale = _coalesce_not_none(sampling_params.true_cfg_scale, true_cfg_scale)

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

        has_neg_prompt = negative_prompt_ids is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

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
        latents, prepared_image_latents = self.prepare_latents(
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

        # Use pre-encoded image_latents from the request if provided, otherwise
        # fall back to what prepare_latents returned (None when images=None).
        if image_latents is None:
            image_latents = prepared_image_latents
        if image_latents is not None:
            image_latents = image_latents.to(device=self.device, dtype=prompt_embeds.dtype)

        # Build img_shapes (includes both target and condition image shapes).
        # Use a list comprehension instead of ``[[…]] * batch_size`` so the
        # outer entries do not alias the same inner list — protects against
        # latent batch_size > 1 callers that mutate per-sample shapes.
        target_shape = (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)
        if image_latents is not None:
            if not vae_image_sizes:
                # ``vae_image_sizes`` is populated by the upstream
                # ``pre_process_func`` whenever ``vae_images`` are encoded.
                # Without it we cannot recover the condition image's aspect
                # ratio from the packed sequence length alone (sqrt(seq_len)
                # only equals the side for perfect squares and otherwise
                # silently produces wrong RoPE positions). Fail closed for
                # both ``None`` and the empty-list case — an empty list
                # would still pass the old ``is None`` check and produce
                # an ``img_shapes`` entry with no condition regions, which
                # silently corrupts the transformer's RoPE positions.
                raise ValueError(
                    "QwenImageEditPlusPipelineWithLogProb.forward() requires a "
                    "non-empty additional_information['vae_image_sizes'] when "
                    f"image_latents is provided; got {vae_image_sizes!r}. The "
                    "upstream pre_process_func sets this field — bypass it only "
                    "if you also supply matching vae_image_sizes."
                )
            condition_shapes = [
                (1, vae_height // self.vae_scale_factor // 2, vae_width // self.vae_scale_factor // 2)
                for vae_width, vae_height in vae_image_sizes
            ]
            img_shapes = [[target_shape, *condition_shapes] for _ in range(batch_size)]
        else:
            img_shapes = [[target_shape] for _ in range(batch_size)]

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
            start = torch.randint(
                sde_window_range[0],
                sde_window_range[1] - sde_window_size + 1,
                (1,),
                generator=generator,
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
            image_latents,
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
                "image_latents": _maybe_to_cpu(image_latents),
                "img_shapes": img_shapes,
            },
        )
