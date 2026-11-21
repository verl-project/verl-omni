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
"""Image-only FlowGRPO rollout using MiniMax H3 transitions on the release PDD grid."""

from typing import Any

import torch
from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3_image
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3_image import (
    MiniMaxH3ImagePipeline,
    minimax_h3_image_run_sigmas,
)

from verl_omni.pipelines.diffusion_rollout_output import (
    install_rollout_postprocessor,
    wants_decoded_preview,
    with_rollout_data,
    with_visual_artifacts,
)
from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    MiniMaxH3RolloutWeightSyncMixin,
    _PromptTokenOverride,
)
from verl_omni.pipelines.minimax_h3_flow_grpo.common import configure_flow_scheduler, sample_h3_transition
from verl_omni.pipelines.minimax_h3_image_flow_grpo.common import (
    image_latent_meta,
)
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.request_batch import sample_per_sample_sde_windows
from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

__all__ = ["MiniMaxH3ImageFlowGRPOPipeline"]

install_rollout_postprocessor(pipeline_minimax_h3_image, "get_minimax_h3_image_post_process_func")


@VllmOmniPipelineBase.register("MiniMaxH3ImagePipeline", algorithm="flow_grpo")
class MiniMaxH3ImageFlowGRPOPipeline(MiniMaxH3RolloutWeightSyncMixin, MiniMaxH3ImagePipeline):
    """Six-step text-to-image rollout with stochastic transitions in the training window."""

    diffusion_io_spec = DiffusionIOSpec(
        artifacts={
            "image_preview": MediaSpec("image", "decoded", "CHW"),
            "image_latent": MediaSpec("image", "latent", "LC"),
        }
    )

    def __init__(self, *, od_config: Any, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        if hasattr(self, "set_progress_bar_config"):
            self.set_progress_bar_config(disable=True)
        if "min_generation_area" not in (od_config.additional_config or {}):
            self.min_generation_area = None
        self._install_lora_layout()
        self._h3_prompt_ids: torch.Tensor | None = None
        self._sampling = None
        self._trajectory = None
        self._final_rows = None

    def encode_prompt(self, prompt: str, references: list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode Agent Loop token IDs verbatim while the native path builds reference vision spans."""
        prompt_ids = getattr(self, "_h3_prompt_ids", None)
        if prompt_ids is None:
            return super()._encode_text(prompt, references)
        tokenizer = self.tokenizer
        self.tokenizer = _PromptTokenOverride(tokenizer, prompt, prompt_ids)
        try:
            return super()._encode_text(prompt, references)
        finally:
            self.tokenizer = tokenizer

    def _encode_text(self, prompt: str, references: list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_prompt(prompt, references)

    def _ensure_prompt_text(self, request: Any) -> None:
        """Include the canonical token IDs in the native prompt-cache key."""
        super()._ensure_prompt_text(request)
        if self._h3_prompt_ids is not None:
            request.prompts[0]["prompt"] = f"[pretokenized:{self._h3_prompt_ids.tolist()}]"

    def _denoise(self, *, hidden, text_tags, grids, anchors, target_noise, latent_h, latent_w, num_steps):
        native = pipeline_minimax_h3_image
        sampling = self._sampling
        extra = (sampling.extra_args or {}) if sampling is not None else {}
        # Warm-up and validation follow the release's exact Euler update.
        if sampling is None or float(extra.get("noise_level", getattr(sampling, "noise_level", 0.0))) == 0:
            self._final_rows = super()._denoise(
                hidden=hidden,
                text_tags=text_tags,
                grids=grids,
                anchors=anchors,
                target_noise=target_noise,
                latent_h=latent_h,
                latent_w=latent_w,
                num_steps=num_steps,
            )
            return self._final_rows
        if grids or anchors.numel():
            raise ValueError("The H3 Image FlowGRPO training recipe expects text-to-image prompts.")

        def option(key, default=None):
            return extra.get(key, getattr(sampling, key, default))

        sigmas = minimax_h3_image_run_sigmas(self.pdd_sigmas)
        scheduler = FlowMatchSDEDiscreteScheduler()
        configure_flow_scheduler(scheduler, sigmas, self.device)
        size = option("sde_window_size")
        bounds = option("sde_window_range") or (0, num_steps - 1)
        size = bounds[1] - bounds[0] if size is None else size
        if not (0 <= bounds[0] < bounds[1] <= num_steps - 1 and 0 < size <= bounds[1] - bounds[0]):
            raise ValueError("H3 Image training windows must exclude the terminal transition to sigma zero.")
        if not option("sde_contiguous", True):
            raise ValueError("H3 Image supports contiguous SDE windows.")
        window_seed = option("sde_window_seed", 42) + max(int(option("global_steps", 1)) - 1, 0)
        [(start, end)] = sample_per_sample_sde_windows(
            sde_window_size=size,
            sde_window_range=bounds,
            num_timesteps=num_steps,
            batch_size=1,
            generator=torch.Generator().manual_seed(window_seed),
            device="cpu",
        )
        generator = torch.Generator(device=self.device).manual_seed(sampling.seed + 1)
        used = len(hidden) + len(target_noise)
        packed = native.minimax_h3_packed_sequence_ref2va_blocks(
            text_len=len(hidden),
            latent_t=1,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=0,
            ref_blocks=[],
            seq_len=native._image_packed_seq_len(used),
        )
        tags = packed["token_tags"].clone()
        tags[packed["text_pos"].view(-1)] = text_tags
        branch = native.MiniMaxH3DenoiseBranch(
            packed=packed, text_embeddings=hidden, token_tags=tags, device=self.device
        )
        branch.prepare_rope_table(self.transformer)
        rows = target_noise.to(self.device)
        audio = rows.new_empty((0, native.MINIMAX_H3_AUDIO_ROW_WIDTH))
        current, scores = [], []
        for k in range(num_steps):
            native.check_request_cancellation()
            t = 1.0 - sigmas[k]
            kwargs = branch.forward_kwargs(
                video_rows=rows,
                audio_rows=audio,
                t_video=t,
                t_audio=t,
                imgvid_cond_timestep=native.MINIMAX_H3_IMGVID_COND_TIMESTEP,
                audio_ref_cond_timestep=1.0,
            )
            kwargs["skip_mask_out_condition"] = True
            velocity, _ = self.transformer(pdd_step=k, **kwargs)
            if start <= k < end:
                sample, score, *_ = sample_h3_transition(
                    scheduler,
                    rows.unsqueeze(0),
                    velocity.unsqueeze(0),
                    k,
                    noise_level=option("noise_level"),
                    sde_type=option("sde_type"),
                    generator=generator,
                )
                current.append(rows)
                rows = sample[0]
                window_end = rows
                scores.append(score)
            else:
                rows = rows + (sigmas[k] - sigmas[k + 1]) * velocity.float()

        self._trajectory = dict(
            trajectory_latents=torch.stack([*current, window_end]).unsqueeze(0),
            trajectory_log_probs=torch.stack(scores, dim=1),
            trajectory_timesteps=scheduler.timesteps[start:end].unsqueeze(0),
            prompt_embeddings={
                "prompt_embeds": hidden.unsqueeze(0),
                "prompt_embeds_mask": torch.ones(1, len(hidden), dtype=torch.long),
            },
            rl={
                "pdd_step_indices": torch.arange(start, end).unsqueeze(0),
                "pdd_step_sigmas": torch.tensor(sigmas[start:end]).unsqueeze(0),
                "latent_meta": image_latent_meta(len(rows), latent_h, latent_w),
                "prompt_token_tags": text_tags.unsqueeze(0),
            },
        )
        self._final_rows = rows
        return rows

    @torch.no_grad()
    def forward(self, request):
        """Generate an image and attach the sampled FlowGRPO transitions."""
        if request.is_dummy_run():
            try:
                return super().forward(request)
            finally:
                self._final_rows = None
        sampling = request.sampling_params
        if sampling.seed is None:
            raise ValueError("H3 Image FlowGRPO requires a per-sample rollout seed.")
        if sampling.generator_device is None:
            sampling.generator_device = "cpu"
        extra = sampling.extra_args or {}
        output_type = extra.get("output_type", sampling.output_type) or "image"
        decode = wants_decoded_preview(output_type, sampling)
        original_type = sampling.output_type
        self._ensure_prompt_text(request)
        self._sampling = sampling
        try:
            sampling.output_type = "pil" if decode else "latent"
            output = super().forward(request)
            result = with_rollout_data(output, **(self._trajectory or {}))
            return with_visual_artifacts(
                result,
                decoded=output.output.permute(0, 3, 1, 2) if decode else None,
                latents=self._final_rows.unsqueeze(0),
                latent_layout="LC",
                output_type=output_type,
                pixel_range="uint8",
                context=f"pipeline={type(self).__name__}, request_id={request.request_id}",
                requested=extra.get("requested_outputs"),
            )
        finally:
            sampling.output_type = original_type
            self._h3_prompt_ids = self._sampling = self._trajectory = self._final_rows = None
