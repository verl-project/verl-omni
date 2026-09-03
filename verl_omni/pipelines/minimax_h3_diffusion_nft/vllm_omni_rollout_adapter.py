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
"""GPU rollout adapter for MiniMax H3 DiffusionNFT."""

from typing import Any

import torch
from vllm_omni.diffusion.models.minimax_h3.condition_noise import (
    minimax_h3_audio_cond_noise_aug_rows,
    minimax_h3_imgvid_cond_noise_aug_rows,
)
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
    MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    MINIMAX_H3_IMGVID_COND_TIMESTEP,
)
from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
    minimax_h3_pack_audio_latent,
    minimax_h3_patchify_video_latent,
)
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas

from verl_omni.pipelines.diffusion_rollout_output import with_rollout_data
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec

from .common import (
    AUDIO_ROW_WIDTH,
    MiniMaxH3RolloutWeightSyncMixin,
    pack_video_audio_rows,
    ref2va_reference_image_short_edge,
    serialize_ref_blocks,
    validate_ref2va_reference_image_short_edge,
)

__all__ = ["MiniMaxH3DiffusionNFTPipeline"]

_VIDEO_PATCH_SIZE = (1, 2, 2)


@VllmOmniPipelineBase.register("MiniMaxH3Pipeline", algorithm="diffusion_nft")
class MiniMaxH3DiffusionNFTPipeline(MiniMaxH3RolloutWeightSyncMixin, MiniMaxH3Pipeline):
    """Rollout pipeline for MiniMax H3 used by DiffusionNFT."""

    #: Declares the joint video/audio rollout streams so the diffusion strategy
    #: does not hard-code the audio tuple position or its 32 kHz sample rate.
    diffusion_io_spec = DiffusionIOSpec(
        primary=MediaSpec("video"),
        auxiliary=(MediaSpec("audio", sample_rate=32000),),
    )

    def __init__(self, *, od_config: Any, prefix: str = "") -> None:
        self._reference_image_short_edge = validate_ref2va_reference_image_short_edge()
        super().__init__(od_config=od_config, prefix=prefix)
        if hasattr(self, "set_progress_bar_config"):
            self.set_progress_bar_config(disable=True)
        self._install_lora_layout()
        self._nft_capture: dict[str, Any] | None = None

    def diffuse(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the official H3 denoiser and retain Actor condition state."""
        task = str(kwargs.get("task", "t2va"))
        if task not in {"t2va", "fl2va", "ref2va"}:
            raise NotImplementedError(f"MiniMax H3 DiffusionNFT does not support task {task!r}.")

        ref_blocks = list(kwargs.get("ref_blocks") or [])
        if task == "ref2va" and not ref_blocks:
            raise ValueError("MiniMax H3 Ref2VA requires reference block metadata.")

        video_latent, audio_latent = super().diffuse(**kwargs)

        condition_rows = video_latent.new_zeros((0, 96), dtype=torch.float32)
        condition_audio_rows = audio_latent.new_zeros((0, AUDIO_ROW_WIDTH), dtype=torch.float32)
        keyframe_indices: list[int] = []
        seed = int(kwargs.get("seed", 42))

        if task in {"fl2va", "ref2va"}:
            visual_condition = kwargs.get("visual_condition")
            condition_shapes = kwargs.get("visual_condition_shapes")
            if condition_shapes is None and kwargs.get("visual_condition_shape") is not None:
                condition_shapes = [kwargs["visual_condition_shape"]]
            keyframe_indices = list(kwargs.get("keyframe_frame_indices") or [])
            if task == "fl2va" and (visual_condition is None or not condition_shapes or not keyframe_indices):
                raise ValueError("MiniMax H3 fl2va rollout did not provide complete visual condition metadata.")
            # A reference set may be audio-only, so there may be no visual condition rows to aug-noise.
            if visual_condition is not None:
                if not condition_shapes:
                    raise ValueError(f"MiniMax H3 {task} rollout did not provide visual condition shapes.")
                condition_rows = minimax_h3_imgvid_cond_noise_aug_rows(
                    visual_condition,
                    condition_shapes=condition_shapes,
                    target_latent_t=int(kwargs["latent_t"]),
                    imgvid_cond_num_frames=len(condition_shapes),
                    seed=seed,
                    noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
                )

        ref_block_meta = None
        ref_block_count = 0
        if task == "ref2va":
            audio_condition = kwargs.get("audio_condition")
            if audio_condition is not None:
                audio_lengths = kwargs.get("audio_condition_lengths")
                if audio_lengths is None and kwargs.get("ref_audio_t") is not None:
                    audio_lengths = [int(kwargs["ref_audio_t"])]
                if not audio_lengths:
                    raise ValueError("MiniMax H3 Ref2VA reference audio length is missing.")
                condition_audio_rows = minimax_h3_audio_cond_noise_aug_rows(
                    audio_condition,
                    condition_audio_t=audio_lengths,
                    seed=seed,
                    noise_aug=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
                )
            ref_block_meta, ref_block_count = serialize_ref_blocks(ref_blocks)

        self._nft_capture = {
            "video_latent": video_latent,
            "audio_latent": audio_latent,
            "condition_video_rows": condition_rows,
            "condition_audio_rows": condition_audio_rows,
            "keyframe_frame_indices": keyframe_indices,
            "ref_block_meta": ref_block_meta,
            "ref_block_count": ref_block_count,
            "task": task,
            "text_embeddings": kwargs.get("text_embeddings"),
            "text_tags": kwargs.get("text_tags"),
            "latent_t": int(kwargs.get("latent_t", 0)),
            "latent_h": int(kwargs.get("latent_h", 0)),
            "latent_w": int(kwargs.get("latent_w", 0)),
            "audio_t": int(kwargs.get("audio_t", 0)),
            "num_steps": int(kwargs.get("num_steps", 50)),
            "video_shift": float(kwargs.get("video_shift", self.default_video_shift)),
            "base_schedule": kwargs.get("base_schedule"),
        }
        return video_latent, audio_latent

    def forward(self, request: Any):
        """Generate video+audio and attach DiffusionNFT training tensors."""
        if int(request.sampling_params.num_outputs_per_prompt or 1) != 1:
            raise NotImplementedError("MiniMax H3 DiffusionNFT requires one output per rollout request.")
        extra_args = request.sampling_params.extra_args or {}
        short_edge = extra_args.get(
            "reference_image_short_edge",
            getattr(request.sampling_params, "reference_image_short_edge", None),
        )
        if short_edge is None:
            short_edge = getattr(self, "_reference_image_short_edge", None)
        with ref2va_reference_image_short_edge(short_edge):
            self._ensure_prompt_text(request)
            try:
                output = super().forward(request)
            finally:
                self._h3_prompt_ids = None
        capture = self._nft_capture
        self._nft_capture = None
        if capture is None:
            return output

        video_rows = minimax_h3_patchify_video_latent(capture["video_latent"], patch_size=_VIDEO_PATCH_SIZE)
        audio_rows = minimax_h3_pack_audio_latent(capture["audio_latent"])
        latents_clean = pack_video_audio_rows(video_rows, audio_rows).float()
        num_video_rows = int(video_rows.shape[0])
        num_audio_rows = int(audio_rows.shape[0])

        latent_meta = torch.tensor(
            [
                [
                    num_video_rows,
                    num_audio_rows,
                    capture["latent_t"],
                    capture["latent_h"],
                    capture["latent_w"],
                    capture["audio_t"],
                ]
            ],
            dtype=torch.long,
        )

        text_embeddings = capture["text_embeddings"]
        text_tags = capture["text_tags"]
        prompt_embeds = text_embeddings.unsqueeze(0)
        prompt_embeds_mask = torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=prompt_embeds.device)

        train_timesteps = self._build_train_timesteps(capture).unsqueeze(0)

        condition_video_rows = capture["condition_video_rows"]
        rl = {
            "latents_clean": latents_clean,
            "train_timesteps": train_timesteps,
            "latent_meta": latent_meta,
            "prompt_token_tags": text_tags.unsqueeze(0),
            "condition_video_rows": condition_video_rows.unsqueeze(0),
            # Row counts survive cross-worker padding so the Actor can slice back to true rows.
            "condition_video_row_count": torch.tensor([[condition_video_rows.shape[0]]], dtype=torch.long),
            "keyframe_frame_indices": torch.tensor(
                [capture["keyframe_frame_indices"]], dtype=torch.long, device=prompt_embeds.device
            ),
        }
        if capture["task"] == "ref2va":
            condition_audio_rows = capture["condition_audio_rows"]
            rl.update(
                {
                    "condition_audio_rows": condition_audio_rows.unsqueeze(0),
                    "condition_audio_row_count": torch.tensor([[condition_audio_rows.shape[0]]], dtype=torch.long),
                    "ref_block_meta": capture["ref_block_meta"].unsqueeze(0),
                    "ref_block_count": torch.tensor([[capture["ref_block_count"]]], dtype=torch.long),
                }
            )

        return with_rollout_data(
            output,
            prompt_embeddings={
                "prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": prompt_embeds_mask,
            },
            rl=rl,
            to_cpu=True,
        )

    @staticmethod
    def _build_train_timesteps(capture: dict[str, Any]) -> torch.Tensor:
        """Build the video-stream training timestep pool."""
        sigmas = minimax_h3_time_shift_sigmas(
            num_steps=capture["num_steps"],
            shift_scale=capture["video_shift"],
            base_schedule=capture.get("base_schedule"),
        )
        if len(sigmas) < 2:
            raise ValueError(
                "Empty DiffusionNFT train-timestep pool: "
                f"num_steps={capture['num_steps']} yields {len(sigmas)} sigma(s)."
            )
        return torch.tensor(sigmas[:-1], dtype=torch.float32) * 1000.0
