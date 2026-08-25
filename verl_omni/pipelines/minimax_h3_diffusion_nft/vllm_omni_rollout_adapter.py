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
from vllm_omni.diffusion.models.minimax_h3.condition_noise import minimax_h3_imgvid_cond_noise_aug_rows
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import MINIMAX_H3_IMGVID_COND_TIMESTEP
from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
    minimax_h3_pack_audio_latent,
    minimax_h3_patchify_video_latent,
)
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas

from verl_omni.pipelines.diffusion_rollout_output import with_rollout_data
from verl_omni.pipelines.model_base import VllmOmniPipelineBase

from .common import MiniMaxH3RolloutWeightSyncMixin, pack_video_audio_rows

__all__ = ["MiniMaxH3DiffusionNFTPipeline"]

_VIDEO_PATCH_SIZE = (1, 2, 2)


@VllmOmniPipelineBase.register("MiniMaxH3Pipeline", algorithm="diffusion_nft")
class MiniMaxH3DiffusionNFTPipeline(MiniMaxH3RolloutWeightSyncMixin, MiniMaxH3Pipeline):
    """Rollout pipeline for MiniMax H3 used by DiffusionNFT."""

    def __init__(self, *, od_config: Any, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        if hasattr(self, "set_progress_bar_config"):
            self.set_progress_bar_config(disable=True)
        self._install_lora_layout()
        self._nft_capture: dict[str, Any] | None = None

    def diffuse(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the official T2VA/FL2VA denoiser and retain Actor condition state."""
        task = str(kwargs.get("task", "t2va"))
        if task not in {"t2va", "fl2va"}:
            raise NotImplementedError(f"MiniMax H3 DiffusionNFT supports t2va and fl2va, got {task!r}.")
        video_latent, audio_latent = super().diffuse(**kwargs)

        condition_rows = video_latent.new_zeros((0, 96), dtype=torch.float32)
        keyframe_indices: list[int] = []
        if task == "fl2va":
            visual_condition = kwargs.get("visual_condition")
            condition_shapes = kwargs.get("visual_condition_shapes")
            if condition_shapes is None and kwargs.get("visual_condition_shape") is not None:
                condition_shapes = [kwargs["visual_condition_shape"]]
            keyframe_indices = list(kwargs.get("keyframe_frame_indices") or [])
            if visual_condition is None or not condition_shapes or not keyframe_indices:
                raise ValueError("MiniMax H3 FL2VA rollout did not provide complete visual condition metadata.")
            condition_rows = minimax_h3_imgvid_cond_noise_aug_rows(
                visual_condition,
                condition_shapes=condition_shapes,
                target_latent_t=int(kwargs["latent_t"]),
                imgvid_cond_num_frames=len(condition_shapes),
                seed=int(kwargs.get("seed", 42)),
                noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
            )

        self._nft_capture = {
            "video_latent": video_latent,
            "audio_latent": audio_latent,
            "condition_video_rows": condition_rows,
            "keyframe_frame_indices": keyframe_indices,
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

        return with_rollout_data(
            output,
            prompt_embeddings={
                "prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": prompt_embeds_mask,
            },
            rl={
                "latents_clean": latents_clean,
                "train_timesteps": train_timesteps,
                "latent_meta": latent_meta,
                "prompt_token_tags": text_tags.unsqueeze(0),
                "condition_video_rows": capture["condition_video_rows"].unsqueeze(0),
                "keyframe_frame_indices": torch.tensor(
                    [capture["keyframe_frame_indices"]], dtype=torch.long, device=prompt_embeds.device
                ),
            },
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
