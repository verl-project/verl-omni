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
"""MiniMax H3 text-to-audio-video rollout for DiffusionART: the official ODE sampler plus every visited state."""

from __future__ import annotations

from typing import Any

import torch
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
    MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    MINIMAX_H3_IMGVID_COND_TIMESTEP,
    minimax_h3_denoise_loop,
)

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import AUDIO_ROW_WIDTH
from verl_omni.pipelines.minimax_h3_diffusion_nft.vllm_omni_rollout_adapter import MiniMaxH3DiffusionNFTPipeline
from verl_omni.pipelines.model_base import VllmOmniPipelineBase

from .common import art_video_seq_len, rows_to_art_rows

__all__ = ["MiniMaxH3DiffusionARTPipeline", "ART_TRAJECTORY_KEYS"]

#: Trajectory fields this adapter adds to the DiffusionNFT rollout payload (training rollouts only).
ART_TRAJECTORY_KEYS = ("all_latents", "all_timesteps", "all_segment_timesteps", "video_seq_len")


@VllmOmniPipelineBase.register("MiniMaxH3Pipeline", algorithm="diffusion_art")
class MiniMaxH3DiffusionARTPipeline(MiniMaxH3DiffusionNFTPipeline):
    """Rollout pipeline for MiniMax H3 used by DiffusionART (text-to-audio-video only).

    The sampler is the official ``minimax_h3_denoise_loop`` (deterministic Euler on the video and
    audio sigma grids, exactly what the DiffusionNFT rollout runs), so the two recipes share the
    rollout distribution; this class only records the state before every transition through the
    loop's ``on_step`` hook. States are published in the DiffusionART token layout
    (:mod:`.common`): ``all_latents`` ``(1, T + 1, 3 * video_rows + audio_rows, 32)`` whose last entry
    is the clean endpoint, ``all_timesteps`` ``(1, T)`` = video sigma x 1000 (the diffusers timestep
    convention the ART engine divides by ``num_train_timesteps``), ``all_segment_timesteps``
    ``(1, T, 2)`` = ``[video, audio]`` sigma x 1000 (H3 denoises audio on a shift-3 grid while video
    uses shift 12, so the loss weights and trust-region mask of each modality use its own flow time),
    and ``video_seq_len``. Everything the DiffusionNFT adapter publishes (``latent_meta``,
    ``prompt_token_tags``, condition rows, ``latents_clean``) is kept, so the training adapter can
    rebuild the packed layout per sample. Validation requests (no ``global_steps`` in the sampling
    extras) skip the trajectory and run the unmodified DiffusionNFT path.
    """

    def __init__(self, *, od_config: Any, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self._art_capture_enabled = False
        self._art_trajectory: dict[str, torch.Tensor] | None = None

    def diffuse(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Official H3 denoise loop; additionally record every visited joint state for training rollouts."""
        task = str(kwargs.get("task", "t2va"))
        if task != "t2va":
            raise NotImplementedError(f"MiniMax H3 DiffusionART supports text-to-audio-video only, got task={task!r}.")
        if not self._art_capture_enabled:
            return super().diffuse(**kwargs)
        if kwargs.get("base_schedule") is not None:
            raise NotImplementedError("MiniMax H3 DiffusionART does not support distilled sigma schedules.")

        inputs = self._build_denoise_inputs(**kwargs)
        branch = inputs["branch"]
        if int((~branch.update_mask).sum()) or int((~branch.audio_update_mask).sum()):
            raise RuntimeError("MiniMax H3 DiffusionART t2va layout unexpectedly contains condition rows.")
        sigmas_video = [float(s) for s in inputs["sigmas_video"]]
        sigmas_audio = [float(s) for s in inputs["sigmas_audio"]]
        states: list[torch.Tensor] = [
            rows_to_art_rows(
                inputs["video_rows"].to(device=self.device, dtype=torch.float32),
                inputs["audio_rows"].to(device=self.device, dtype=torch.float32),
            )
        ]
        num_transitions = len(sigmas_video) - 1
        transformer = self._transformer_for_task(task)
        with self._resident_dit_layers_on_device(enabled=transformer is self.transformer):
            with self.progress_bar(total=num_transitions) as progress:

                def _record(step: int, video_rows: torch.Tensor, audio_rows: torch.Tensor) -> None:
                    del step
                    states.append(rows_to_art_rows(video_rows.float(), audio_rows.float()))
                    progress.update()

                video_rows, audio_rows = minimax_h3_denoise_loop(
                    model=transformer,
                    positive=branch,
                    initial_video_rows=inputs["video_rows"],
                    initial_audio_rows=inputs["audio_rows"],
                    keyframe_cond_rows=inputs["cond_anchor"],
                    audio_ref_rows=inputs["audio_anchor"],
                    sigmas_video=sigmas_video,
                    sigmas_audio=sigmas_audio,
                    device=self.device,
                    imgvid_cond_noise_aug_for_inference=MINIMAX_H3_IMGVID_COND_TIMESTEP,
                    audio_cond_noise_aug_for_inference=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
                    on_step=_record,
                )
        if len(states) != num_transitions + 1:
            raise RuntimeError(
                f"MiniMax H3 DiffusionART recorded {len(states)} states for {num_transitions} transitions."
            )
        latent_t, latent_h, latent_w, audio_t = (
            int(kwargs[k]) for k in ("latent_t", "latent_h", "latent_w", "audio_t")
        )
        video_latent, audio_latent = self._unpack_denoised_rows(
            branch, video_rows, audio_rows, latent_t=latent_t, latent_h=latent_h, latent_w=latent_w, audio_t=audio_t
        )
        # Same capture the DiffusionNFT adapter builds for t2va, so ``forward`` publishes the identical metadata.
        self._nft_capture = {
            "video_latent": video_latent,
            "audio_latent": audio_latent,
            "condition_video_rows": video_latent.new_zeros((0, 96), dtype=torch.float32),
            "condition_audio_rows": audio_latent.new_zeros((0, AUDIO_ROW_WIDTH), dtype=torch.float32),
            "keyframe_frame_indices": [],
            "ref_block_meta": None,
            "ref_block_count": 0,
            "task": task,
            "text_embeddings": kwargs.get("text_embeddings"),
            "text_tags": kwargs.get("text_tags"),
            "latent_t": latent_t,
            "latent_h": latent_h,
            "latent_w": latent_w,
            "audio_t": audio_t,
            "num_steps": int(kwargs.get("num_steps", 50)),
            "video_shift": float(kwargs.get("video_shift", self.default_video_shift)),
            "base_schedule": kwargs.get("base_schedule"),
        }
        schedule = torch.tensor([sigmas_video[:-1], sigmas_audio[:-1]], dtype=torch.float32).T * 1000.0  # (T, 2)
        self._art_trajectory = {
            "all_latents": torch.stack(states, dim=0).unsqueeze(0).cpu(),
            "all_timesteps": schedule[:, 0].unsqueeze(0).cpu(),
            "all_segment_timesteps": schedule.unsqueeze(0).cpu(),
            "video_seq_len": torch.tensor([art_video_seq_len(int(inputs["video_rows"].shape[0]))], dtype=torch.long),
        }
        return video_latent, audio_latent

    def forward(self, request: Any):
        """DiffusionNFT payload plus the ART trajectory for training requests."""
        extra_args = request.sampling_params.extra_args or {}
        # The agent loop only stamps ``global_steps`` on training rollouts; validation runs the plain sampler.
        self._art_capture_enabled = "global_steps" in extra_args
        self._art_trajectory = None
        try:
            output = super().forward(request)
        finally:
            self._art_capture_enabled = False
        trajectory, self._art_trajectory = self._art_trajectory, None
        if trajectory is None:
            return output
        envelope = output.output
        if not isinstance(envelope, dict) or "rl" not in (envelope.get("metadata") or {}):
            # Warm-up / placeholder outputs carry no training payload (see the base adapter); nothing to extend.
            return output
        rl = envelope["metadata"]["rl"]
        if int(rl["latent_meta"].reshape(-1)[0]) * 3 != int(trajectory["video_seq_len"].item()):
            raise RuntimeError("MiniMax H3 DiffusionART video_seq_len disagrees with latent_meta.")
        rl.update(trajectory)
        return output
