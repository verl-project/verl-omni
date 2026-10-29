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


"""Lance text-to-image/video rollouts using BAGEL's shared FlowGRPO scheduler."""

from __future__ import annotations

import numpy as np
import torch
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.lance.pipeline_lance import LancePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest

from verl_omni.pipelines.bagel_flow_grpo.common import maybe_to_cpu
from verl_omni.pipelines.bagel_flow_grpo.vllm_omni_rollout_adapter import BagelPipelineWithLogProb
from verl_omni.pipelines.diffusion_rollout_output import rollout_output, with_visual_artifacts
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

from .common import LANCE_TIMESTEP_SHIFT, setup_lance_sigmas


@VllmOmniPipelineBase.register("OmniLanceForConditionalGeneration", algorithm="flow_grpo")
class LancePipelineWithLogProb(BagelPipelineWithLogProb, LancePipeline):
    """Preserve Lance construction and media layout under BAGEL's RL wrapper."""

    diffusion_io_spec = DiffusionIOSpec(
        artifacts={
            "image_preview": MediaSpec("image", "decoded", "CHW"),
            "image_latent": MediaSpec("image", "latent", "LC"),
            "video_preview": MediaSpec("video", "decoded", "TCHW"),
            "video_latent": MediaSpec("video", "latent", "LC"),
        }
    )
    flowgrpo_timestep_shift: float = LANCE_TIMESTEP_SHIFT

    @classmethod
    def flowgrpo_setup_sigmas(cls, scheduler: FlowMatchSDEDiscreteScheduler, num_steps: int, shift: float) -> None:
        """Use Lance's shifted schedule with one extra endpoint."""
        setup_lance_sigmas(scheduler, num_steps, shift=shift)

    def _ensure_bagel_prompt_text(self, req: OmniDiffusionRequest) -> None:
        super()._ensure_bagel_prompt_text(req)
        req.prompts[0]["modalities"] = ["video" if self._is_video else "image"]

    def _decode_video_from_latent(self, bagel, vae, latent, video_shape):
        if self._final_latent is None:
            self._final_latent = latent
        return super()._decode_video_from_latent(bagel, vae, latent, video_shape)

    def _build_flowgrpo_output(
        self,
        req: OmniDiffusionRequest,
        output: DiffusionOutput,
        native_latent: torch.Tensor,
        traj_latents: torch.Tensor | None,
        traj_timesteps: torch.Tensor | None,
        traj_log_probs: torch.Tensor | None,
    ) -> DiffusionOutput:
        """Keep all decoded frames and the unchanged flat video latents."""
        if not self._is_video:
            return super()._build_flowgrpo_output(
                req, output, native_latent, traj_latents, traj_timesteps, traj_log_probs
            )
        payload = output.output["payload"]
        context = f"pipeline={type(self).__name__}, request_id={req.request_id}"
        if "video" not in payload or payload.keys() - {"video", "trajectory"}:
            raise ValueError(f"{context}: expected video and optional trajectory, got {list(payload)}")
        frames = np.stack([np.asarray(frame) for frame in payload["video"]])
        media = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0)  # [B, T, C, H, W]
        result = rollout_output(
            media=media,
            media_key="video",
            trajectory_latents=maybe_to_cpu(traj_latents),
            trajectory_timesteps=maybe_to_cpu(traj_timesteps),
            trajectory_log_probs=maybe_to_cpu(traj_log_probs),
            metadata=output.output.get("metadata"),
            to_cpu=False,
        )
        return with_visual_artifacts(
            result,
            decoded=media,
            latents=maybe_to_cpu(native_latent).unsqueeze(0),
            latent_layout="LC",
            modality="video",
            decoded_layout="TCHW",
            pixel_range="uint8",
            fps=req.sampling_params.frame_rate,
            output_type=req.sampling_params.output_type or "pil",
            context=context,
            requested=(req.sampling_params.extra_args or {}).get("requested_outputs"),
        )
