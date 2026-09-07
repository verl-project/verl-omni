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

"""vLLM-Omni rollout adapter for MiniMax H3 T2VA, FL2VA, and Ref2VA FlowGRPO."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
from vllm_omni.diffusion.models.minimax_h3.condition_noise import (
    minimax_h3_audio_cond_noise_aug_rows,
    minimax_h3_imgvid_cond_noise_aug_rows,
)
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
    MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    MINIMAX_H3_IMGVID_COND_TIMESTEP,
    MiniMaxH3DenoiseBranch,
)
from vllm_omni.diffusion.models.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)
from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.diffusion_rollout_output import with_rollout_data
from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    ref2va_reference_image_short_edge,
    serialize_ref_blocks,
    validate_ref2va_reference_image_short_edge,
)
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

from .common import (
    H3_AUDIO_SHIFT,
    H3_VIDEO_SHIFT,
    combine_log_probs,
    configure_flow_scheduler,
    flatten_joint_latents,
    h3_sigma_schedules,
    sample_h3_transition,
)
from .weight_sync import MiniMaxH3WeightSyncMixin

__all__ = ["MiniMaxH3PipelineWithLogProb"]


def _pad_first_dim(value: torch.Tensor, target: int) -> torch.Tensor:
    if value.shape[0] > target:
        raise ValueError(f"MiniMax H3 metadata length {value.shape[0]} exceeds configured cap {target}.")
    return F.pad(value, (0, 0) * (value.ndim - 1) + (0, target - value.shape[0]))


@VllmOmniPipelineBase.register("MiniMaxH3Pipeline", algorithm="flow_grpo")
class MiniMaxH3PipelineWithLogProb(MiniMaxH3WeightSyncMixin, MiniMaxH3Pipeline):
    """Adapt ``MiniMaxH3Pipeline`` for single-request T2VA, FL2VA, and Ref2VA FlowGRPO.

    Overrides:
        - ``__init__`` adds request-scoped FlowGRPO and CPS state.
        - ``diffuse`` replaces the standard denoise loop with CPS sampling and records target-only video/audio
          transitions, log probabilities, and Actor replay metadata.
        - ``forward`` preserves Agent Loop token IDs, configures FlowGRPO, and attaches the trajectory to
          ``DiffusionOutput``.

    The weight-sync mixin extends upstream prompt encoding while retaining its text-encoder TP collectives.
    """

    supports_request_batch = False

    diffusion_io_spec = DiffusionIOSpec(
        primary=MediaSpec("video"),
        auxiliary=(MediaSpec("audio", sample_rate=32000),),
    )

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        self._reference_image_short_edge = validate_ref2va_reference_image_short_edge()
        super().__init__(od_config=od_config, prefix=prefix)
        self.install_h3_lora_layout()
        self._flow_grpo_noise_level = 0.8
        self._flow_grpo_sde_type = "cps"
        self._flow_grpo_window_size: int | None = None
        self._flow_grpo_window_range: list[int] | None = None
        self._flow_grpo_sde_contiguous = True
        self._flow_grpo_seed = 42
        self._flow_grpo_trajectory: dict[str, torch.Tensor] = {}
        self._h3_max_text_len = 1024

    def _configure_flow_grpo(self, request: OmniDiffusionRequest) -> None:
        if request.sampling_params.extra_args is None:
            request.sampling_params.extra_args = {}
        extra_args = request.sampling_params.extra_args
        if int(request.sampling_params.num_outputs_per_prompt or 1) != 1:
            raise NotImplementedError("MiniMax H3 FlowGRPO v1 supports one output per request.")
        self._flow_grpo_noise_level = float(extra_args.get("noise_level", 0.8))
        self._flow_grpo_sde_type = str(extra_args.get("sde_type", "cps"))
        self._flow_grpo_window_size = extra_args.get("sde_window_size")
        self._flow_grpo_window_range = extra_args.get("sde_window_range")
        self._flow_grpo_sde_contiguous = bool(extra_args.get("sde_contiguous", True))
        global_step = int(extra_args.get("global_steps", 1))
        self._flow_grpo_seed = int(extra_args.get("sde_window_seed", 42)) + max(global_step - 1, 0)
        self._h3_max_text_len = int(request.sampling_params.max_sequence_length or 1024)
        self._flow_grpo_trajectory = {}

    def _layout_outputs(
        self,
        branch: MiniMaxH3DenoiseBranch,
        packed: dict[str, torch.Tensor],
        text_embeddings: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        used_seq_len = int(packed["cu_seqlens"][1].item())
        video_rows = int(branch.img_pos.shape[0])
        audio_rows = int(branch.audio_pos.shape[0])
        layout_cap = video_rows + audio_rows + self._h3_max_text_len
        text_len = int(text_embeddings.shape[0])
        if text_len > self._h3_max_text_len:
            raise ValueError(
                f"MiniMax H3 encoded text length {text_len} exceeds max_sequence_length={self._h3_max_text_len}."
            )

        prompt = F.pad(text_embeddings, (0, 0, 0, self._h3_max_text_len - text_len)).unsqueeze(0)
        prompt_mask = F.pad(
            torch.ones(text_len, dtype=torch.long, device=text_embeddings.device),
            (0, self._h3_max_text_len - text_len),
        ).unsqueeze(0)
        position_ids = _pad_first_dim(packed["img_position_ids"][:used_seq_len], layout_cap).unsqueeze(0)
        token_tags = _pad_first_dim(branch.static_kwargs["token_tags"][:used_seq_len], layout_cap).unsqueeze(0)
        text_indices = _pad_first_dim(packed["text_pos"].view(-1), self._h3_max_text_len).unsqueeze(0)
        return {
            "prompt_embeds": prompt,
            "prompt_embeds_mask": prompt_mask,
            "h3_seq_len": torch.tensor([used_seq_len], device=text_embeddings.device),
            "h3_video_rows": torch.tensor([video_rows], device=text_embeddings.device),
            "h3_audio_rows": torch.tensor([audio_rows], device=text_embeddings.device),
            "h3_position_ids": position_ids,
            "h3_token_tags": token_tags,
            "h3_video_indices": branch.img_pos.unsqueeze(0),
            "h3_audio_indices": branch.audio_pos.unsqueeze(0),
            "h3_text_indices": text_indices,
            "h3_video_update_mask": branch.update_mask_dev.unsqueeze(0),
        }

    def _ref2va_replay_outputs(
        self,
        *,
        text_embeddings: torch.Tensor,
        text_tags: torch.Tensor,
        target_video_rows: torch.Tensor,
        target_audio_rows: torch.Tensor,
        visual_anchor: torch.Tensor | None,
        audio_anchor: torch.Tensor | None,
        ref_blocks: list[dict[str, Any]],
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
    ) -> dict[str, torch.Tensor]:
        text_len = int(text_embeddings.shape[0])
        if text_len > self._h3_max_text_len:
            raise ValueError(
                f"MiniMax H3 encoded text length {text_len} exceeds max_sequence_length={self._h3_max_text_len}."
            )
        prompt = F.pad(text_embeddings, (0, 0, 0, self._h3_max_text_len - text_len)).unsqueeze(0)
        prompt_mask = F.pad(
            torch.ones(text_len, dtype=torch.long, device=text_embeddings.device),
            (0, self._h3_max_text_len - text_len),
        ).unsqueeze(0)
        prompt_tags = F.pad(text_tags, (0, self._h3_max_text_len - text_len)).unsqueeze(0)
        ref_block_meta, ref_block_count = serialize_ref_blocks(ref_blocks)
        condition_video = (
            visual_anchor
            if visual_anchor is not None
            else target_video_rows.new_zeros((0, target_video_rows.shape[-1]))
        )
        condition_audio = (
            audio_anchor if audio_anchor is not None else target_audio_rows.new_zeros((0, target_audio_rows.shape[-1]))
        )
        return {
            "prompt_embeds": prompt,
            "prompt_embeds_mask": prompt_mask,
            "prompt_token_tags": prompt_tags,
            "latent_meta": torch.tensor(
                [[target_video_rows.shape[0], target_audio_rows.shape[0], latent_t, latent_h, latent_w, audio_t]],
                dtype=torch.long,
                device=text_embeddings.device,
            ),
            "condition_video_rows": condition_video.unsqueeze(0),
            "condition_audio_rows": condition_audio.unsqueeze(0),
            "condition_video_row_count": torch.tensor([[condition_video.shape[0]]], dtype=torch.long),
            "condition_audio_row_count": torch.tensor([[condition_audio.shape[0]]], dtype=torch.long),
            "ref_block_meta": ref_block_meta.to(text_embeddings.device).unsqueeze(0),
            "ref_block_count": torch.tensor([[ref_block_count]], dtype=torch.long),
        }

    def diffuse(
        self,
        *,
        task: str,
        text_embeddings: torch.Tensor,
        text_tags: torch.Tensor,
        seed: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
        num_frames: int,
        num_steps: int,
        video_shift: float,
        audio_shift: float,
        visual_condition: torch.Tensor | None,
        visual_condition_shape: tuple[int, int, int] | None,
        audio_condition: torch.Tensor | None,
        ref_audio_t: int | None,
        ref_blocks: list[dict[str, Any]] | None = None,
        visual_condition_shapes: list[tuple[int, int, int]] | None = None,
        audio_condition_lengths: list[int] | None = None,
        keyframe_frame_indices: list[int] | None = None,
        base_schedule: Sequence[float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_video_rows, target_audio_rows = self._initial_noise(
            seed=seed,
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
        )
        target_video_rows = target_video_rows.to(self.device)
        target_audio_rows = target_audio_rows.to(self.device)

        if task == "ref2va":
            if not ref_blocks:
                raise ValueError("MiniMax H3 Ref2VA requires reference block metadata.")
            packed = minimax_h3_packed_sequence_ref2va_blocks(
                text_len=int(text_embeddings.shape[0]),
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                audio_t=audio_t,
                ref_blocks=ref_blocks,
            )
        elif task in {"t2va", "fl2va"}:
            keyframe_indices = list(keyframe_frame_indices or [])
            packed = minimax_h3_packed_sequence(
                text_len=int(text_embeddings.shape[0]),
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                audio_t=audio_t,
                include_keyframe_cond=task == "fl2va",
                keyframe_frame_indices=keyframe_indices if task == "fl2va" else None,
                frame_count=num_frames if task == "fl2va" else None,
            )
        else:
            raise NotImplementedError(f"MiniMax H3 FlowGRPO supports t2va, fl2va, and ref2va, got {task!r}.")
        if base_schedule is not None:
            raise NotImplementedError("MiniMax H3 FlowGRPO does not support distilled checkpoint sigma schedules.")
        if not math.isclose(video_shift, H3_VIDEO_SHIFT, rel_tol=0.0, abs_tol=1e-6) or not math.isclose(
            audio_shift, H3_AUDIO_SHIFT, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                "MiniMax H3 FlowGRPO requires video_shift=12.0 and audio_shift=3.0 "
                "to keep rollout and Actor sigma schedules aligned."
            )

        tags = packed["token_tags"].clone()
        tags[packed["text_pos"]] = text_tags.cpu()
        branch = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=text_embeddings,
            token_tags=tags,
            device=self.device,
        )

        visual_anchor = visual_condition
        if task == "fl2va" and (visual_anchor is None or not keyframe_indices):
            raise ValueError("MiniMax H3 FL2VA rollout did not provide complete visual condition metadata.")
        if visual_anchor is not None:
            condition_shapes = visual_condition_shapes
            if condition_shapes is None and visual_condition_shape is not None:
                condition_shapes = [visual_condition_shape]
            if not condition_shapes:
                raise ValueError("MiniMax H3 visual condition shape is missing.")
            visual_anchor = minimax_h3_imgvid_cond_noise_aug_rows(
                visual_anchor,
                condition_shapes=condition_shapes,
                target_latent_t=latent_t,
                imgvid_cond_num_frames=len(condition_shapes),
                seed=seed,
                noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
            ).to(self.device)

        audio_anchor = audio_condition
        if audio_anchor is not None:
            condition_audio_t = audio_condition_lengths
            if condition_audio_t is None and ref_audio_t is not None:
                condition_audio_t = [ref_audio_t]
            if not condition_audio_t:
                raise ValueError("MiniMax H3 reference audio length is missing.")
            audio_anchor = minimax_h3_audio_cond_noise_aug_rows(
                audio_anchor,
                condition_audio_t=condition_audio_t,
                seed=seed,
                noise_aug=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
            ).to(self.device)

        video_rows = target_video_rows.new_zeros((branch.img_pos.shape[0], target_video_rows.shape[-1]))
        video_rows[branch.update_mask_dev] = target_video_rows
        num_condition_video = int((~branch.update_mask_dev).sum().item())
        if visual_anchor is None:
            if num_condition_video:
                raise ValueError("MiniMax H3 Ref2VA visual condition rows are missing.")
        elif visual_anchor.shape[0] != num_condition_video:
            raise ValueError(
                f"MiniMax H3 visual condition rows {visual_anchor.shape[0]} do not match layout rows "
                f"{num_condition_video}."
            )
        else:
            video_rows[~branch.update_mask_dev] = visual_anchor

        audio_rows = target_audio_rows.new_zeros((branch.audio_pos.shape[0], target_audio_rows.shape[-1]))
        audio_rows[branch.audio_update_mask_dev] = target_audio_rows
        num_condition_audio = int((~branch.audio_update_mask_dev).sum().item())
        if audio_anchor is None:
            if num_condition_audio:
                raise ValueError("MiniMax H3 Ref2VA audio condition rows are missing.")
        elif audio_anchor.shape[0] != num_condition_audio:
            raise ValueError(
                f"MiniMax H3 audio condition rows {audio_anchor.shape[0]} do not match layout rows "
                f"{num_condition_audio}."
            )
        else:
            audio_rows[~branch.audio_update_mask_dev] = audio_anchor

        video_sigmas, audio_sigmas = h3_sigma_schedules(num_steps, video_shift, audio_shift)
        video_scheduler = FlowMatchSDEDiscreteScheduler()
        audio_scheduler = FlowMatchSDEDiscreteScheduler()
        configure_flow_scheduler(video_scheduler, video_sigmas, self.device)
        configure_flow_scheduler(audio_scheduler, audio_sigmas, self.device)
        num_transitions = num_steps - 1
        if self._flow_grpo_window_size is None:
            selected = set(range(num_transitions))
        else:
            window_size = int(self._flow_grpo_window_size)
            low, high = self._flow_grpo_window_range or [0, num_transitions]
            high = min(high, num_transitions)
            if low < 0 or window_size <= 0 or high - low < window_size:
                raise ValueError(
                    f"Invalid MiniMax H3 SDE window: size={window_size}, "
                    f"range={[low, high]}, transitions={num_transitions}."
                )
            step_generator = torch.Generator().manual_seed(self._flow_grpo_seed)
            if self._flow_grpo_sde_contiguous:
                start = int(torch.randint(low, high - window_size + 1, (1,), generator=step_generator).item())
                selected = set(range(start, start + window_size))
            else:
                order = torch.randperm(high - low, generator=step_generator)[:window_size].tolist()
                selected = {low + index for index in order}
        generator = torch.Generator(device=self.device).manual_seed(seed + 1)
        current_latents = []
        next_latents = []
        log_probs = []
        step_indices = []
        selected_video_sigmas = []
        selected_audio_sigmas = []

        transformer = self._transformer_for_task(task)
        with self._resident_dit_layers_on_device(enabled=transformer is self.transformer):
            with self.progress_bar(total=num_transitions) as progress:
                for step in range(num_transitions):
                    video_sigma = float(video_sigmas[step])
                    audio_sigma = float(audio_sigmas[step])
                    video_t = 1.0 - video_sigma
                    audio_timestep = 1.0 - audio_sigma
                    self.record_denoise_step(step, normalized_timestep=video_sigma)
                    model_inputs = branch.forward_kwargs(
                        video_rows=video_rows,
                        audio_rows=audio_rows,
                        t_video=video_t,
                        t_audio=audio_timestep,
                        imgvid_cond_timestep=max(video_t, MINIMAX_H3_IMGVID_COND_TIMESTEP),
                        audio_ref_cond_timestep=max(audio_timestep, MINIMAX_H3_AUDIO_REF_COND_TIMESTEP),
                    )
                    video_velocity, audio_velocity = transformer(**model_inputs)
                    is_selected = step in selected
                    video_transition = sample_h3_transition(
                        video_scheduler,
                        video_rows[branch.update_mask_dev].unsqueeze(0),
                        video_velocity[branch.update_mask_dev].unsqueeze(0),
                        step,
                        noise_level=self._flow_grpo_noise_level if is_selected else 0.0,
                        sde_type=self._flow_grpo_sde_type,
                        generator=generator,
                        return_log_prob=is_selected,
                    )
                    audio_transition = sample_h3_transition(
                        audio_scheduler,
                        audio_rows[branch.audio_update_mask_dev].unsqueeze(0),
                        audio_velocity[branch.audio_update_mask_dev].unsqueeze(0),
                        step,
                        noise_level=self._flow_grpo_noise_level if is_selected else 0.0,
                        sde_type=self._flow_grpo_sde_type,
                        generator=generator,
                        return_log_prob=is_selected,
                    )
                    next_video_rows = video_rows.clone()
                    next_video_rows[branch.update_mask_dev] = video_transition[0][0]
                    if visual_anchor is not None:
                        next_video_rows[~branch.update_mask_dev] = visual_anchor
                    next_audio_rows = audio_rows.clone()
                    next_audio_rows[branch.audio_update_mask_dev] = audio_transition[0][0]
                    if audio_anchor is not None:
                        next_audio_rows[~branch.audio_update_mask_dev] = audio_anchor
                    if is_selected:
                        video_log_prob = video_transition[1]
                        audio_log_prob = audio_transition[1]
                        if video_log_prob is None or audio_log_prob is None:
                            raise RuntimeError("MiniMax H3 rollout did not compute log probabilities.")
                        if task == "ref2va":
                            current_video = video_rows[branch.update_mask_dev]
                            current_audio = audio_rows[branch.audio_update_mask_dev]
                            next_video = next_video_rows[branch.update_mask_dev]
                            next_audio = next_audio_rows[branch.audio_update_mask_dev]
                        else:
                            current_video, current_audio = video_rows, audio_rows
                            next_video, next_audio = next_video_rows, next_audio_rows
                        current_latents.append(
                            flatten_joint_latents(current_video.unsqueeze(0), current_audio.unsqueeze(0))
                        )
                        next_latents.append(flatten_joint_latents(next_video.unsqueeze(0), next_audio.unsqueeze(0)))
                        log_probs.append(combine_log_probs(video_log_prob, audio_log_prob))
                        step_indices.append(step)
                        selected_video_sigmas.append(video_sigma)
                        selected_audio_sigmas.append(audio_sigma)
                    video_rows = next_video_rows
                    audio_rows = next_audio_rows
                    progress.update()
        self.record_denoise_step(None)

        if not current_latents:
            raise RuntimeError("MiniMax H3 rollout selected no stochastic transitions.")
        if task == "ref2va":
            replay_outputs = self._ref2va_replay_outputs(
                text_embeddings=text_embeddings,
                text_tags=text_tags,
                target_video_rows=target_video_rows,
                target_audio_rows=target_audio_rows,
                visual_anchor=visual_anchor,
                audio_anchor=audio_anchor,
                ref_blocks=ref_blocks,
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                audio_t=audio_t,
            )
        else:
            replay_outputs = self._layout_outputs(branch, packed, text_embeddings)
        self._flow_grpo_trajectory = {
            "all_latents": torch.stack(current_latents, dim=1),
            "all_next_latents": torch.stack(next_latents, dim=1),
            "all_timesteps": (1.0 - torch.tensor(selected_video_sigmas, device=self.device)).unsqueeze(0),
            "all_log_probs": torch.stack(log_probs, dim=1),
            "h3_step_indices": torch.tensor(step_indices, device=self.device).unsqueeze(0),
            "h3_audio_timesteps": (1.0 - torch.tensor(selected_audio_sigmas, device=self.device)).unsqueeze(0),
            **replay_outputs,
        }

        video_latent = minimax_h3_unpatchify_video_tokens(
            video_rows[branch.update_mask_dev],
            latent_shape=(latent_t, latent_h // 2, latent_w // 2, 24),
            patch_size=(1, 2, 2),
        )
        audio_latent = minimax_h3_unpack_audio_tokens(
            audio_rows[branch.audio_update_mask_dev],
            audio_t=audio_t * 2,
            audio_channel=2,
        )
        return video_latent, audio_latent

    @torch.no_grad()
    def forward(self, request: DiffusionRequestBatch) -> DiffusionOutput:
        if len(request.requests) != 1:
            raise ValueError(f"MiniMax H3 FlowGRPO expects one request, got {len(request.requests)}.")
        req = request.requests[0]
        self._configure_flow_grpo(req)
        extra_args = req.sampling_params.extra_args or {}
        short_edge = extra_args.get(
            "reference_image_short_edge",
            getattr(req.sampling_params, "reference_image_short_edge", None),
        )
        if short_edge is None:
            short_edge = getattr(self, "_reference_image_short_edge", None)
        with ref2va_reference_image_short_edge(short_edge):
            self._ensure_prompt_text(request)
            try:
                output = super().forward(request)
            finally:
                self._h3_prompt_ids = None
        if not self._flow_grpo_trajectory:
            raise RuntimeError("MiniMax H3 FlowGRPO rollout produced no trajectory.")
        trajectory = {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in self._flow_grpo_trajectory.items()
        }
        replay_fields = (
            "all_next_latents",
            "h3_step_indices",
            "h3_audio_timesteps",
            "h3_video_rows",
            "h3_audio_rows",
            "h3_seq_len",
            "h3_position_ids",
            "h3_token_tags",
            "h3_video_indices",
            "h3_audio_indices",
            "h3_text_indices",
            "h3_video_update_mask",
            "prompt_token_tags",
            "latent_meta",
            "condition_video_rows",
            "condition_audio_rows",
            "condition_video_row_count",
            "condition_audio_row_count",
            "ref_block_meta",
            "ref_block_count",
        )
        replay_fields = tuple(key for key in replay_fields if key in trajectory)
        return with_rollout_data(
            output,
            trajectory_latents=trajectory["all_latents"],
            trajectory_log_probs=trajectory["all_log_probs"],
            trajectory_timesteps=trajectory["all_timesteps"],
            prompt_embeddings={
                "prompt_embeds": trajectory["prompt_embeds"],
                "prompt_embeds_mask": trajectory["prompt_embeds_mask"],
            },
            rl={key: trajectory[key] for key in replay_fields},
            to_cpu=True,
        )
