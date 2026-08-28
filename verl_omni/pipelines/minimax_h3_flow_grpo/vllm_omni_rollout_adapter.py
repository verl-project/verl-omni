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

"""vLLM-Omni rollout adapter for MiniMax H3 T2VA FlowGRPO."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import MiniMaxH3DenoiseBranch
from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence
from vllm_omni.diffusion.models.minimax_h3.packed_tokens import (
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.diffusion_rollout_output import with_rollout_data
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
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
    """Adapt ``MiniMaxH3Pipeline`` for single-request T2VA FlowGRPO rollout.

    Overrides:
        - ``__init__`` adds request-scoped FlowGRPO and CPS state.
        - ``diffuse`` replaces the standard denoise loop with CPS sampling and records joint video/audio transitions,
          log probabilities, and Actor replay metadata.
        - ``forward`` restores the prompt text, configures FlowGRPO, and attaches the trajectory to ``DiffusionOutput``.

    Prompt encoding, including text-encoder TP, is handled by the upstream H3 pipeline.
    """

    supports_request_batch = False

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
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

    def _inject_prompt_text(self, request: OmniDiffusionRequest) -> None:
        # TODO: Let upstream H3 consume prompt token IDs directly. Its text
        # encoder uses tensor parallelism, and overriding encode_prompt would
        # duplicate substantial encoding and distributed-communication logic.
        # For now, decode the truncated IDs back to text and reuse upstream.
        if not isinstance(request.prompt, dict):
            raise TypeError("MiniMax H3 rollout expects a dict prompt containing `prompt_token_ids`.")
        token_ids = request.prompt.get("prompt_token_ids")
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().reshape(-1).tolist()
        elif token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        if not token_ids:
            raise ValueError("MiniMax H3 rollout requires non-empty `prompt_token_ids`.")
        prompt = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        if not prompt:
            raise ValueError("MiniMax H3 tokenizer decoded an empty prompt.")
        request.prompt = {**request.prompt, "prompt": prompt}

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
        video_rows, audio_rows = self._initial_noise(
            seed=seed,
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
        )
        video_rows = video_rows.to(self.device)
        audio_rows = audio_rows.to(self.device)

        if task == "ref2va":
            # TODO: Build the Ref2VA packed blocks and preserve their anchors during Actor replay.
            raise NotImplementedError("MiniMax H3 FlowGRPO does not support Ref2VA yet.")
        packed = minimax_h3_packed_sequence(
            text_len=int(text_embeddings.shape[0]),
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
            include_keyframe_cond=task == "fl2va",
        )

        visual_anchor = visual_condition
        if visual_anchor is not None:
            # TODO: Add FL2VA/Ref2VA visual anchors to CPS transitions and Actor replay.
            raise NotImplementedError("MiniMax H3 FlowGRPO does not support visual conditions yet.")

        audio_anchor = audio_condition
        if audio_anchor is not None:
            # TODO: Add Ref2VA audio anchors to CPS transitions and Actor replay.
            raise NotImplementedError("MiniMax H3 FlowGRPO does not support audio conditions yet.")

        if task == "fl2va":
            # A valid FL2VA request should carry a visual condition. Keep a
            # clear boundary in case upstream request validation changes.
            raise NotImplementedError("MiniMax H3 FlowGRPO does not support FL2VA yet.")
        del (
            base_schedule,
            num_frames,
            visual_condition_shape,
            ref_audio_t,
            ref_blocks,
            visual_condition_shapes,
            audio_condition_lengths,
            keyframe_frame_indices,
        )
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
                        imgvid_cond_timestep=video_t,
                        audio_ref_cond_timestep=audio_timestep,
                    )
                    video_velocity, audio_velocity = transformer(**model_inputs)
                    is_selected = step in selected
                    video_transition = sample_h3_transition(
                        video_scheduler,
                        video_rows.unsqueeze(0),
                        video_velocity.unsqueeze(0),
                        step,
                        noise_level=self._flow_grpo_noise_level if is_selected else 0.0,
                        sde_type=self._flow_grpo_sde_type,
                        generator=generator,
                        return_log_prob=is_selected,
                    )
                    audio_transition = sample_h3_transition(
                        audio_scheduler,
                        audio_rows.unsqueeze(0),
                        audio_velocity.unsqueeze(0),
                        step,
                        noise_level=self._flow_grpo_noise_level if is_selected else 0.0,
                        sde_type=self._flow_grpo_sde_type,
                        generator=generator,
                        return_log_prob=is_selected,
                    )
                    if is_selected:
                        video_log_prob = video_transition[1]
                        audio_log_prob = audio_transition[1]
                        if video_log_prob is None or audio_log_prob is None:
                            raise RuntimeError("MiniMax H3 rollout did not compute log probabilities.")
                        current_latents.append(flatten_joint_latents(video_rows.unsqueeze(0), audio_rows.unsqueeze(0)))
                        next_latents.append(flatten_joint_latents(video_transition[0], audio_transition[0]))
                        log_probs.append(
                            combine_log_probs(
                                video_log_prob,
                                audio_log_prob,
                            )
                        )
                        step_indices.append(step)
                        selected_video_sigmas.append(video_sigma)
                        selected_audio_sigmas.append(audio_sigma)
                    video_rows = video_transition[0][0]
                    audio_rows = audio_transition[0][0]
                    progress.update()
        self.record_denoise_step(None)

        if not current_latents:
            raise RuntimeError("MiniMax H3 rollout selected no stochastic transitions.")
        self._flow_grpo_trajectory = {
            "all_latents": torch.stack(current_latents, dim=1),
            "all_next_latents": torch.stack(next_latents, dim=1),
            "all_timesteps": (1.0 - torch.tensor(selected_video_sigmas, device=self.device)).unsqueeze(0),
            "all_log_probs": torch.stack(log_probs, dim=1),
            "h3_step_indices": torch.tensor(step_indices, device=self.device).unsqueeze(0),
            "h3_audio_timesteps": (1.0 - torch.tensor(selected_audio_sigmas, device=self.device)).unsqueeze(0),
            **self._layout_outputs(branch, packed, text_embeddings),
        }

        video_latent = minimax_h3_unpatchify_video_tokens(
            video_rows,
            latent_shape=(latent_t, latent_h // 2, latent_w // 2, 24),
            patch_size=(1, 2, 2),
        )
        audio_latent = minimax_h3_unpack_audio_tokens(audio_rows, audio_t=audio_t * 2, audio_channel=2)
        return video_latent, audio_latent

    @torch.no_grad()
    def forward(self, request: DiffusionRequestBatch) -> DiffusionOutput:
        if len(request.requests) != 1:
            raise ValueError(f"MiniMax H3 FlowGRPO expects one request, got {len(request.requests)}.")
        req = request.requests[0]
        self._inject_prompt_text(req)
        self._configure_flow_grpo(req)
        output = super().forward(request)
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
        )
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
