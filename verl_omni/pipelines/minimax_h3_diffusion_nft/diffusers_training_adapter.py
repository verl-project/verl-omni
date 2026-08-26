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
"""MiniMax H3 training adapter for DiffusionNFT."""

from typing import Optional

import torch
from tensordict import TensorDict

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config import DiffusionModelConfig

from .common import (
    build_layout_from_meta,
    build_row_timesteps,
    h3_dit_timestep,
    h3_velocity_to_flow_match,
    keyframe_indices_to_anchors,
    pack_video_audio_rows,
    prepare_h3_processor_files,
    split_dual_velocity,
    unpack_video_audio_rows,
    validate_lora_target_modules,
)

__all__ = ["MiniMaxH3DiffusionNFT"]


@DiffusionModelBase.register("MiniMaxH3Pipeline", algorithm="diffusion_nft")
class MiniMaxH3DiffusionNFT(DiffusionModelBase):
    """Forward-process MiniMax H3 adapter used by DiffusionNFT."""

    @classmethod
    def validate_lora_config(cls, model_config: DiffusionModelConfig) -> None:
        """Reject LoRA targets the rollout weight sync cannot transport (shares common.py whitelist)."""
        if model_config.lora_rank > 0:
            validate_lora_target_modules(model_config.target_modules)

    @classmethod
    def prepare_processor_files(cls, model_path: str) -> str:
        """Make the official Qwen3-VL processor discoverable by AutoProcessor."""
        return prepare_h3_processor_files(model_path)

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        """Build the video-shifted rectified-flow scheduler."""
        from diffusers import FlowMatchEulerDiscreteScheduler

        pipeline = model_config.pipeline
        scheduler = FlowMatchEulerDiscreteScheduler(shift=pipeline.get("video_flow_shift", 12.0))
        cls.set_timesteps(scheduler, model_config, device="cpu")
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler, model_config: DiffusionModelConfig, device: str):
        """Set video-stream timesteps."""
        scheduler.set_timesteps(model_config.pipeline.num_inference_steps, device=device)

    @classmethod
    def prepare_model_inputs(
        cls,
        module,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: Optional[torch.Tensor],
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_embeds_mask: Optional[torch.Tensor],
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, Optional[dict]]:
        """Unpack joint latents and prepare H3 transformer inputs."""
        del step, negative_prompt_embeds, negative_prompt_embeds_mask
        meta = micro_batch["latent_meta"][0].reshape(-1).tolist()
        num_video_rows, num_audio_rows = int(meta[0]), int(meta[1])
        video_rows, audio_rows = unpack_video_audio_rows(latents, num_video_rows, num_audio_rows)
        condition_video_rows = micro_batch.get("condition_video_rows", None)
        if condition_video_rows is None:
            condition_video_rows = video_rows.new_zeros((video_rows.shape[0], 0, video_rows.shape[-1]))
        frame_indices = micro_batch.get("keyframe_frame_indices", None)
        frame_indices = [] if frame_indices is None else frame_indices[0].reshape(-1).tolist()
        prompt_token_tags = micro_batch.get("prompt_token_tags", None)

        model_inputs = {
            "video_rows": video_rows,
            "audio_rows": audio_rows,
            "condition_video_rows": condition_video_rows,
            "keyframe_anchors": keyframe_indices_to_anchors(frame_indices),
            "prompt_token_tags": prompt_token_tags,
            "encoder_hidden_states": prompt_embeds,
            "encoder_mask": prompt_embeds_mask,
            "timestep": h3_dit_timestep(timesteps.float()),
            "latent_meta": meta,
        }
        return model_inputs, None

    @classmethod
    def forward(
        cls,
        module,
        model_config: DiffusionModelConfig,
        model_inputs: dict,
        negative_model_inputs: Optional[dict] = None,
    ) -> torch.Tensor:
        """Run H3 per sample and return packed flow-match velocities."""
        del negative_model_inputs
        video_rows = model_inputs["video_rows"]
        audio_rows = model_inputs["audio_rows"]
        condition_video_rows = model_inputs["condition_video_rows"]
        keyframe_anchors = model_inputs["keyframe_anchors"]
        prompt_token_tags = model_inputs["prompt_token_tags"]
        encoder_hidden_states = model_inputs["encoder_hidden_states"]
        encoder_mask = model_inputs["encoder_mask"]
        timestep = model_inputs["timestep"]
        meta = model_inputs["latent_meta"]
        device = video_rows.device
        raw_patch = getattr(getattr(module, "config", None), "patch_size", (1, 2, 2))
        patch_size = (int(raw_patch[0]), int(raw_patch[1]), int(raw_patch[2]))

        batch = video_rows.shape[0]
        if encoder_mask is not None:
            text_lengths = encoder_mask.long().sum(dim=1).tolist()  # one host sync per micro-batch
        else:
            text_lengths = [encoder_hidden_states.shape[1]] * batch

        packed_velocities = []
        for index in range(batch):
            num_text_tokens = int(text_lengths[index])
            sample_text_tags = None if prompt_token_tags is None else prompt_token_tags[index, :num_text_tokens]
            position_ids, token_tags, video_indices, audio_indices, text_indices, num_cond_video, num_cond_audio = (
                build_layout_from_meta(
                    meta,
                    num_text_tokens,
                    patch_size,
                    keyframe_anchors=keyframe_anchors,
                    text_token_tags=sample_text_tags,
                )
            )
            sample_condition = condition_video_rows[index]
            if sample_condition.shape[0] != num_cond_video:
                raise ValueError(
                    f"MiniMax H3 condition rows {sample_condition.shape[0]} do not match layout rows {num_cond_video}."
                )
            full_video_rows = torch.cat([sample_condition, video_rows[index]], dim=0).unsqueeze(0)
            video_t = float(timestep[index])
            unique_timesteps, timestep_indices = build_row_timesteps(
                video_indices,
                audio_indices,
                num_cond_video,
                num_cond_audio,
                num_text_tokens,
                video_timestep=video_t,
                audio_timestep=video_t,
                condition_video_timestep=max(video_t, 0.999),
                condition_audio_timestep=video_t,
            )
            result = module(
                hidden_states=full_video_rows,
                audio_hidden_states=audio_rows[index : index + 1],
                encoder_hidden_states=encoder_hidden_states[index : index + 1, :num_text_tokens],
                timestep=unique_timesteps.to(device),
                timestep_indices=timestep_indices.to(device),
                token_tags=token_tags.to(device),
                position_ids=position_ids.to(device),
                video_indices=video_indices.to(device),
                audio_indices=audio_indices.to(device),
                text_indices=text_indices.to(device),
                return_dict=False,
            )
            v_video, v_audio = split_dual_velocity(result)
            v_video = v_video[:, num_cond_video:]
            packed_velocities.append(
                pack_video_audio_rows(h3_velocity_to_flow_match(v_video), h3_velocity_to_flow_match(v_audio))
            )
        return torch.cat(packed_velocities, dim=0)

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module,
        scheduler,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs,
        step: int,
    ):
        """Reject reverse-SDE sampling for the forward-process objective."""
        raise NotImplementedError(
            "MiniMaxH3DiffusionNFT is a forward-process objective and does not "
            "sample the reverse SDE. Reverse-sampling (flow_grpo) is a separate milestone."
        )
