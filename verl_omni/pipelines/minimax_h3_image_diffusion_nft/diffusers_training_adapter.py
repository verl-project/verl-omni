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
"""MiniMax H3 image training adapter for DiffusionNFT."""

import os
from typing import Any, Optional

import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    build_row_timesteps,
    h3_velocity_to_flow_match,
    prepare_h3_processor_files,
    validate_lora_target_modules,
)
from verl_omni.pipelines.minimax_h3_image_diffusion_nft.common import (
    IMAGE_AUDIO_ROW_WIDTH,
    IMAGE_CONDITION_TIMESTEP,
    IMAGE_ROW_WIDTH,
    PDD_TIMESTEP_SCALE,
    build_image_layout,
    debug_dump,
    debug_enabled,
    ensure_pdd_heads,
    parse_image_latent_meta,
    pdd_dit_timestep,
    pdd_step_indices,
)
from verl_omni.pipelines.minimax_h3_image_diffusion_nft.rollout_kernels import (
    ATTENTION_SEGMENTS as ROLLOUT_ATTENTION_SEGMENTS,
)
from verl_omni.pipelines.minimax_h3_image_diffusion_nft.rollout_kernels import CALLS as ROLLOUT_KERNEL_CALLS
from verl_omni.pipelines.minimax_h3_image_diffusion_nft.rollout_kernels import install_rollout_kernels
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config import DiffusionModelConfig

__all__ = ["MiniMaxH3ImageDiffusionNFT"]


def _pdd_sigmas(micro_batch: TensorDict) -> torch.Tensor:
    sigmas = micro_batch.get("pdd_sigmas", None)
    if sigmas is None:
        raise ValueError(
            "MiniMax H3 image DiffusionNFT requires the rollout's `pdd_sigmas`; "
            "use the MiniMaxH3ImagePipeline rollout adapter."
        )
    return sigmas.reshape(sigmas.shape[0], -1)


def _exact_sigmas(timesteps: torch.Tensor, pdd_sigmas: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Snap possibly truncated train timesteps to ``(step index, rollout sigma)``."""
    steps = pdd_step_indices(timesteps, pdd_sigmas)
    sigmas = pdd_sigmas.to(device=steps.device, dtype=torch.float32).gather(1, steps.unsqueeze(1)).squeeze(1)
    return steps, sigmas


@DiffusionModelBase.register("MiniMaxH3ImagePipeline", algorithm="diffusion_nft")
class MiniMaxH3ImageDiffusionNFT(DiffusionModelBase):
    """Forward-process adapter for the 6-step PDD MiniMax H3 image DiT (t2i and edit)."""

    pack_micro_batch = os.environ.get("VERL_OMNI_H3_IMAGE_PACK_MICRO_BATCH", "0") == "1"
    use_rollout_kernels = os.environ.get("VERL_OMNI_H3_IMAGE_ROLLOUT_KERNELS", "1") != "0"

    @classmethod
    def validate_lora_config(cls, model_config: DiffusionModelConfig) -> None:
        """Reject LoRA targets the rollout weight sync cannot transport (shares the H3 whitelist)."""
        if model_config.lora_rank > 0:
            validate_lora_target_modules(model_config.target_modules)

    @classmethod
    def validate_engine_config(cls, engine_config) -> None:
        """Reject Actor sequence parallelism, whose output-head gather precedes the per-step head."""
        if int(getattr(engine_config, "ulysses_sequence_parallel_size", 1) or 1) > 1:
            raise NotImplementedError("MiniMax H3 image DiffusionNFT does not support Actor Ulysses SP yet.")

    @classmethod
    def prepare_processor_files(cls, model_path: str) -> str:
        """Make the official Qwen3-VL processor discoverable by AutoProcessor."""
        return prepare_h3_processor_files(model_path)

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        """Provide the timestep scale only; the PDD sigma grid arrives with every rollout sample."""
        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
        cls.set_timesteps(scheduler, model_config, device="cpu")
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler, model_config: DiffusionModelConfig, device: str):
        """Keep the scheduler's step count aligned with the configured rollout."""
        scheduler.set_timesteps(model_config.pipeline.num_inference_steps, device=device)

    @classmethod
    def prepare_training_inputs(
        cls, micro_batch: TensorDict, step: int, scheduler
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Noise the clean latents at the rollout's exact sigma, undoing the ``.long()`` truncation."""
        scale = float(scheduler.config.num_train_timesteps)
        if scale != PDD_TIMESTEP_SCALE:
            raise ValueError(f"MiniMax H3 image expects num_train_timesteps={PDD_TIMESTEP_SCALE:g}, got {scale:g}.")
        tu.assign_non_tensor(micro_batch, num_train_timesteps=int(scale))
        _, sigmas = _exact_sigmas(micro_batch["train_timesteps"][:, step], _pdd_sigmas(micro_batch))
        context = cls._noise_latents(micro_batch, "latents_clean", "forward_noise", step, sigmas)
        if debug_enabled():
            from verl_omni.pipelines.minimax_h3_image_diffusion_nft.checkpoint import tensor_sha256

            clean = micro_batch["latents_clean"]
            debug_dump(
                "train_in",
                {
                    "step": int(step),
                    "latents_sha256": [tensor_sha256(row.detach().float().cpu()) for row in clean],
                    "latents_dtype": str(clean.dtype),
                    "sigmas": sigmas.detach().float().cpu(),
                    "block_calls": dict(ROLLOUT_KERNEL_CALLS),
                },
            )
        return context["xt"], sigmas * PDD_TIMESTEP_SCALE, context

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
        """Unpack image rows and resolve each sample's PDD step; there is no negative branch."""
        del step, negative_prompt_embeds, negative_prompt_embeds_mask
        num_rows, latent_h, latent_w = parse_image_latent_meta(micro_batch["latent_meta"])
        pdd_steps, sigmas = _exact_sigmas(timesteps, _pdd_sigmas(micro_batch))
        condition_rows = micro_batch.get("condition_video_rows", None)
        if condition_rows is None:
            condition_rows = latents.new_zeros((latents.shape[0], 0, IMAGE_ROW_WIDTH))
        model_inputs = {
            "image_rows": latents.reshape(latents.shape[0], num_rows, IMAGE_ROW_WIDTH),
            "condition_rows": condition_rows,
            "condition_row_count": micro_batch.get("condition_video_row_count", None),
            "ref_block_meta": micro_batch.get("ref_block_meta", None),
            "ref_block_count": micro_batch.get("ref_block_count", None),
            "prompt_token_tags": micro_batch.get("prompt_token_tags", None),
            "encoder_hidden_states": prompt_embeds,
            "encoder_mask": prompt_embeds_mask,
            "timestep": pdd_dit_timestep(sigmas),
            "pdd_step": pdd_steps,
            "latent_grid": (latent_h, latent_w),
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
        """Run the DiT per sample with its step's fused head and return flow-match velocities."""
        del negative_model_inputs
        image_rows = model_inputs["image_rows"]
        condition_rows = model_inputs["condition_rows"]
        condition_row_count = model_inputs["condition_row_count"]
        ref_block_meta = model_inputs["ref_block_meta"]
        ref_block_count = model_inputs["ref_block_count"]
        prompt_token_tags = model_inputs["prompt_token_tags"]
        encoder_hidden_states = model_inputs["encoder_hidden_states"]
        encoder_mask = model_inputs["encoder_mask"]
        timestep = model_inputs["timestep"]
        pdd_steps = model_inputs["pdd_step"].tolist()
        latent_h, latent_w = model_inputs["latent_grid"]
        device = image_rows.device

        head = ensure_pdd_heads(module, model_config)
        if cls.use_rollout_kernels:
            install_rollout_kernels(module)
        batch = image_rows.shape[0]
        if encoder_mask is not None:
            text_lengths = encoder_mask.long().sum(dim=1).tolist()
        else:
            text_lengths = [encoder_hidden_states.shape[1]] * batch
        empty_audio = image_rows.new_zeros((1, 0, IMAGE_AUDIO_ROW_WIDTH))
        no_audio = torch.empty(0, dtype=torch.long)

        if cls.pack_micro_batch and cls.use_rollout_kernels and batch > 1:
            return cls._forward_packed(module, head, model_inputs, text_lengths)

        velocities = []
        for index in range(batch):
            num_text_tokens = int(text_lengths[index])
            num_refs = 0 if ref_block_count is None else int(ref_block_count[index].reshape(-1)[0])
            position_ids, token_tags, image_indices, text_indices, num_condition = build_image_layout(
                latent_h,
                latent_w,
                num_text_tokens,
                ref_block_meta=None if ref_block_meta is None else ref_block_meta[index],
                ref_block_count=num_refs,
                text_token_tags=None if prompt_token_tags is None else prompt_token_tags[index, :num_text_tokens],
            )
            sample_condition = condition_rows[index]
            if condition_row_count is not None:
                sample_condition = sample_condition[: int(condition_row_count[index].reshape(-1)[0])]
            if sample_condition.shape[0] != num_condition:
                raise ValueError(
                    f"MiniMax H3 image condition rows {sample_condition.shape[0]} "
                    f"do not match layout rows {num_condition}."
                )
            unique_timesteps, timestep_indices = build_row_timesteps(
                image_indices,
                no_audio,
                num_condition,
                0,
                num_text_tokens,
                video_timestep=float(timestep[index]),
                audio_timestep=float(timestep[index]),
                condition_video_timestep=IMAGE_CONDITION_TIMESTEP,
                condition_audio_timestep=1.0,
            )
            head.pdd_step = int(pdd_steps[index])
            try:
                result = module(
                    hidden_states=torch.cat([sample_condition, image_rows[index]], dim=0).unsqueeze(0),
                    audio_hidden_states=empty_audio,
                    encoder_hidden_states=encoder_hidden_states[index : index + 1, :num_text_tokens],
                    timestep=unique_timesteps.to(device),
                    timestep_indices=timestep_indices.to(device),
                    token_tags=token_tags.to(device),
                    position_ids=position_ids.to(device),
                    video_indices=image_indices.to(device),
                    audio_indices=no_audio.to(device),
                    text_indices=text_indices.to(device),
                    return_dict=False,
                )
            finally:
                head.pdd_step = None
            velocity = result[0][:, num_condition:]
            velocities.append(h3_velocity_to_flow_match(velocity).reshape(1, -1))
        return torch.cat(velocities, dim=0)

    @classmethod
    def _forward_packed(cls, module, head, model_inputs: dict, text_lengths: list[int]) -> torch.Tensor:
        """One module call for the whole micro-batch: the samples are consecutive documents of one packed sequence.

        Every op of the DiT is row-wise except attention, which runs per document (``ATTENTION_SEGMENTS``), and the
        per-step output head (``pdd_segments``).
        """
        if getattr(module, "gradient_checkpointing", False) and torch.is_grad_enabled():
            raise NotImplementedError(
                "VERL_OMNI_H3_IMAGE_PACK_MICRO_BATCH=1 requires gradient checkpointing off: the recomputed "
                "attention would not see the per-document segments."
            )
        image_rows = model_inputs["image_rows"]
        condition_rows = model_inputs["condition_rows"]
        condition_row_count = model_inputs["condition_row_count"]
        ref_block_meta = model_inputs["ref_block_meta"]
        ref_block_count = model_inputs["ref_block_count"]
        prompt_token_tags = model_inputs["prompt_token_tags"]
        encoder_hidden_states = model_inputs["encoder_hidden_states"]
        timestep = model_inputs["timestep"]
        pdd_steps = model_inputs["pdd_step"].tolist()
        latent_h, latent_w = model_inputs["latent_grid"]
        device = image_rows.device
        batch = image_rows.shape[0]
        no_audio = torch.empty(0, dtype=torch.long)

        video_rows, text_rows, positions, tags, video_indices, text_indices = [], [], [], [], [], []
        timesteps, timestep_indices = [], []
        packed_segments, text_segments, head_segments, output_slices = [], [], [], []
        row_offset = text_offset = video_offset = timestep_offset = 0
        for index in range(batch):
            num_text_tokens = int(text_lengths[index])
            num_refs = 0 if ref_block_count is None else int(ref_block_count[index].reshape(-1)[0])
            position_ids, token_tags, image_indices, text_index, num_condition = build_image_layout(
                latent_h,
                latent_w,
                num_text_tokens,
                ref_block_meta=None if ref_block_meta is None else ref_block_meta[index],
                ref_block_count=num_refs,
                text_token_tags=None if prompt_token_tags is None else prompt_token_tags[index, :num_text_tokens],
            )
            sample_condition = condition_rows[index]
            if condition_row_count is not None:
                sample_condition = sample_condition[: int(condition_row_count[index].reshape(-1)[0])]
            if sample_condition.shape[0] != num_condition:
                raise ValueError(
                    f"MiniMax H3 image condition rows {sample_condition.shape[0]} "
                    f"do not match layout rows {num_condition}."
                )
            unique_timesteps, row_timestep_indices = build_row_timesteps(
                image_indices,
                no_audio,
                num_condition,
                0,
                num_text_tokens,
                video_timestep=float(timestep[index]),
                audio_timestep=float(timestep[index]),
                condition_video_timestep=IMAGE_CONDITION_TIMESTEP,
                condition_audio_timestep=1.0,
            )
            length = int(position_ids.shape[0])
            num_video = int(sample_condition.shape[0] + image_rows.shape[1])
            video_rows.extend([sample_condition, image_rows[index]])
            text_rows.append(encoder_hidden_states[index, :num_text_tokens])
            positions.append(position_ids)
            tags.append(token_tags)
            video_indices.append(image_indices + row_offset)
            text_indices.append(text_index + row_offset)
            timesteps.append(unique_timesteps)
            timestep_indices.append(row_timestep_indices + timestep_offset)
            packed_segments.append((row_offset, row_offset + length))
            text_segments.append((text_offset, text_offset + num_text_tokens))
            head_segments.append((row_offset, row_offset + length, int(pdd_steps[index])))
            output_slices.append((video_offset + num_condition, video_offset + num_video))
            row_offset += length
            text_offset += num_text_tokens
            video_offset += num_video
            timestep_offset += int(unique_timesteps.numel())

        ROLLOUT_ATTENTION_SEGMENTS.clear()
        ROLLOUT_ATTENTION_SEGMENTS[row_offset] = packed_segments
        ROLLOUT_ATTENTION_SEGMENTS[text_offset] = text_segments
        head.pdd_segments = head_segments
        try:
            result = module(
                hidden_states=torch.cat(video_rows, dim=0).unsqueeze(0),
                audio_hidden_states=image_rows.new_zeros((1, 0, IMAGE_AUDIO_ROW_WIDTH)),
                encoder_hidden_states=torch.cat(text_rows, dim=0).unsqueeze(0),
                timestep=torch.cat(timesteps).to(device),
                timestep_indices=torch.cat(timestep_indices).to(device),
                token_tags=torch.cat(tags).to(device),
                position_ids=torch.cat(positions).to(device),
                video_indices=torch.cat(video_indices).to(device),
                audio_indices=no_audio.to(device),
                text_indices=torch.cat(text_indices).to(device),
                return_dict=False,
            )
        finally:
            ROLLOUT_ATTENTION_SEGMENTS.clear()
            head.pdd_segments = None
        velocity = result[0]
        return torch.cat(
            [h3_velocity_to_flow_match(velocity[:, start:end]).reshape(1, -1) for start, end in output_slices], dim=0
        )

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
        """Reject reverse-SDE sampling: the fused heads are exact only for the deterministic Euler step."""
        raise NotImplementedError(
            "MiniMaxH3ImageDiffusionNFT is a forward-process objective; the PDD release has no SDE sampler."
        )
