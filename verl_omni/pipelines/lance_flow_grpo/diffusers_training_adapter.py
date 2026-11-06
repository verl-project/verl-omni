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


"""Lance FlowGRPO training using BAGEL's MoT replay with Lance geometry and sigmas."""

from __future__ import annotations

import logging

import torch

from verl_omni.pipelines.bagel_flow_grpo.bagel_model import get_flattened_position_ids
from verl_omni.pipelines.bagel_flow_grpo.diffusers_training_adapter import BagelDiffusion
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

from .common import setup_lance_sigmas
from .lance_model import LanceForTraining

logger = logging.getLogger(__name__)


@DiffusionModelBase.register("OmniLanceForConditionalGeneration", algorithm="flow_grpo")
class LanceDiffusion(BagelDiffusion):
    """DiffusionModelBase wrapper for :class:`LanceForTraining`.

    The image and the video checkpoint share this adapter: nothing here needs to
    know which one is loaded, because ``_get_latent_pos_ids`` adds the temporal
    axis whenever the request itself carries more than one frame.
    """

    @classmethod
    def build_module(cls, model_config: DiffusionModelConfig, torch_dtype: torch.dtype):
        """Load Lance through its own checkpoint path.

        Args:
            model_config: Model config carrying ``local_path``.
            torch_dtype: Target dtype.

        Returns:
            The loaded ``LanceForTraining`` module.
        """
        logger.info("Loading LanceForTraining from %s", model_config.local_path)
        return LanceForTraining.from_pretrained(
            model_config.local_path,
            torch_dtype=torch_dtype,
            trust_remote_code=model_config.trust_remote_code,
        )

    @classmethod
    def set_timesteps(
        cls,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        device: str,
    ) -> None:
        """Apply Lance's sigma schedule.

        Lance's rollout shifts timesteps by 3.5 where BAGEL uses 3.0; the
        trainer has to use the same shift or the replayed log-probs are taken
        at different sigmas than the ones the rollout sampled.

        Args:
            scheduler: Scheduler to configure in place.
            model_config: Model config carrying ``pipeline.num_inference_steps``.
            device: Device for the scheduler buffers.
        """
        setup_lance_sigmas(scheduler, model_config.pipeline.num_inference_steps, device=device)

    @classmethod
    def prepare_model_inputs(
        cls,
        module,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
        micro_batch,
        step: int,
    ) -> tuple[dict, dict]:
        """Pass the rollout's rotary anchor for the latent block into the replay.

        ``LancePipeline._forward_image_edit`` and its video sibling anchor the
        noise block at the reference's own positions rather than after the text,
        so a replay that assumed the text length would rotate the whole block by
        the length of the reference context.  The pipeline exports the value it
        used and the trainer replays on that basis.  A run without the field
        (text-to-image, text-to-video) is unaffected.
        """
        model_inputs, negative_model_inputs = super().prepare_model_inputs(
            module,
            model_config,
            latents,
            timesteps,
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            micro_batch,
            step,
        )
        rope_anchor = micro_batch.get("rope_anchor") if micro_batch is not None else None
        if rope_anchor is not None:
            model_inputs["position_anchor"] = rope_anchor
            negative_model_inputs["position_anchor"] = rope_anchor
        cls._set_reference_timesteps(model_inputs, negative_model_inputs, micro_batch)
        cls._add_edit_condition(model_inputs, negative_model_inputs, micro_batch)
        return model_inputs, negative_model_inputs

    @classmethod
    def _set_reference_timesteps(cls, model_inputs, negative_model_inputs, micro_batch) -> None:
        """Apply timestep zero to reference-frame tokens, matching the rollout forward."""
        pinned = micro_batch.get("frame_condition_token_indexes") if micro_batch is not None else None
        if pinned is None:
            return
        hidden_states = model_inputs["hidden_states"]
        indexes = pinned.to(hidden_states.device, torch.long)
        if indexes.ndim == 1:
            indexes = indexes.unsqueeze(0).expand(hidden_states.shape[0], -1)
        latent_tokens = hidden_states.shape[1]
        for inputs in (model_inputs, negative_model_inputs):
            timestep = inputs["timestep"]
            if timestep.ndim == 1:
                timestep = timestep[:, None].expand(-1, latent_tokens)
            inputs["timestep"] = timestep.scatter(-1, indexes, 0.0)

    @staticmethod
    def _select_scheduler_tokens(
        sample: torch.Tensor,
        model_output: torch.Tensor,
        prev_sample: torch.Tensor,
        scheduler_inputs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exclude fixed reference tokens from log-probs and the returned KL means."""
        pinned = scheduler_inputs.get("frame_condition_token_indexes")
        if pinned is None or pinned.numel() == 0:
            return sample, model_output, prev_sample
        batch_size, num_tokens, channels = sample.shape
        indexes = pinned.to(sample.device, torch.long)
        if indexes.ndim == 1:
            indexes = indexes.unsqueeze(0).expand(batch_size, -1)
        active_tokens = num_tokens - indexes.shape[1]
        if active_tokens <= 0:
            raise ValueError("Lance replay requires at least one unpinned latent token.")
        active = torch.ones((batch_size, num_tokens), dtype=torch.bool, device=sample.device)
        active.scatter_(1, indexes, False)
        positions = torch.arange(num_tokens, device=sample.device).expand(batch_size, -1)
        selected = positions[active].reshape(batch_size, active_tokens, 1).expand(-1, -1, channels)
        return tuple(tensor.gather(1, selected) for tensor in (sample, model_output, prev_sample))

    #: Rollout-exported fields an image-edit trajectory replays with.
    _CONDITION_KEYS = (
        "condition_prefix_ids",
        "condition_prefix_positions",
        "condition_ref_rows",
        "condition_ref_positions",
        "condition_ref_is_gen",
        "condition_latent_positions",
        "condition_latent_grid",
    )

    @classmethod
    def _add_edit_condition(cls, model_inputs, negative_model_inputs, micro_batch) -> None:
        """Attach the exported reference rows, one tail per branch.

        The two CFG branches share everything except the tail: the conditional
        branch carries the instruction, the unconditional one is the same
        sequence with that segment removed, exactly as the pipeline prefilled
        them.
        """
        if micro_batch is None or micro_batch.get("condition_ref_rows") is None:
            return
        shared = {key: micro_batch[key] for key in cls._CONDITION_KEYS if micro_batch.get(key) is not None}
        for inputs, suffix in ((model_inputs, "gen"), (negative_model_inputs, "cfg")):
            inputs["condition"] = {
                **shared,
                "condition_tail_ids": micro_batch[f"condition_{suffix}_tail_ids"],
                "condition_tail_positions": micro_batch[f"condition_{suffix}_tail_positions"],
                "condition_tail_mask": micro_batch[f"condition_{suffix}_tail_mask"],
            }

    @classmethod
    def _get_latent_pos_ids(cls, model_config: DiffusionModelConfig, module, device) -> torch.Tensor:
        """BAGEL's grid, extended with a temporal axis for video requests.

        Frames above one add the ``t * side**2`` rows the video position table is
        indexed with, so the trainer places the same rows
        ``LanceBagel._per_token_mrope_for_video_latent`` places on the rollout
        side.  A single frame is BAGEL's image grid unchanged.
        """
        config = module.config
        num_frames = int(getattr(model_config.pipeline, "num_frames", 1) or 1)
        if num_frames <= 1:
            return super()._get_latent_pos_ids(model_config, module, device)
        latent_ds = config.latent_patch_size * config.vae_downsample
        img_h = min(model_config.pipeline.height // latent_ds, config.max_latent_size)
        img_w = min(model_config.pipeline.width // latent_ds, config.max_latent_size)
        frames = (num_frames - 1) // int(config.vae_downsample_temporal) + 1
        if frames > config.max_num_frames:
            raise ValueError(f"Lance checkpoint supports {config.max_num_frames} latent frames, got {frames}.")
        pos_ids = get_flattened_position_ids(
            img_h * latent_ds,
            img_w * latent_ds,
            latent_ds,
            config.max_latent_size,
            num_frames=frames,
        )
        return pos_ids.to(device)
