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
"""Replay MiniMax H3 Image PDD transitions for FlowGRPO."""

import json
from functools import lru_cache
from pathlib import Path

import torch
from verl.utils.device import get_device_name
from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence_ref2va_blocks
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3_image import (
    minimax_h3_image_pdd_grid,
    minimax_h3_image_run_sigmas,
)

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import prepare_h3_processor_files, validate_lora_target_modules
from verl_omni.pipelines.minimax_h3_flow_grpo.common import configure_flow_scheduler, sample_h3_transition
from verl_omni.pipelines.minimax_h3_image_flow_grpo.common import parse_image_latent_meta
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler


@lru_cache
def _actor_model_class():
    """Add the missing PEFT interface locally until Diffusers provides it upstream."""
    try:
        from diffusers import MiniMaxH3ImageTransformer2DModel
        from diffusers.loaders import PeftAdapterMixin
    except ImportError as exc:
        raise ImportError("H3 Image FlowGRPO requires Diffusers 0.43.0 or newer.") from exc

    if issubclass(MiniMaxH3ImageTransformer2DModel, PeftAdapterMixin):
        return MiniMaxH3ImageTransformer2DModel

    # TODO: remove after the native Diffusers image model inherits PeftAdapterMixin.
    class _MiniMaxH3ImageActorModel(MiniMaxH3ImageTransformer2DModel, PeftAdapterMixin):
        """Native image model with the standard Diffusers LoRA interface."""

    return _MiniMaxH3ImageActorModel


@DiffusionModelBase.register("MiniMaxH3ImagePipeline", algorithm="flow_grpo")
class MiniMaxH3ImageFlowGRPO(DiffusionModelBase):
    """Score the recorded next image rows with the original step's frozen PDD head."""

    prepare_processor_files = staticmethod(prepare_h3_processor_files)

    @classmethod
    def build_module(cls, model_config, torch_dtype):
        """Load native image weights with a model-local PEFT compatibility mixin."""
        module = _actor_model_class().from_pretrained(
            model_config.config_path or model_config.local_path,
            subfolder="" if model_config.config_path else model_config.transformer_subfolder,
            torch_dtype=torch_dtype,
        )
        module.set_attention_backend(model_config.attn_backend)
        return module

    @classmethod
    def validate_lora_config(cls, model_config):
        """Keep adapter targets compatible with the existing H3 weight mapping."""
        if model_config.lora_rank > 0:
            validate_lora_target_modules(model_config.target_modules)

    @classmethod
    def validate_engine_config(cls, engine_config):
        """Require an unsharded sequence for native per-step PDD heads."""
        if int(getattr(engine_config, "ulysses_sequence_parallel_size", 1) or 1) != 1:
            raise NotImplementedError("H3 Image per-step heads require Actor SP=1.")

    @classmethod
    def build_scheduler(cls, model_config):
        """Build the shared FlowGRPO scheduler on the release sigma grid."""
        scheduler = FlowMatchSDEDiscreteScheduler()
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler, model_config, device):
        """Read the exact PDD grid from the native release directory."""
        release_path = getattr(model_config, "local_path", None) or model_config.path
        config_file = Path(release_path) / "pdd_config.json"
        sigmas = minimax_h3_image_run_sigmas(minimax_h3_image_pdd_grid(json.loads(config_file.read_text())))
        if model_config.pipeline.num_inference_steps != len(sigmas) - 1:
            raise ValueError("H3 Image rollout step count must match the released PDD grid.")
        configure_flow_scheduler(scheduler, sigmas, device)

    @classmethod
    def prepare_model_inputs(
        cls,
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
    ):
        """Select image rows, text embeddings, and replay metadata for one transition."""
        _, height, width = parse_image_latent_meta(micro_batch["latent_meta"])
        return {
            "rows": latents[:, step],
            "heads": micro_batch["pdd_step_indices"][:, step],
            "grid": (height, width),
            "selected_sigmas": micro_batch["pdd_step_sigmas"][:, step],
            "embeddings": prompt_embeds,
            "mask": prompt_embeds_mask,
            "tags": micro_batch["prompt_token_tags"],
        }, None

    @classmethod
    def forward_and_sample_previous_step(
        cls, module, scheduler, model_config, model_inputs, negative_model_inputs, scheduler_inputs, step
    ):
        """Recompute the recorded next state's score with its original PDD head."""
        heads = model_inputs["heads"]
        head = int(heads[0])
        if not torch.all(heads == head):
            raise ValueError("H3 Image requires a shared SDE window within an Actor microbatch.")
        sigma = float(scheduler.sigmas[head])
        if not torch.all(model_inputs["selected_sigmas"] == sigma):
            raise ValueError("H3 Image Actor and rollout PDD sigma grids disagree.")
        height, width = model_inputs["grid"]
        velocities = []
        for i, rows in enumerate(model_inputs["rows"]):
            text = model_inputs["embeddings"][i]
            mask = model_inputs["mask"]
            if mask is not None:
                text = text[: int(mask[i].sum())]
            layout = minimax_h3_packed_sequence_ref2va_blocks(
                text_len=len(text),
                latent_t=1,
                latent_h=height,
                latent_w=width,
                audio_t=0,
                ref_blocks=[],
            )
            device = rows.device
            used = int(layout["cu_seqlens"][1])
            tags = layout["token_tags"][:used].clone().to(device)
            tags[layout["text_pos"].to(device)] = model_inputs["tags"][i, : len(text)].to(device)
            output = module(
                hidden_states=rows.unsqueeze(0),
                encoder_hidden_states=text.unsqueeze(0),
                timestep=torch.tensor([1.0 - sigma], device=device, dtype=torch.float32),
                timestep_indices=torch.zeros(len(tags), device=device, dtype=torch.long),
                token_tags=tags,
                position_ids=layout["img_position_ids"].reshape(-1, 3)[:used].to(device),
                image_indices=layout["img_pos"].to(device),
                text_indices=layout["text_pos"].to(device),
                pdd_step_index=head,
                return_dict=False,
            )
            velocities.append(output[0])
        _, log_prob, mean, std, sqrt_dt = sample_h3_transition(
            scheduler,
            model_inputs["rows"],
            torch.cat(velocities),
            head,
            noise_level=model_config.algo.noise_level,
            sde_type=model_config.algo.sde_type,
            prev_sample=scheduler_inputs["all_latents"][:, step + 1],
        )
        return log_prob, mean, std, sqrt_dt
