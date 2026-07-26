from __future__ import annotations

from typing import Optional

import torch
from tensordict import TensorDict

from verl_omni.pipelines.boogu_image_flow_grpo.common import apply_boogu_text_cfg, build_boogu_freqs_cis
from verl_omni.pipelines.boogu_image_flow_grpo.diffusers_training_adapter import BooguImageFlowGRPO
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config import DiffusionModelConfig

__all__ = ["BooguImageDiffusionNFT"]


@DiffusionModelBase.register("BooguImageTurboPipeline", algorithm="diffusion_nft")
@DiffusionModelBase.register("BooguImagePipeline", algorithm="diffusion_nft")
class BooguImageDiffusionNFT(BooguImageFlowGRPO):
    @classmethod
    def prepare_model_inputs(
        cls,
        module,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_embeds_mask: Optional[torch.Tensor],
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, Optional[dict]]:
        del micro_batch, step
        if prompt_embeds_mask is None:
            raise ValueError("prompt_embeds_mask is required for BooguImageDiffusionNFT.prepare_model_inputs")

        freqs_cis = build_boogu_freqs_cis(
            tuple(module.config.axes_dim_rope),
            tuple(module.config.axes_lens),
            theta=10000,
        )

        model_inputs = {
            "hidden_states": latents,
            "timestep": timesteps / 1000.0,
            "instruction_hidden_states": prompt_embeds,
            "freqs_cis": freqs_cis,
            "instruction_attention_mask": prompt_embeds_mask,
            "ref_image_hidden_states": None,
            "return_dict": False,
        }

        negative_model_inputs = None
        if negative_prompt_embeds is not None:
            if negative_prompt_embeds_mask is None:
                raise ValueError("negative_prompt_embeds_mask is required when negative_prompt_embeds is provided.")
            negative_model_inputs = {
                **model_inputs,
                "instruction_hidden_states": negative_prompt_embeds,
                "instruction_attention_mask": negative_prompt_embeds_mask,
            }

        return model_inputs, negative_model_inputs

    @classmethod
    def forward(
        cls,
        module,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        prediction = module(**model_inputs)
        if model_config.pipeline.true_cfg_scale > 1.0 and negative_model_inputs is not None:
            negative_prediction = module(**negative_model_inputs)
            prediction = apply_boogu_text_cfg(prediction, negative_prediction, model_config.pipeline.true_cfg_scale)
        return prediction
