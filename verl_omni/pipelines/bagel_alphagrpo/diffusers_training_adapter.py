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

"""Replay AlphaGRPO's exact thinking and three image CFG contexts."""

import torch

from verl_omni.pipelines.bagel_flow_grpo.diffusers_training_adapter import BagelDiffusion
from verl_omni.pipelines.model_base import DiffusionModelBase

from .bagel_model import BagelForAlphaGRPO


@DiffusionModelBase.register("OmniBagelForConditionalGeneration", algorithm="alphagrpo")
class BagelAlphaGRPO(BagelDiffusion):
    """BAGEL text and image policy adapter."""

    @classmethod
    def build_module(cls, model_config, torch_dtype):
        """Load the language head together with the shared MoT transformer."""
        return BagelForAlphaGRPO.from_pretrained(model_config.local_path, torch_dtype=torch_dtype)

    @classmethod
    def configure_trainable_params(cls, module, model_config):
        """Train both MoT transformer pathways while freezing embeddings and image projections."""
        for name, parameter in module.named_parameters():
            parameter.requires_grad = name.startswith("layers.")
            if parameter.requires_grad:
                parameter.data = parameter.data.float()

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
        """Use rollout token IDs [B, L] for all three CFG branches."""
        inputs, negative = super().prepare_model_inputs(
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
        batch = micro_batch
        inputs["text_token_ids"] = batch["image_condition_ids"]
        inputs["text_attention_mask"] = batch["image_condition_mask"]
        negative["text_token_ids"] = batch["cfg_text_ids"]
        negative["text_attention_mask"] = batch["cfg_text_mask"]
        negative["cfg_image_inputs"] = {
            **inputs,
            "text_token_ids": batch["cfg_image_ids"],
            "text_attention_mask": batch["cfg_image_mask"],
        }
        return inputs, negative

    @staticmethod
    def _forward_rows(module, inputs):
        """Trim each context before forwarding so image RoPE uses that sample's actual length."""
        predictions = []
        for row in range(inputs["hidden_states"].shape[0]):
            length = int(inputs["text_attention_mask"][row].sum().item())
            row_inputs = {key: value[row : row + 1] for key, value in inputs.items()}
            row_inputs["text_token_ids"] = row_inputs["text_token_ids"][:, :length]
            row_inputs["text_attention_mask"] = row_inputs["text_attention_mask"][:, :length]
            predictions.append(module(**row_inputs)[0])
        return torch.cat(predictions)

    @classmethod
    def forward_and_sample_previous_step(
        cls, module, scheduler, model_config, model_inputs, negative_model_inputs, scheduler_inputs, step
    ):
        """Recompute image likelihoods with the rollout's system, prompt and thinking contexts."""
        noise_pred = cls._forward_rows(module, model_inputs)
        cfg = cls._get_cfg_params(model_config)
        sigmas = scheduler_inputs["all_timesteps"][:, step]
        cfg_mask = (sigmas > cfg["cfg_interval_low"]) & (sigmas <= cfg["cfg_interval_high"])
        if cfg["cfg_text_scale"] > 1:
            negative = dict(negative_model_inputs)
            image_inputs = negative.pop("cfg_image_inputs")
            cfg_text_pred = cls._forward_rows(module, negative)
            cfg_image_pred = cls._forward_rows(module, image_inputs) if cfg["cfg_img_scale"] > 1 else None
            guided = cls._combine_cfg(
                noise_pred,
                cfg_text_pred,
                cfg_image_pred,
                cfg["cfg_text_scale"],
                cfg["cfg_img_scale"],
                cfg["cfg_renorm_type"],
                cfg["cfg_renorm_min"],
            )
            noise_pred = torch.where(cfg_mask[:, None, None], guided, noise_pred)
        _, log_prob, mean, std, sqrt_dt = scheduler.sample_previous_step(
            sample=scheduler_inputs["all_latents"][:, step].float(),
            model_output=noise_pred.float(),
            timestep=scheduler_inputs["all_timesteps"][:, step],
            noise_level=model_config.algo.noise_level,
            prev_sample=scheduler_inputs["all_latents"][:, step + 1].float(),
            sde_type=model_config.algo.sde_type,
            return_logprobs=True,
            return_sqrt_dt=True,
            include_logprob_normalizer=False,
        )
        return log_prob, mean, std, sqrt_dt
