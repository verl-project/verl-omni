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

"""SD3 CRD training with a conditional policy and a CFG-guided pretrained anchor."""

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.sd3_dgpo.diffusers_training_adapter import StableDiffusion3DGPO

__all__ = ["StableDiffusion3CRD"]


@DiffusionModelBase.register("StableDiffusion3Pipeline", algorithm="crd")
class StableDiffusion3CRD(StableDiffusion3DGPO):
    """Share forward noise across policies, with an independent moving reference."""

    old_policy_adapter = "moving_reference"

    @classmethod
    def prepare_training_inputs(cls, micro_batch, step, scheduler):
        """Construct the exact velocity target without dividing by the timestep."""
        latents, timesteps, context = super().prepare_training_inputs(micro_batch, step, scheduler)
        context["velocity_target"] = (micro_batch["forward_noise"][:, step].float() - context["x0"].float()).detach()
        return latents, timesteps, context

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
        """Build negative conditioning even when the sampling policy uses CFG=1."""
        inputs = cls.build_transformer_inputs(
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            pooled_prompt_embeds=micro_batch["pooled_prompt_embeds"],
        )
        negative = None
        if model_config.pipeline.reference_guidance_scale > 1:
            if negative_prompt_embeds is None or "negative_pooled_prompt_embeds" not in micro_batch:
                raise ValueError(
                    "CRD pretrained-reference CFG requires negative sequence and pooled prompt embeddings."
                )
            negative = cls.build_transformer_inputs(
                latents=latents,
                timesteps=timesteps,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                pooled_prompt_embeds=micro_batch["negative_pooled_prompt_embeds"],
            )
        return inputs, negative

    @classmethod
    def forward_reference(cls, module, model_config, model_inputs, negative_model_inputs):
        """Apply CFG only to the frozen pretrained prediction."""
        prediction = module(**model_inputs)[0]
        scale = model_config.pipeline.reference_guidance_scale
        if scale > 1:
            if negative_model_inputs is None:
                raise ValueError("CRD reference CFG requires negative inputs.")
            negative = module(**negative_model_inputs)[0]
            prediction = negative + scale * (prediction - negative)
        return prediction

    @staticmethod
    def prepare_loss_inputs(micro_batch, step):
        """Keep raw reward separate from the normalized CRD advantage."""
        return {
            "advantages": micro_batch["advantages"][:, step],
            **{key: micro_batch[key] for key in ("raw_rewards", "group_index", "group_size")},
        }
