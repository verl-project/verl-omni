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

"""CRD reuses the deterministic SD3 endpoint rollout contract."""

import torch

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.sd3_dgpo.vllm_omni_rollout_adapter import StableDiffusion3DGPOPipeline

__all__ = ["StableDiffusion3CRDPipeline"]


@VllmOmniPipelineBase.register("StableDiffusion3Pipeline", algorithm="crd")
class StableDiffusion3CRDPipeline(StableDiffusion3DGPOPipeline):
    """ODE samples with clean latents for centered reward distillation."""

    supports_request_batch = True
    always_encode_negative_prompt = True

    def diffuse(
        self,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        negative_pooled_prompt_embeds: torch.Tensor | None,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        do_cfg: bool,
        guidance_scale: float,
        noise_level: float,
        sde_window: tuple[int, int] | list[tuple[int, int]],
        sde_type: str,
        generator: torch.Generator | list[torch.Generator] | None,
        logprobs: bool,
    ) -> tuple[torch.Tensor, None, None, torch.Tensor]:
        """Skip terminal zero-length ODE transitions while retaining the training schedule."""
        num_steps = len(timesteps)
        while num_steps and timesteps[num_steps - 1] == 0:
            num_steps -= 1
        if num_steps:
            latents, _, _, _ = super().diffuse(
                prompt_embeds,
                pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                latents,
                timesteps[:num_steps],
                do_cfg,
                guidance_scale,
                noise_level,
                sde_window,
                sde_type,
                generator,
                logprobs,
            )
        return latents, None, None, timesteps.unsqueeze(0).expand(latents.shape[0], -1)
