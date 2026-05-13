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

"""BAGEL (MoT) training-side adapter for FlowGRPO.

Registers as ``OmniBagelForConditionalGeneration`` so the FSDP engine
can load and train the model via the DiffusionModelBase registry.

Key differences from standard diffusion models (e.g. Qwen-Image):
  * BAGEL is a *Mixture-of-Thought* transformer that processes text token
    IDs and noisy latent patches in a single forward pass (no separate
    text encoder).
  * ``prompt_embeds`` are not used.  Instead the raw prompt token IDs
    (available as ``micro_batch["prompts"]``) are passed directly to the
    model as ``text_token_ids``.
  * CFG uses a 3-branch scheme during rollout, but for FSDP training
    (computing log-probs of the rollout trajectory) only the conditional
    forward is needed.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from tensordict import TensorDict
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

from .bagel_model import BagelForTraining, get_flattened_position_ids

logger = logging.getLogger(__name__)

TIMESTEP_SHIFT = 3.0  # must match BagelPipeline.forward() hardcoded value


@DiffusionModelBase.register("OmniBagelForConditionalGeneration", algorithm="flow_grpo")
class BagelDiffusion(DiffusionModelBase):
    """DiffusionModelBase wrapper for ``BagelForTraining`` (MoT)."""

    @classmethod
    def build_module(cls, model_config: DiffusionModelConfig, torch_dtype: torch.dtype):
        logger.info("Loading BagelForTraining from %s", model_config.local_path)
        return BagelForTraining.from_pretrained(model_config.local_path, torch_dtype=torch_dtype)

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        # Build on GPU so scheduler buffers are comparable with cuda timesteps in FSDP forward.
        scheduler = FlowMatchSDEDiscreteScheduler()
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler: FlowMatchSDEDiscreteScheduler, model_config: DiffusionModelConfig, device: str):
        num_inference_steps = model_config.pipeline.num_inference_steps
        # Use torch.float32 on ``device`` to be bit-exact with BAGEL rollout's
        # ``torch.linspace`` schedule; otherwise ``index_for_timestep`` may miss.
        t = torch.linspace(1, 0, num_inference_steps, dtype=torch.float32, device=device)
        t_shifted = TIMESTEP_SHIFT * t / (1 + (TIMESTEP_SHIFT - 1) * t)
        sigmas = t_shifted[:-1].tolist()

        scheduler.set_shift(1.0)  # identity — sigmas already shifted
        # Pass ``timesteps=sigmas`` to skip diffusers' default ``sigmas * 1000``
        # conversion; BAGEL rollout records raw sigma values as timesteps.
        scheduler.set_timesteps(sigmas=sigmas, timesteps=sigmas, device=device)
        scheduler.set_begin_index(0)

    @classmethod
    def _get_latent_pos_ids(cls, model_config: DiffusionModelConfig, module, device) -> torch.Tensor:
        """Compute latent position IDs from model config / image dimensions."""
        config = module.config
        img_h = model_config.pipeline.height // (config.latent_patch_size * config.vae_downsample)
        img_w = model_config.pipeline.width // (config.latent_patch_size * config.vae_downsample)
        # Clamp to max_latent_size
        img_h = min(img_h, config.max_latent_size)
        img_w = min(img_w, config.max_latent_size)
        latent_ds = config.latent_patch_size * config.vae_downsample
        H_px = img_h * latent_ds
        W_px = img_w * latent_ds
        pos_ids = get_flattened_position_ids(H_px, W_px, latent_ds, config.max_latent_size)
        return pos_ids.to(device)

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
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, dict]:
        B = latents.shape[0]
        device = latents.device

        hidden_states = latents[:, step]
        timestep = timesteps[:, step]

        # Extract text token IDs from prompt data
        prompts = micro_batch["prompts"]  # (B, L_prompt) padded
        attention_mask = micro_batch["attention_mask"]  # (B, L_prompt)

        # Build per-sample text_token_ids (remove padding)
        text_token_ids_list = []
        for i in range(B):
            mask = attention_mask[i].bool()
            ids = prompts[i][mask]
            text_token_ids_list.append(ids)

        # Pad to same length within batch
        max_text_len = max(ids.shape[0] for ids in text_token_ids_list)
        text_token_ids = torch.zeros(B, max_text_len, dtype=torch.long, device=device)
        text_attention_mask = torch.zeros(B, max_text_len, dtype=torch.bool, device=device)
        for i, ids in enumerate(text_token_ids_list):
            text_token_ids[i, : ids.shape[0]] = ids
            text_attention_mask[i, : ids.shape[0]] = True

        # Compute latent position IDs
        latent_pos_ids = cls._get_latent_pos_ids(model_config, module, device)
        latent_pos_ids = latent_pos_ids.unsqueeze(0).expand(B, -1)

        model_inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "text_token_ids": text_token_ids,
            "text_attention_mask": text_attention_mask,
            "latent_pos_ids": latent_pos_ids,
        }

        # For BAGEL, unconditional pass uses text_token_ids=None
        negative_model_inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "text_token_ids": None,
            "latent_pos_ids": latent_pos_ids,
        }

        return model_inputs, negative_model_inputs

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ):
        assert scheduler_inputs is not None
        latents = scheduler_inputs["all_latents"]
        timesteps = scheduler_inputs["all_timesteps"]

        noise_pred = module(**model_inputs)[0]

        # CFG during training (if configured)
        true_cfg_scale = model_config.pipeline.true_cfg_scale
        if true_cfg_scale > 1.0:
            assert negative_model_inputs is not None
            neg_noise_pred = module(**negative_model_inputs)[0]
            comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            noise_pred = comb_pred * (cond_norm / noise_norm)

        _, log_prob, prev_sample_mean, std_dev_t = scheduler.sample_previous_step(
            sample=latents[:, step].float(),
            model_output=noise_pred.float(),
            timestep=timesteps[:, step],
            noise_level=model_config.algo.noise_level,
            prev_sample=latents[:, step + 1].float(),
            sde_type=model_config.algo.sde_type,
            return_logprobs=True,
        )
        return log_prob, prev_sample_mean, std_dev_t
