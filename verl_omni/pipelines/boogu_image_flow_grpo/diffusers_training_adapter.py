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

"""Boogu-Image training-side adapter for diffusers-based diffusion RL.

The training engine loads the checkpoint's canonical
``BooguImageTransformer2DModel`` via ``diffusers.AutoModel`` with
``trust_remote_code=True``. The checkpoint's ``transformer_boogu.py`` is a
shim that re-exports the class from the ``boogu`` package, so the
``boogu-image`` package (https://github.com/boogu-project/Boogu-Image) must be
installed on trainer workers. Set
``actor_rollout_ref.model.trust_remote_code=true`` in the launch config.
"""

import json
import os
from functools import lru_cache
from typing import Optional

import torch
from tensordict import TensorDict
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

from .common import (
    apply_boogu_text_cfg,
    boogu_timestep_from_scheduler,
    configure_boogu_sde_timesteps,
    get_boogu_freqs_cis,
    resolve_text_guidance_scale,
)

__all__ = ["BooguImage"]


@lru_cache(maxsize=8)
def _load_vae_scale_factor(model_path: str) -> int:
    """Read the VAE downsampling factor from the checkpoint (mirrors vllm-omni)."""
    vae_config_path = os.path.join(model_path, "vae", "config.json")
    try:
        with open(vae_config_path) as f:
            vae_config = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return 8
    if "block_out_channels" not in vae_config:
        return 8
    return 2 ** (len(vae_config["block_out_channels"]) - 1)


def _build_boogu_scheduler(model_path: str) -> FlowMatchSDEDiscreteScheduler:
    """Load the SDE scheduler on top of the checkpoint's scheduler config.

    The checkpoint declares Boogu's own ``FlowMatchEulerDiscreteScheduler``
    variant; ``from_pretrained`` keeps its extra keys (``do_shift``,
    ``time_shift_version``, ``seq_len``, ...) on the config, where
    :func:`resolve_time_shift` reads them at ``set_timesteps`` time.
    """
    return FlowMatchSDEDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=model_path,
        subfolder="scheduler",
    )


def _configure_boogu_scheduler(
    scheduler: FlowMatchSDEDiscreteScheduler,
    *,
    model_path: str,
    height: int,
    width: int,
    num_inference_steps: int,
    device: str,
) -> None:
    vae_scale_factor = _load_vae_scale_factor(model_path)
    num_tokens = (int(height) // vae_scale_factor) * (int(width) // vae_scale_factor)
    configure_boogu_sde_timesteps(
        scheduler,
        num_inference_steps=num_inference_steps,
        num_tokens=num_tokens,
        device=device,
    )


@DiffusionModelBase.register("BooguImagePipeline", algorithm="flow_grpo")
class BooguImage(DiffusionModelBase):
    """Training adapter for the Boogu-Image (T2I) diffusion model.

    Registered under ``"BooguImagePipeline"`` — the shared
    ``model_index.json::_class_name`` of the Base (T2I) and Edit (TI2I)
    checkpoints. This adapter covers the text-to-image path; reference-image
    conditioning (``ref_image_hidden_states``) is not wired yet.
    """

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        scheduler = _build_boogu_scheduler(model_config.local_path)
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler: FlowMatchSDEDiscreteScheduler, model_config: DiffusionModelConfig, device: str):
        _configure_boogu_scheduler(
            scheduler,
            model_path=model_config.local_path,
            height=model_config.pipeline.height,
            width=model_config.pipeline.width,
            num_inference_steps=model_config.pipeline.num_inference_steps,
            device=device,
        )

    @classmethod
    def forward(
        cls,
        module,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Single velocity prediction.

        The canonical Boogu transformer returns a plain tensor when
        ``return_dict=False`` (not the diffusers ``(sample,)`` tuple), so the
        base class's ``module(**model_inputs)[0]`` would slice the batch
        dimension instead of unpacking a tuple.
        """
        out = module(**model_inputs)
        if isinstance(out, tuple):
            return out[0]
        if hasattr(out, "sample"):
            return out.sample
        return out

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
    ) -> tuple[dict, Optional[dict]]:
        """Build Boogu-specific transformer inputs for one denoising step.

        The rollout ships scheduler-native timesteps (``sigma * N``,
        descending); the transformer expects Boogu time ``t = 1 - sigma`` in
        ``[0, 1]`` (its embedding layer applies ``timestep_scale`` internally).
        Keyword names follow the canonical
        ``BooguImageTransformer2DModel.forward`` signature verbatim.
        """
        del micro_batch
        hidden_states = latents[:, step]
        num_train_timesteps = _scheduler_num_train_timesteps(model_config.local_path)
        timestep = boogu_timestep_from_scheduler(timesteps[:, step], num_train_timesteps).to(hidden_states.dtype)

        freqs_cis = get_boogu_freqs_cis(module.config.axes_dim_rope, module.config.axes_lens)

        model_inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "instruction_hidden_states": prompt_embeds,
            "freqs_cis": freqs_cis,
            "instruction_attention_mask": prompt_embeds_mask,
            "return_dict": False,
        }

        if negative_prompt_embeds is None or negative_prompt_embeds_mask is None:
            return model_inputs, None

        negative_model_inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "instruction_hidden_states": negative_prompt_embeds,
            "freqs_cis": freqs_cis,
            "instruction_attention_mask": negative_prompt_embeds_mask,
            "return_dict": False,
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
        """Run the transformer (with sequential text CFG) and one SDE step.

        The combined Boogu velocity is **negated** before it reaches the
        diffusers-convention scheduler — see ``common.py`` for the derivation.

        Returns:
            tuple: ``(log_prob, prev_sample_mean, std_dev_t, sqrt_dt)``.
        """
        assert scheduler_inputs is not None
        latents = scheduler_inputs["all_latents"]
        timesteps = scheduler_inputs["all_timesteps"]

        noise_pred = cls.forward(module, model_config, model_inputs)
        guidance_scale = resolve_text_guidance_scale(model_config.pipeline.guidance_scale)
        if guidance_scale > 1.0:
            assert negative_model_inputs is not None, (
                "Boogu-Image text CFG is active (guidance_scale > 1) but the rollout "
                "shipped no negative prompt embeddings. Provide a negative_prompt in "
                "the dataset or set pipeline.guidance_scale=1.0."
            )
            negative_noise_pred = cls.forward(module, model_config, negative_model_inputs)
            noise_pred = apply_boogu_text_cfg(noise_pred, negative_noise_pred, guidance_scale)

        _, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = scheduler.sample_previous_step(
            sample=latents[:, step].float(),
            model_output=noise_pred.float().neg(),
            timestep=timesteps[:, step],
            noise_level=model_config.algo.noise_level,
            prev_sample=latents[:, step + 1].float(),
            sde_type=model_config.algo.sde_type,
            return_logprobs=True,
            return_sqrt_dt=True,
        )
        return log_prob, prev_sample_mean, std_dev_t, sqrt_dt


@lru_cache(maxsize=8)
def _scheduler_num_train_timesteps(model_path: str) -> int:
    config_path = os.path.join(model_path, "scheduler", "scheduler_config.json")
    try:
        with open(config_path) as f:
            return int(json.load(f).get("num_train_timesteps", 1000))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return 1000
