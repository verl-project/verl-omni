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

"""Boogu-Image training adapter using its canonical ``boogu-image`` transformer."""

import json
import os
import shutil
from functools import lru_cache
from pathlib import Path
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
    build_boogu_native_scheduler,
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
    channels = vae_config.get("block_out_channels")
    return 2 ** (len(channels) - 1) if channels else 8


def _make_config_checkpointable(module: torch.nn.Module) -> None:
    """Add the config API expected by ``FSDPCheckpointManager`` locally."""
    config = getattr(module, "config", None)
    if config is None or hasattr(config, "save_pretrained"):
        return

    class _CheckpointableConfig(type(config)):  # type: ignore[misc]
        def save_pretrained(self, save_directory, **_):
            module.save_config(save_directory)

    object.__setattr__(module, "_internal_dict", _CheckpointableConfig(config))


@DiffusionModelBase.register("BooguImagePipeline", algorithm="flow_grpo")
class BooguImage(DiffusionModelBase):
    """Serve Boogu T2I and Edit without generic concat-and-crop conditioning."""

    @classmethod
    def prepare_processor_files(cls, model_path: str) -> Optional[str]:
        """Supply the processor config omitted by released Boogu checkpoints."""
        processor_dir = Path(model_path) / "processor"
        if not processor_dir.is_dir():
            return None
        config_path = processor_dir / "config.json"
        if not config_path.is_file():
            mllm_config = Path(model_path) / "mllm" / "config.json"
            if mllm_config.is_file():
                shutil.copyfile(mllm_config, config_path)
            else:
                config_path.write_text(json.dumps({"model_type": "qwen3_vl"}), encoding="utf-8")
        return str(processor_dir)

    @classmethod
    def build_module(cls, model_config: DiffusionModelConfig, torch_dtype: torch.dtype):
        """Load the canonical class because the checkpoint has no ``auto_map``."""
        try:
            from boogu.models.transformers.transformer_boogu import BooguImageTransformer2DModel
        except ImportError as exc:
            raise ImportError(
                "Boogu-Image training requires the `boogu-image` package: "
                "pip install 'boogu-image @ git+https://github.com/boogu-project/Boogu-Image.git'"
            ) from exc

        module = BooguImageTransformer2DModel.from_pretrained(
            model_config.config_path or model_config.local_path,
            subfolder="" if model_config.config_path else model_config.transformer_subfolder,
            torch_dtype=torch_dtype,
        )
        _make_config_checkpointable(module)
        return module

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(model_config.local_path, subfolder="scheduler")
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler: FlowMatchSDEDiscreteScheduler, model_config: DiffusionModelConfig, device: str):
        pipeline = model_config.pipeline
        scale = _load_vae_scale_factor(model_config.local_path)
        configure_boogu_sde_timesteps(
            scheduler,
            native_scheduler=build_boogu_native_scheduler(model_config.local_path),
            num_inference_steps=pipeline.num_inference_steps,
            num_tokens=(int(pipeline.height) // scale) * (int(pipeline.width) // scale),
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
        """Accept the tuple, diffusers output, and raw-tensor return forms."""
        out = module(**model_inputs)
        return out[0] if isinstance(out, tuple) else out.sample if hasattr(out, "sample") else out

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
        """Build one step's inputs, mapping scheduler sigma to Boogu ``t=1-sigma``."""
        hidden_states = latents[:, step]
        num_train_timesteps = _scheduler_num_train_timesteps(model_config.local_path)
        timestep = boogu_timestep_from_scheduler(timesteps[:, step], num_train_timesteps).to(hidden_states.dtype)
        freqs_cis = get_boogu_freqs_cis(module.config.axes_dim_rope, module.config.axes_lens)
        image_latents = micro_batch.get("condition_image_latents", None)
        if image_latents is not None:
            if image_latents.dim() != 4 or image_latents.shape[0] != hidden_states.shape[0]:
                raise ValueError(
                    "condition_image_latents must be (B, C, H, W) with the micro-batch batch size; "
                    f"got {tuple(image_latents.shape)} vs hidden_states {tuple(hidden_states.shape)}."
                )
            image_latents = image_latents.to(device=hidden_states.device, dtype=hidden_states.dtype)
        ref_image_hidden_states = None if image_latents is None else [[image] for image in image_latents]
        shared_inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "freqs_cis": freqs_cis,
            "ref_image_hidden_states": ref_image_hidden_states,
            "return_dict": False,
        }
        model_inputs = {
            **shared_inputs,
            "instruction_hidden_states": prompt_embeds,
            "instruction_attention_mask": prompt_embeds_mask,
        }

        if negative_prompt_embeds is None or negative_prompt_embeds_mask is None:
            return model_inputs, None
        negative_model_inputs = {
            **shared_inputs,
            "instruction_hidden_states": negative_prompt_embeds,
            "instruction_attention_mask": negative_prompt_embeds_mask,
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
        """Run sequential text CFG and one negated-velocity SDE step."""
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
