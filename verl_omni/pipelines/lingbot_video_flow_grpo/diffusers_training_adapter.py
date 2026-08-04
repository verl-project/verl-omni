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

"""Training-side FlowGRPO adapter for the LingBot-Video Dense T2V model."""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
from diffusers import ModelMixin
from tensordict import TensorDict
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

from .common import apply_cfg, guidance_scale, install_flash_attn_interface_compat, shifted_sigmas

__all__ = ["LingBotVideoDenseFlowGRPO"]


def _pipeline_value(pipeline, name: str, default):
    value = getattr(pipeline, name, None)
    if value is None and hasattr(pipeline, "get"):
        value = pipeline.get(name, None)
    return default if value is None else value


# Derived PEFT-capable classes, memoized per base class.  Reusing one derived
# class per base keeps ``isinstance`` checks stable across calls, and binding it
# at module scope makes instances picklable (a bare ``type(...)`` class has no
# importable qualified name).
_PEFT_CAPABLE_CLS_CACHE: dict[type, type] = {}


@DiffusionModelBase.register("LingBotVideoPipeline", algorithm="flow_grpo")
class LingBotVideoDenseFlowGRPO(DiffusionModelBase):
    """FlowGRPO integration for the non-MoE LingBot Video transformer.

    LingBot's transformer is supplied by the optional, pinned
    ``lingbot-video`` package rather than the installed diffusers package.  It
    is loaded as a bare module so FSDP/PEFT see ``blocks.*`` directly.
    """

    @classmethod
    def build_module(cls, model_config: DiffusionModelConfig, torch_dtype: torch.dtype) -> Optional[torch.nn.Module]:
        install_flash_attn_interface_compat(prefer_fa3=model_config.attn_backend == "_flash_3_varlen_hub")
        try:
            from lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel
        except ImportError as exc:
            raise ImportError(
                "LingBot Dense FlowGRPO requires the optional `lingbot-video` dependency. "
                "Install `verl-omni[lingbot-video]` before starting training."
            ) from exc

        module = cls._peft_capable_transformer_cls(LingBotVideoTransformer3DModel).from_pretrained(
            model_config.local_path,
            subfolder=model_config.transformer_subfolder,
            torch_dtype=torch_dtype,
        )
        cls._patch_time_embedder_input_dtype(module)
        cls._patch_config_save_pretrained(module)
        num_experts = int(getattr(module.config, "num_experts", 0))
        if num_experts != 0:
            raise ValueError(
                "LingBotVideoDenseFlowGRPO supports only robbyant/lingbot-video-dense-1.3b "
                f"(transformer.config.num_experts == 0), got {num_experts}."
            )
        return module

    @staticmethod
    def _peft_capable_transformer_cls(base_cls: type) -> type:
        """Return ``base_cls`` extended with diffusers' ``PeftAdapterMixin``.

        Every diffusers transformer the training engine drives inherits
        ``PeftAdapterMixin`` (which supplies ``add_adapter``/``set_adapter``/
        ``disable_adapters``/``load_lora_adapter``), and the shared
        ``LoRAAdapterMixin`` engine is written against that API.  The optional
        ``lingbot-video`` package predates PEFT and ships a bare ``ModelMixin``
        subclass, so we re-add the mixin here rather than teaching the shared
        engine a second, LingBot-only adapter dialect.  ``base_cls`` stays first
        in the MRO so its ``forward`` and config loading are untouched, and the
        module tree keeps its ``blocks.*`` names (no ``base_model.model.``
        wrapper prefix), so FSDP wrapping and LoRA weight export behave exactly
        as they do for the other transformers.  The derived class is memoized and
        bound at module scope so instances stay picklable.
        """
        from diffusers.loaders import PeftAdapterMixin

        if issubclass(base_cls, PeftAdapterMixin):
            return base_cls
        derived = _PEFT_CAPABLE_CLS_CACHE.get(base_cls)
        if derived is None:
            derived = type(f"{base_cls.__name__}WithPeft", (base_cls, PeftAdapterMixin), {"__module__": __name__})
            globals()[derived.__name__] = derived  # make the dynamic class importable for pickling
            _PEFT_CAPABLE_CLS_CACHE[base_cls] = derived
        return derived

    @staticmethod
    def _patch_config_save_pretrained(module: torch.nn.Module) -> None:
        """Make LingBot's FrozenDict config compatible with FSDP checkpoint saving."""

        config = getattr(module, "config", None)
        if config is None or hasattr(config, "save_pretrained"):
            return

        def save_pretrained(self, save_directory: str | os.PathLike) -> None:
            os.makedirs(save_directory, exist_ok=True)
            output_config_file = os.path.join(save_directory, "config.json")
            with open(output_config_file, "w", encoding="utf-8") as f:
                json.dump(dict(self), f, indent=4, sort_keys=True)

        config.save_pretrained = save_pretrained.__get__(config, type(config))

    @staticmethod
    def _patch_time_embedder_input_dtype(module: torch.nn.Module) -> None:
        """Keep LingBot timestep embeddings compatible with FSDP mixed precision.

        The upstream LingBot forward computes ``timestep.float()`` before calling
        ``time_embedder``.  Diffusers' ``TimestepEmbedding`` does not cast that
        projection to its linear weight dtype, so FSDP mixed precision can expose
        a Float input / BFloat16 weight mismatch.  Casting at the module boundary
        is a no-op for fp32 training and follows the dtype used by the active
        linear/LoRA wrapper during mixed-precision forward.

        A narrow pre-hook is used rather than a broader remedy on purpose: the
        fp32 re-promotion happens *inside* the forward (past the top-level
        inputs), so FSDP2's ``cast_forward_inputs`` cannot reach it, and a global
        ``torch.autocast`` over the transformer would also lower the FlowGRPO
        log-prob/SDE math we intend to keep in fp32.
        """
        time_embedder = getattr(module, "time_embedder", None)
        if time_embedder is None or getattr(time_embedder, "_verl_omni_dtype_hook_installed", False):
            return

        def _weight_dtype(embedder: torch.nn.Module) -> torch.dtype | None:
            linear_1 = getattr(embedder, "linear_1", None)
            weight = getattr(linear_1, "weight", None)
            if weight is None and hasattr(linear_1, "base_layer"):
                weight = getattr(linear_1.base_layer, "weight", None)
            return None if weight is None else weight.dtype

        def _cast_sample_to_weight_dtype(embedder: torch.nn.Module, args: tuple):
            if not args or not isinstance(args[0], torch.Tensor):
                return args
            dtype = _weight_dtype(embedder)
            if dtype is None or args[0].dtype == dtype:
                return args
            return (args[0].to(dtype=dtype), *args[1:])

        time_embedder.register_forward_pre_hook(_cast_sample_to_weight_dtype)
        time_embedder._verl_omni_dtype_hook_installed = True

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig) -> FlowMatchSDEDiscreteScheduler:
        scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path=model_config.local_path,
            subfolder="scheduler",
        )
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(
        cls,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        device: str,
    ) -> None:
        pipeline = model_config.pipeline
        num_steps = int(_pipeline_value(pipeline, "num_inference_steps", 40))
        shift = float(_pipeline_value(pipeline, "shift", 3.0))
        scheduler.set_timesteps(num_steps, device=device, sigmas=shifted_sigmas(num_steps, shift))

    @classmethod
    def prepare_model_inputs(
        cls,
        module: ModelMixin,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds_mask: torch.Tensor | None,
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, Optional[dict]]:
        del module, micro_batch
        selected_latents = latents[:, step]
        # LingBot expects the scheduler's [0, 1000] timestep values, not the
        # normalized sigma.  The official pipeline preserves this scale.
        selected_timesteps = timesteps[:, step].float()
        model_inputs = {
            "hidden_states": selected_latents,
            "timestep": selected_timesteps,
            "encoder_hidden_states": prompt_embeds,
            "encoder_attention_mask": prompt_embeds_mask,
            "return_dict": False,
        }

        if guidance_scale(model_config.pipeline) > 1.0:
            if negative_prompt_embeds is None or negative_prompt_embeds_mask is None:
                raise ValueError("LingBot CFG requires negative prompt embeddings and their attention mask.")
            negative_model_inputs: Optional[dict] = {
                "hidden_states": selected_latents,
                "timestep": selected_timesteps,
                "encoder_hidden_states": negative_prompt_embeds,
                "encoder_attention_mask": negative_prompt_embeds_mask,
                "return_dict": False,
            }
        else:
            negative_model_inputs = None
        return model_inputs, negative_model_inputs

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module: ModelMixin,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ):
        if scheduler_inputs is None:
            raise ValueError("LingBot FlowGRPO requires rollout `all_latents` and `all_timesteps`.")
        latents = scheduler_inputs["all_latents"]
        timesteps = scheduler_inputs["all_timesteps"]

        noise_pred = cls.forward(module, model_config, model_inputs)
        scale = guidance_scale(model_config.pipeline)
        if scale > 1.0:
            if negative_model_inputs is None:
                raise ValueError("LingBot CFG requires negative model inputs.")
            noise_pred = apply_cfg(noise_pred, cls.forward(module, model_config, negative_model_inputs), scale)

        _, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = scheduler.sample_previous_step(
            sample=latents[:, step].float(),
            model_output=noise_pred.float(),
            timestep=timesteps[:, step],
            noise_level=model_config.algo.noise_level,
            prev_sample=latents[:, step + 1].float(),
            sde_type=model_config.algo.sde_type,
            return_logprobs=True,
            return_sqrt_dt=True,
        )
        return log_prob, prev_sample_mean, std_dev_t, sqrt_dt
