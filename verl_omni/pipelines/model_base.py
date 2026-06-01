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

from abc import ABC, abstractmethod
from typing import Optional

import torch
from diffusers import ModelMixin, SchedulerMixin
from tensordict import TensorDict

from verl_omni.workers.config import DiffusionModelConfig


class DiffusionModelBase(ABC):
    """Abstract base class for diffusion model training helpers.

    Different diffusion models have very different forward / sampling logic.
    Subclass this ABC and implement the three abstract methods to plug your
    model into the verl training loop.

    To register, decorate your subclass with
    ``@DiffusionModelBase.register("name", algorithm="...")``. The *name* must match the
    ``_class_name`` value in the pipeline's ``model_index.json`` (which is
    auto-detected into ``DiffusionModelConfig.architecture``). The *algorithm*
    must match ``DiffusionModelConfig.algorithm``.

    Example::

        @DiffusionModelBase.register("QwenImagePipeline", algorithm="flow_grpo")
        class QwenImage(DiffusionModelBase):
            ...
    """

    _registry: dict[tuple[str, str], type["DiffusionModelBase"]] = {}

    @classmethod
    def register(cls, architecture: str, algorithm: str):
        """Class decorator that registers a subclass for ``(architecture, algorithm)``."""

        def decorator(subclass: type["DiffusionModelBase"]) -> type["DiffusionModelBase"]:
            cls._registry[(architecture, algorithm)] = subclass
            return subclass

        return decorator

    @classmethod
    def get_class(cls, model_config: DiffusionModelConfig) -> type["DiffusionModelBase"]:
        """Return the registered subclass for ``(architecture, algorithm)``."""
        architecture = model_config.architecture
        algorithm = model_config.algorithm
        key = (architecture, algorithm)

        if key not in cls._registry and model_config.external_lib is not None:
            from verl.utils.import_utils import import_external_libs

            import_external_libs(model_config.external_lib)

        try:
            return cls._registry[key]
        except KeyError:
            registered = sorted(cls._registry.keys())
            raise NotImplementedError(
                f"No diffusion model registered for (architecture={architecture!r}, "
                f"algorithm={algorithm!r}). Registered: {registered}. "
                f"Set ``external_lib`` in DiffusionModelConfig to load your implementation."
            ) from None

    @classmethod
    @abstractmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig) -> SchedulerMixin:
        """Build and configure the diffusion scheduler for this model.
        The returned scheduler should have timesteps and sigmas already set.

        Args:
            model_config (DiffusionModelConfig): the configuration of the diffusion model.
        """
        pass

    @classmethod
    @abstractmethod
    def set_timesteps(cls, scheduler: SchedulerMixin, model_config: DiffusionModelConfig, device: str):
        """Set timesteps and sigmas on the scheduler and move them to *device*.

        Args:
            scheduler (SchedulerMixin): the scheduler used for the diffusion process.
            model_config (DiffusionModelConfig): the configuration of the diffusion model.
            device (str): the device to move the timesteps and sigmas to.
        """
        pass

    @classmethod
    @abstractmethod
    def prepare_model_inputs(
        cls,
        module: ModelMixin,
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
        """Build architecture-specific inputs for reverse-trajectory training.
        This packages the stored rollout ``x_t`` for a model forward; the
        scheduler step that scores/samples toward ``x_{t-1}`` happens later.
        The caller is responsible for universal pre-processing (common tensor extraction
        and nested-embed unpadding) before invoking this method.

        Args:
            module (ModelMixin): the diffusion transformer module.
            model_config (DiffusionModelConfig): the configuration of the diffusion model.
            latents (torch.Tensor): full latent tensor from the micro-batch, shape (B, T, ...).
            timesteps (torch.Tensor): full timestep tensor from the micro-batch, shape (B, T).
            prompt_embeds (torch.Tensor): dense positive prompt embeddings, shape (B, L, D).
            prompt_embeds_mask (torch.Tensor): attention mask for prompt_embeds, shape (B, L).
            negative_prompt_embeds (torch.Tensor): dense negative prompt embeddings, shape (B, L, D).
            negative_prompt_embeds_mask (torch.Tensor): attention mask for negative_prompt_embeds.
            micro_batch (TensorDict): the full micro-batch, available for architecture-specific
                metadata (e.g. height, width, vae_scale_factor).
            step (int): the current denoising step index.
        """
        pass

    @classmethod
    @abstractmethod
    def forward_and_sample_previous_step(
        cls,
        module: ModelMixin,
        scheduler: SchedulerMixin,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ):
        """Forward the model and sample the previous step.
        Used for RL-algorithms based on reversed-sampling (FlowGRPO, DanceGRPO, etc.).

        Args:
            module (ModelMixin): the diffusion model to be forwarded.
            scheduler (SchedulerMixin): the scheduler used for the diffusion process.
            model_config (DiffusionModelConfig): the configuration of the diffusion model.
            model_inputs (dict[str, torch.Tensor]): the inputs to the diffusion model.
            negative_model_inputs (Optional[dict[str, torch.Tensor]]): the negative inputs for guidance.
            scheduler_inputs (Optional[TensorDict | dict[str, torch.Tensor]]): the extra inputs for the scheduler,
                which may contain the latents and timesteps.
            step (int): the current step in the diffusion process.

        Returns:
            tuple: ``(log_prob, prev_sample_mean, std_dev_t, sqrt_dt)``
        """
        pass

    @classmethod
    def prepare_forward_diffusion_inputs(
        cls,
        module: ModelMixin,
        model_config: DiffusionModelConfig,
        xt: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_embeds_mask: Optional[torch.Tensor],
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, Optional[dict]]:
        """Build inputs for forward-process losses such as DiffusionNFT.
        This packages an already re-noised ``x_t`` constructed from rollout
        ``x_0``; the forward noising itself happens before this hook.

        Model-specific adapters can override this for architecture-specific
        kwargs. The default matches common Diffusers transformer argument names.
        """
        model_inputs = {
            "hidden_states": xt,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "encoder_attention_mask": prompt_embeds_mask,
            "return_dict": False,
        }
        negative_model_inputs = None
        if negative_prompt_embeds is not None:
            negative_model_inputs = {
                **model_inputs,
                "encoder_hidden_states": negative_prompt_embeds,
                "encoder_attention_mask": negative_prompt_embeds_mask,
            }
        return model_inputs, negative_model_inputs

    @classmethod
    def forward_velocity(
        cls,
        module: ModelMixin,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Return a velocity/prediction tensor for forward-process objectives."""
        return module(**model_inputs)[0]


class VllmOmniPipelineBase:
    """Registry base for vllm-omni custom diffusion pipeline classes.

    To register, decorate your custom pipeline class with
    ``@VllmOmniPipelineBase.register("name", algorithm="...")``. The *name* must match the
    ``_class_name`` value in the pipeline's ``model_index.json`` (which is
    auto-detected into ``DiffusionModelConfig.architecture``). The *algorithm*
    must match ``DiffusionModelConfig.algorithm``.

    Example::

        @VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="flow_grpo")
        class QwenImagePipelineWithLogProb(QwenImagePipeline):
            ...
    """

    _registry: dict[tuple[str, str], type] = {}

    @classmethod
    def register(cls, architecture: str, algorithm: str):
        """Class decorator that registers a pipeline for ``(architecture, algorithm)``."""

        def decorator(subclass: type) -> type:
            cls._registry[(architecture, algorithm)] = subclass
            return subclass

        return decorator

    @classmethod
    def get_class(cls, architecture: str, algorithm: str) -> type | None:
        """Return the registered pipeline class for ``(architecture, algorithm)``, or ``None``."""
        return cls._registry.get((architecture, algorithm))

    @classmethod
    def get_pipeline_path(cls, architecture: str, algorithm: str) -> str | None:
        """Return the fully-qualified dotted import path for ``(architecture, algorithm)``, or ``None``."""
        pipeline_cls = cls.get_class(architecture, algorithm)
        if pipeline_cls is None:
            return None
        return f"{pipeline_cls.__module__}.{pipeline_cls.__qualname__}"
