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

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from diffusers import ModelMixin, SchedulerMixin
from tensordict import TensorDict

from verl_omni.workers.config import DiffusionModelConfig

logger = logging.getLogger(__name__)


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
            if architecture == "QwenImagePipeline":
                logger.info(
                    "Applying monkey-patch for QwenImageTransformer2DModel Ulysses SP "
                    "This workaround will be removed once we upgrade to a diffusers release that "
                    "includes the upstream fix."
                )
                from verl_omni.models.diffusers.qwen_image import apply_qwen_image_ulysses_mask_fix

                apply_qwen_image_ulysses_mask_fix()
            return cls._registry[key]
        except KeyError:
            registered = sorted(cls._registry.keys())
            raise NotImplementedError(
                f"No diffusion model registered for (architecture={architecture!r}, "
                f"algorithm={algorithm!r}). Registered: {registered}. "
                f"Set ``external_lib`` in DiffusionModelConfig to load your implementation."
            ) from None

    @classmethod
    def build_module(cls, model_config: DiffusionModelConfig, torch_dtype: torch.dtype) -> Optional[torch.nn.Module]:
        """Load the model without ``diffusers.AutoModel``.

        Return ``None`` to use the default ``AutoModel`` path.
        Override this for models that diffusers cannot load.
        """
        return None

    @classmethod
    def configure_train_mode(cls, module: torch.nn.Module) -> None:
        """Hook called after ``module.train()`` for architecture-specific overrides."""
        return

    @classmethod
    def configure_trainable_params(
        cls,
        module: torch.nn.Module,
        model_config: DiffusionModelConfig,
    ) -> None:
        """Hook called after module build to set ``requires_grad`` on trainable params.

        Args:
            module: The loaded model module (pre-FSDP).
            model_config: The ``DiffusionModelConfig``.
        """
        return

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
    ) -> tuple[dict, Optional[dict]]:
        """Build architecture-specific inputs for a model forward.
        For reverse-trajectory algorithms, ``latents`` and ``timesteps`` usually
        contain the full rollout trajectory and ``step`` selects the current
        slice. For forward-process objectives, callers may pass an already
        selected/noised latent and timestep directly.
        The caller is responsible for universal pre-processing (common tensor extraction
        and nested-embed unpadding) before invoking this method.

        Args:
            module (ModelMixin): the diffusion transformer module.
            model_config (DiffusionModelConfig): the configuration of the diffusion model.
            latents (torch.Tensor): latent tensor from the micro-batch; either a full trajectory
                of shape (B, T, ...) or a selected/noised latent of shape (B, ...).
            timesteps (torch.Tensor): timestep tensor from the micro-batch; either a full
                trajectory of shape (B, T) or a selected timestep of shape (B,).
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
    def forward(
        cls,
        module: ModelMixin,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Run a single model prediction.
        Used both for forward-process objectives (noising clean latents ``x0 -> xt``
        then optimizing predictions directly) and as the prediction step inside
        reverse-sampling algorithms (FlowGRPO et al.). Model adapters only need to
        override this when prediction requires extra handling such as CFG, negative
        inputs, or output conversion.
        """
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


class OmniModelBase(ABC):
    """Plug-in base class for omni model training adapters."""

    _registry: dict[tuple[str, str], type["OmniModelBase"]] = {}

    @classmethod
    def register(cls, architecture: str, stage: str = "thinker"):
        """Class decorator that registers a subclass for ``(architecture, stage)``."""

        def decorator(subclass: type["OmniModelBase"]) -> type["OmniModelBase"]:
            cls._registry[(architecture, stage)] = subclass
            return subclass

        return decorator

    @classmethod
    def get_class(cls, model_config) -> type["OmniModelBase"]:
        """Return the registered subclass for ``(architecture, stage)``."""
        architecture = model_config.get("architecture", None)
        stage = model_config.get("model_stage", "thinker")
        key = (architecture, stage)

        if key not in cls._registry and model_config.get("external_lib", None) is not None:
            from verl.utils.import_utils import import_external_libs

            import_external_libs(model_config.external_lib)

        try:
            return cls._registry[key]
        except KeyError:
            registered = sorted(cls._registry.keys())
            raise NotImplementedError(
                f"No omni model registered for (architecture={architecture!r}, stage={stage!r}). "
                f"Registered: {registered}. Set ``external_lib`` to load your implementation."
            ) from None

    @classmethod
    @abstractmethod
    def get_model_architecture(cls) -> type:
        """Return the Hugging Face model class to register for training."""
        pass

    @classmethod
    @abstractmethod
    def get_model_loading_kwargs(cls, model_config) -> dict[str, Any]:
        """Return architecture-specific model loading kwargs."""
        pass

    @classmethod
    @abstractmethod
    def get_strip_modules(cls, model_config) -> list[str]:
        """Return submodule prefixes to strip before wrapping the trainable stage."""
        pass

    @classmethod
    @abstractmethod
    def configure_processor(cls, model_path: str, model_config):
        """Load and configure the multimodal processor for this omni model."""
        pass

    @classmethod
    @abstractmethod
    def configure_tokenizer(cls, model_path: str, model_config):
        """Load and configure the tokenizer for this omni model."""
        pass

    @classmethod
    def get_frozen_parameter_patterns(cls, model_config) -> list[str]:
        """Return regex patterns for parameters to freeze during training."""
        return []

    @classmethod
    def configure_model(cls, module: torch.nn.Module, model_config) -> torch.nn.Module:
        """Apply generic stage stripping and parameter freezing."""
        import re

        for submod_name in cls.get_strip_modules(model_config):
            if hasattr(module, submod_name):
                delattr(module, submod_name)

        frozen_patterns = [re.compile(pattern) for pattern in cls.get_frozen_parameter_patterns(model_config)]
        if frozen_patterns:
            for name, parameter in module.named_parameters():
                if any(pattern.search(name) for pattern in frozen_patterns):
                    parameter.requires_grad = False
        return module

    @classmethod
    def apply_model_patches(cls) -> None:
        """Install import-time compatibility patches, if the adapter needs any."""
        return

    @classmethod
    def prepare_training_inputs(cls, module: torch.nn.Module, model_config, micro_batch: TensorDict) -> dict[str, Any]:
        """Build training forward kwargs from a micro-batch."""
        return dict(micro_batch)


class OmniRolloutPipelineBase(ABC):
    """Registry base for vLLM-Omni rollout pipeline topologies."""

    _registry: dict[str, type["OmniRolloutPipelineBase"]] = {}

    @classmethod
    def register(cls, model_type: str):
        """Class decorator that registers a subclass for ``model_type``."""

        def decorator(subclass: type["OmniRolloutPipelineBase"]) -> type["OmniRolloutPipelineBase"]:
            cls._registry[model_type] = subclass
            return subclass

        return decorator

    @classmethod
    def get_class(cls, model_type: str) -> type["OmniRolloutPipelineBase"] | None:
        """Return the registered rollout adapter for ``model_type``."""
        return cls._registry.get(model_type)

    @classmethod
    @abstractmethod
    def get_stage_config(cls, pipeline_mode: str = "thinker_only", num_gpus: int = 8) -> list[dict[str, Any]]:
        """Generate a vLLM-Omni stage config list."""
        pass

    @classmethod
    @abstractmethod
    def get_deploy_config(cls, pipeline_mode: str = "thinker_only", num_gpus: int = 8) -> dict[str, Any]:
        """Generate a full vLLM-Omni deploy config."""
        pass

    @classmethod
    @abstractmethod
    def get_pipeline_model_type(cls) -> str:
        """Return the vLLM-Omni pipeline model type."""
        pass
