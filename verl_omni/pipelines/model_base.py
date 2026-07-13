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
from verl.utils import tensordict_utils as tu

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

        if architecture == "QwenImagePipeline":
            logger.info(
                "Applying monkey-patch for QwenImageTransformer2DModel Ulysses SP "
                "This workaround will be removed once we upgrade to a diffusers release that "
                "includes the upstream fix."
            )
            from verl_omni.models.diffusers.qwen_image import apply_qwen_image_ulysses_mask_fix

            apply_qwen_image_ulysses_mask_fix()
        return cls.get_class_by_name(architecture, algorithm, model_config.external_lib)

    @classmethod
    def get_class_by_name(
        cls,
        architecture: str,
        algorithm: str,
        external_lib: Optional[str] = None,
    ) -> type["DiffusionModelBase"]:
        """Resolve an adapter before a full ``DiffusionModelConfig`` exists."""
        key = (architecture, algorithm)
        if external_lib is not None:
            from verl.utils.import_utils import import_external_libs

            import_external_libs(external_lib)
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
    def prepare_processor_files(cls, model_path: str) -> Optional[str]:
        """Prepare model-specific processor files before ``hf_processor()`` loads them.

        Override this when a model ships a ``processor`` directory that needs
        adapter-owned config fixes before Hugging Face can load it. Return an
        alternate processor path when the model directory should not be
        modified in place.
        """
        return None

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

        When ``_target_seq_len`` is present in ``model_inputs`` (set by
        :meth:`DiffusionI2IModelBase.inject_condition`), it is popped before
        the forward call and used to slice the output back to the noise
        segment, dropping the condition-image token predictions.
        """
        model_inputs = dict(model_inputs)
        target_seq_len = model_inputs.pop("_target_seq_len", None)
        noise_pred = module(**model_inputs)[0]
        if target_seq_len is not None:
            if noise_pred.shape[1] < target_seq_len:
                raise ValueError(
                    f"forward: model output seq_len ({noise_pred.shape[1]}) < "
                    f"target_seq_len ({target_seq_len}). The condition concat may "
                    f"have been dropped or the model truncated the output."
                )
            noise_pred = noise_pred[:, :target_seq_len]
        return noise_pred


class DiffusionI2IModelBase(DiffusionModelBase):
    """Base class for image-conditioned diffusion model training helpers.

    Inherits all T2I logic from :class:`DiffusionModelBase`. Adds a two-step
    condition injection hook:

    1. ``prepare_condition`` extracts condition tensors from ``micro_batch``.
    2. ``inject_condition`` merges condition tensors into ``model_inputs``.

    The training dispatcher requires I2I adapters to return a non-empty
    condition. ``inject_condition`` itself remains a no-op for direct callers
    that pass ``None``.

    The default ``inject_condition`` implements a common concat-crop pattern:
    concatenate ``image_latents`` onto ``hidden_states``
    along the token dimension and set ``_target_seq_len`` so that
    :meth:`DiffusionModelBase.forward` slices the prediction back to the
    noise segment. Models with non-concat conditioning (Wan I2V, LTX2 I2AV)
    override ``inject_condition``.
    """

    @staticmethod
    def unwrap_i2i_metadata(value: Any) -> Any:
        """Convert TensorDict non-tensor wrappers back to plain Python metadata."""
        if hasattr(value, "tolist"):
            return value.tolist()
        return value

    @classmethod
    def get_i2i_metadata(cls, micro_batch: TensorDict, key: str, default: Any = None) -> Any:
        """Read and unwrap I2I metadata carried through ``DataProto.non_tensor_batch``."""
        value = tu.get_non_tensor_data(micro_batch, key, default=default)
        return cls.unwrap_i2i_metadata(value)

    @classmethod
    def get_i2i_scalar_metadata(cls, micro_batch: TensorDict, key: str, default: Any = None) -> Any:
        """Read scalar I2I metadata, requiring it to be constant across the micro-batch."""
        value = cls.get_i2i_metadata(micro_batch, key, default=default)
        if value is None:
            return default

        def _to_scalar(item):
            if hasattr(item, "item") and not isinstance(item, str | bytes):
                return item.item()
            return item

        if isinstance(value, list | tuple):
            if len(value) == 0:
                return default
            values = [_to_scalar(item) for item in value]
            first = values[0]
            if any(item != first for item in values[1:]):
                raise ValueError(f"I2I metadata {key!r} differs across the micro-batch: {values!r}")
            return first

        return _to_scalar(value)

    @classmethod
    def prepare_condition(
        cls,
        micro_batch: TensorDict,
        latents: torch.Tensor,
        step: int,
    ) -> Optional[dict]:
        """Extract condition fields from ``micro_batch``.

        T2I default returns ``None``. I2I adapters override this to pull
        model-specific condition tensors from the micro-batch and return them
        under the keys that :meth:`inject_condition` expects. The default
        concat-crop implementation requires ``image_latents``. Adapters that
        need position metadata or non-concat conditioning must override
        :meth:`inject_condition`.

        Note: the *micro-batch* keys carrying condition tensors must not
        collide with keys the MFU FLOPs counter interprets as the denoised
        latent (``image_latents``, ``latents_clean``, ``all_latents``,
        ``audio_latents``). Use a distinct key such as
        ``condition_image_latents`` on the micro-batch, then map it to the
        ``image_latents`` slot in the returned condition dict.

        Args:
            micro_batch (TensorDict): the full micro-batch.
            latents (torch.Tensor): the latent tensor for the current step.
            step (int): the current denoising step index.

        Returns:
            Optional[dict]: a flat dict of condition tensors, or ``None``
            when no condition is present (T2I degenerate path).
        """
        return None

    @classmethod
    def inject_condition(
        cls,
        model_inputs: dict,
        negative_model_inputs: Optional[dict],
        condition: Optional[dict],
    ) -> tuple[dict, Optional[dict]]:
        """Merge condition tensors into ``model_inputs``.

        Default implementation: concatenate ``image_latents`` onto
        ``hidden_states`` along the token dimension and set
        ``_target_seq_len`` so that
        :meth:`DiffusionModelBase.forward` slices the prediction back.

        When ``condition`` is ``None`` or empty, this is a no-op (T2I
        degenerate path). Models with non-concat conditioning (Wan I2V,
        LTX2 I2AV) override this method.

        Sequence-parallel note: Ulysses SP splits ``hidden_states`` along the
        token dimension with ``chunk(sp_size)``, so the concatenated
        ``noise + condition`` token count must be divisible by ``sp_size``.
        When ``condition`` carries ``sp_size`` (> 1), this method asserts the
        combined sequence length is aligned and raises otherwise, so the
        condition latent resolution must be chosen upstream (rollout-side VAE
        encode) rather than zero-padded here (padding would inject unmasked
        condition tokens).
        """
        if not condition:
            return model_inputs, negative_model_inputs

        image_latents = condition.get("image_latents")
        if image_latents is None:
            raise ValueError("inject_condition requires condition['image_latents']")

        # Guard: "image_latents" is reserved by the MFU FLOPs counter.
        if "image_latents" in model_inputs:
            raise ValueError(
                "inject_condition: 'image_latents' found in model_inputs; "
                "this key is reserved by the MFU FLOPs counter for the denoised "
                "latent. The rollout adapter likely output 'image_latents' instead "
                "of 'condition_image_latents'. Check the rollout adapter's "
                "custom_output keys."
            )

        hidden_states = model_inputs["hidden_states"]
        if image_latents.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "inject_condition: condition image_latents batch size "
                f"({image_latents.shape[0]}) does not match hidden_states batch size "
                f"({hidden_states.shape[0]})."
            )

        if image_latents.dim() != 3:
            raise ValueError(
                f"inject_condition: condition image_latents must be 3-D "
                f"(batch, seq, dim), got shape {image_latents.shape}"
            )

        # SP alignment: combined seq_len must be divisible by sp_size.
        target_seq_len = hidden_states.shape[1]
        combined_seq_len = target_seq_len + image_latents.shape[1]
        sp_size = condition.get("sp_size")
        if isinstance(sp_size, int) and sp_size > 1 and combined_seq_len % sp_size != 0:
            raise ValueError(
                "inject_condition: combined noise+condition token length "
                f"({combined_seq_len} = {target_seq_len} + {image_latents.shape[1]}) "
                f"is not divisible by sequence-parallel size ({sp_size}). "
                "Choose a condition image resolution whose packed latent length keeps "
                "the combined sequence divisible by sp_size (align at rollout-side VAE "
                "encode); do not zero-pad the condition here."
            )

        for inputs in (model_inputs, negative_model_inputs):
            if inputs is None:
                continue
            inputs["hidden_states"] = torch.cat(
                [
                    inputs["hidden_states"],
                    image_latents.to(
                        device=inputs["hidden_states"].device,
                        dtype=inputs["hidden_states"].dtype,
                    ),
                ],
                dim=1,
            )
            inputs["_target_seq_len"] = target_seq_len

        return model_inputs, negative_model_inputs


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
