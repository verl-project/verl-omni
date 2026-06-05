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

"""Base class for non-diffusers model modules used in verl-omni training.

Non-diffusers models are standalone `nn.Module` implementations that do
*not* inherit from `diffusers.ModelMixin` and are *not* loaded through
`diffusers.AutoModel.from_pretrained`.  They manage their own architecture,
configuration format, weight-loading logic, and (optionally) internal text
processing (token embedding inside the forward pass).
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from safetensors.torch import save_file as safetensors_save_file

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

__all__ = ["NonDiffusersModelBase"]


class NonDiffusersModelBase(nn.Module, ABC):
    """ABC for non-diffusers diffusion / flow-matching model modules.

    Provides the infrastructure every non-diffusers model needs to
    participate in verl-omni's FSDP training loop:

    * LoRA / PEFT adapter lifecycle (required by ``LoRAAdapterMixin``).
    * Gradient checkpointing with an opt-in guard — raises if the subclass
      has not wired the feature via `_checkpointed_call`.
    * FSDP wrapping hints via `_no_split_modules`.
    * Config persistence via `save_pretrained` and `_save_config`.

    Subclasses must implement `from_pretrained` and `forward`, and should
    set `_no_split_modules` for layer-level FSDP sharding.  To enable
    gradient checkpointing set `_supports_gradient_checkpointing = True`
    and wrap each layer call with `_checkpointed_call`.

    Example::

        class MyModel(NonDiffusersModelBase):
            _no_split_modules = ["MyTransformerLayer"]
            _supports_gradient_checkpointing = True

            def forward(self, h, t, **kwargs):
                for layer in self.layers:
                    h = self._checkpointed_call(layer, h, t)
                return h

            @classmethod
            def from_pretrained(cls, model_path, torch_dtype=torch.bfloat16):
                ...
    """

    # ------------------------------------------------------------------
    #  Class-level FSDP configuration
    # ------------------------------------------------------------------

    #: FSDP leaf-module class names used by `get_fsdp_wrap_policy`.
    #: Set in subclasses to enable layer-level sharding, e.g.
    #: ``["MyTransformerLayer"]``.  When empty the engine falls back to
    #: per-leaf LoRA wrapping or size-based policies.
    _no_split_modules: list[str] = []

    # ------------------------------------------------------------------
    #  Gradient checkpointing
    # ------------------------------------------------------------------

    #: Opt-in flag — set to True in subclasses that wire `_checkpointed_call`
    #: into their `forward` loop.  `enable_gradient_checkpointing` raises
    #: ValueError if this is still False, guarding against silent no-ops.
    _supports_gradient_checkpointing: bool = False

    #: Runtime toggle set by `enable_gradient_checkpointing` /
    #: `disable_gradient_checkpointing`.
    gradient_checkpointing: bool = False

    #: Custom checkpoint function; None falls back to
    #: `torch.utils.checkpoint.checkpoint`.
    _gradient_checkpointing_func: object | None = None

    @property
    def is_gradient_checkpointing(self) -> bool:
        """Return whether gradient checkpointing is currently active."""
        return self.gradient_checkpointing

    def enable_gradient_checkpointing(
        self,
        gradient_checkpointing_func: object | None = None,
    ) -> None:
        """Enable gradient checkpointing.

        Raises ValueError if `_supports_gradient_checkpointing` is False —
        the subclass must opt-in and wire `_checkpointed_call` into its
        `forward` before this can be called.

        Args:
            gradient_checkpointing_func: Optional custom checkpoint
                function.  When None, `_checkpointed_call` uses
                `torch.utils.checkpoint.checkpoint` with
                ``use_reentrant=False``.
        """
        if not self._supports_gradient_checkpointing:
            raise ValueError(
                f"{type(self).__name__} does not support gradient "
                f"checkpointing.  To enable it, set "
                f"`_supports_gradient_checkpointing = True` in the class "
                f"definition and use `_checkpointed_call` to wrap each "
                f"layer call in your `forward` method."
            )

        if gradient_checkpointing_func is not None:
            self._gradient_checkpointing_func = gradient_checkpointing_func
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing.

        After this call `_checkpointed_call` passes straight through to the
        layer without `torch.utils.checkpoint.checkpoint` wrapping.
        """
        self.gradient_checkpointing = False

    def _checkpointed_call(self, fn, *args, **ckpt_kwargs):
        """Call *fn*, wrapping with gradient checkpointing when enabled.

        Subclasses should use this as the single entry point for
        checkpointed layer iteration in `forward`::

            for layer in self.layers:
                h = self._checkpointed_call(layer, h, t)

        For layer calls needing a custom closure (e.g. extra non-tensor
        arguments captured from the enclosing scope)::

            for layer in self.layers:
                def _fn(h, t):
                    return layer(h, t, extra=flag)
                h = self._checkpointed_call(_fn, h, t)

        Args:
            fn (callable): The callable to invoke (layer module or closure).
            *args: Positional arguments forwarded to *fn* (must be tensors).
            **ckpt_kwargs: Forwarded to `torch.utils.checkpoint.checkpoint`
                (e.g. ``use_reentrant``).  Not passed to *fn*.

        Returns:
            Tensor: The output of ``fn(*args)``, checkpointed if applicable.
        """
        if not self.gradient_checkpointing or not self.training:
            return fn(*args)

        ckpt_kwargs.setdefault("use_reentrant", False)
        ckpt_func = self._gradient_checkpointing_func
        if ckpt_func is None:
            ckpt_func = torch.utils.checkpoint.checkpoint
        return ckpt_func(fn, *args, **ckpt_kwargs)

    # ------------------------------------------------------------------
    #  LoRA / PEFT adapter lifecycle
    # ------------------------------------------------------------------

    def add_adapter(self, adapter_config, adapter_name: str = "default") -> None:
        """Inject a PEFT LoRA adapter via `peft.inject_adapter_in_model`.

        The config is stored in `self.peft_config` for later reference.

        Args:
            adapter_config: A PEFT config object (e.g. `LoraConfig`).
            adapter_name (str): Name for this adapter.  Defaults to
                ``"default"``.
        """
        from peft import inject_adapter_in_model

        if not hasattr(self, "peft_config"):
            self.peft_config: dict[str, object] = {}
        self.peft_config[adapter_name] = adapter_config
        inject_adapter_in_model(adapter_config, self, adapter_name)

    def load_lora_adapter(self, adapter_path: str, adapter_name: str = "default") -> None:
        """Load a pre-trained LoRA adapter from *adapter_path*.

        Reads `adapter_config.json` and `adapter_model.safetensors`,
        injects a new adapter via `add_adapter`, then copies checkpoint
        weights into the newly created parameters.  Mismatched keys are
        logged as warnings — missing keys keep their initial values,
        unexpected keys are ignored.

        Args:
            adapter_path (str): Directory containing the saved adapter.
            adapter_name (str): Name to register the adapter under.
        """
        from peft import LoraConfig, get_peft_model_state_dict
        from safetensors.torch import load_file as safetensors_load_file

        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
        adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")

        if not os.path.isfile(adapter_config_path):
            raise FileNotFoundError(f"LoRA adapter config not found at {adapter_config_path}")
        if not os.path.isfile(adapter_weights_path):
            raise FileNotFoundError(f"LoRA adapter weights not found at {adapter_weights_path}")

        with open(adapter_config_path) as f:
            lora_config = LoraConfig.from_dict(json.load(f))

        self.add_adapter(lora_config, adapter_name=adapter_name)

        # Load adapter weights into the newly created parameters.
        adapter_state_dict = safetensors_load_file(adapter_weights_path)
        current_state = get_peft_model_state_dict(self, adapter_name=adapter_name)

        # Only load keys that exist in both (defensive against mismatches).
        loadable_keys = set(adapter_state_dict.keys()) & set(current_state.keys())
        missing_load = set(current_state.keys()) - set(adapter_state_dict.keys())
        unexpected_load = set(adapter_state_dict.keys()) - set(current_state.keys())

        if missing_load:
            logger.warning(
                "LoRA adapter %r: %d keys in model but not in checkpoint. They will keep their initial values.",
                adapter_name,
                len(missing_load),
            )
        if unexpected_load:
            logger.warning(
                "LoRA adapter %r: %d keys in checkpoint but not in model. They will be ignored.",
                adapter_name,
                len(unexpected_load),
            )

        for key in loadable_keys:
            current_state[key].copy_(adapter_state_dict[key])

    def set_adapter(self, adapter_name: str) -> None:
        """Activate a named PEFT adapter across all submodules.

        Args:
            adapter_name (str): Name of the adapter to activate.
        """
        for module in self.modules():
            if module is self:
                continue
            set_adapter_fn = getattr(module, "set_adapter", None)
            if callable(set_adapter_fn):
                set_adapter_fn(adapter_name)

    def disable_adapters(self) -> None:
        """Disable all PEFT adapters so only base weights are used."""
        for module in self.modules():
            if module is self:
                continue
            disable_adapters_fn = getattr(module, "disable_adapters", None)
            if callable(disable_adapters_fn):
                disable_adapters_fn()

    def enable_adapters(self) -> None:
        """Re-enable all PEFT adapters after `disable_adapters`."""
        for module in self.modules():
            if module is self:
                continue
            enable_adapters_fn = getattr(module, "enable_adapters", None)
            if callable(enable_adapters_fn):
                enable_adapters_fn()

    @contextmanager
    def disable_adapter(self) -> Iterator[None]:
        """Context manager that temporarily disables all PEFT adapters.

        Usage::

            with model.disable_adapter():
                output = model(...)
        """
        try:
            self.disable_adapters()
            yield
        finally:
            self.enable_adapters()

    # ------------------------------------------------------------------
    #  Checkpoint persistence
    # ------------------------------------------------------------------

    def _save_config(self, save_directory: str) -> None:
        """Save the model configuration to *save_directory*.

        Calls `self.config.save_pretrained(save_directory)` if the config
        object exposes that method.  Override if your config uses a
        different serialisation format.

        Args:
            save_directory (str): Target directory (created if needed).
        """
        if hasattr(self.config, "save_pretrained"):
            self.config.save_pretrained(save_directory)
        else:
            logger.warning(
                "Model config has no save_pretrained method; "
                "skipping config save.  Override _save_config in your subclass."
            )

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        """Save model config and weights to *save_directory*.

        Writes config via `_save_config` and weights via safetensors
        (`model.safetensors`) by default.  Pass
        ``safe_serialization=False`` to use `torch.save` instead.

        Args:
            save_directory (str): Target directory (created if needed).
            safe_serialization (bool): If True (default), use safetensors.
        """
        os.makedirs(save_directory, exist_ok=True)
        self._save_config(save_directory)

        if safe_serialization:
            weights_path = os.path.join(save_directory, "model.safetensors")
            state_dict = self.state_dict()
            # Convert non-tensor entries that may have leaked into state_dict.
            clean_state = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
            safetensors_save_file(clean_state, weights_path)
        else:
            weights_path = os.path.join(save_directory, "pytorch_model.bin")
            torch.save(self.state_dict(), weights_path)

        logger.info("Model saved to %s", save_directory)

    # ------------------------------------------------------------------
    #  Abstract interface
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        model_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> NonDiffusersModelBase:
        """Load a pretrained model from *model_path*.

        Subclasses must parse config, instantiate the model, load weights,
        and cast to *torch_dtype*.

        Args:
            model_path (str): Directory containing config and weights.
            torch_dtype (torch.dtype): Target dtype (default bfloat16).
            **kwargs: Additional model-specific arguments.

        Returns:
            NonDiffusersModelBase: An instance of the subclass with
                loaded weights.
        """
        ...

    @abstractmethod
    def forward(self, **kwargs):
        """Forward pass of the model.

        Subclasses must implement the architecture-specific computation.
        The exact signature is model-dependent; the training adapter
        constructs inputs via `DiffusionModelBase.prepare_model_inputs`.
        """
        ...
