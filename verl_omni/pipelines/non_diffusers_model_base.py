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
from typing import Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = ["NonDiffusersModelBase"]


class NonDiffusersModelBase(nn.Module, ABC):
    """ABC for non-diffusers models used in verl-omni FSDP training.

    Provides LoRA/PEFT adapter lifecycle (required by ``LoRAAdapterMixin``),
    gradient checkpointing with an opt-in guard, FSDP wrapping hints via
    ``_no_split_modules``, and checkpoint persistence.

    Subclasses must implement ``from_pretrained`` and ``forward``, and set
    ``_no_split_modules`` for layer-level FSDP sharding.

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

    # Abstract interface — subclasses must implement these.
    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        model_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> NonDiffusersModelBase:
        """Load a pretrained model from *model_path*."""
        ...

    @abstractmethod
    def forward(self, **kwargs):
        """Forward pass; signature is model-dependent."""
        ...

    # Set in subclasses to enable layer-level FSDP sharding.
    _no_split_modules: list[str] = []

    # Opt in by setting ``_supports_gradient_checkpointing = True`` and
    # wiring ``_checkpointed_call`` into ``forward``.
    _supports_gradient_checkpointing: bool = False
    gradient_checkpointing: bool = False
    _gradient_checkpointing_func: Callable | None = None

    def enable_gradient_checkpointing(
        self,
        gradient_checkpointing_func: Callable | None = None,
    ) -> None:
        """Enable gradient checkpointing.

        Raises ValueError if ``_supports_gradient_checkpointing`` is False.
        """
        if not self._supports_gradient_checkpointing:
            raise ValueError(
                f"{type(self).__name__} does not support gradient "
                f"checkpointing.  Set ``_supports_gradient_checkpointing = True`` "
                f"and use ``_checkpointed_call`` in your ``forward`` method."
            )
        if gradient_checkpointing_func is not None:
            self._gradient_checkpointing_func = gradient_checkpointing_func
        self.gradient_checkpointing = True

    def _checkpointed_call(self, fn, *args, **ckpt_kwargs):
        """Call *fn*, wrapping with checkpoint when enabled and training."""
        if not self.gradient_checkpointing or not self.training:
            return fn(*args)

        ckpt_kwargs.setdefault("use_reentrant", False)
        ckpt_func = self._gradient_checkpointing_func
        if ckpt_func is None:
            ckpt_func = torch.utils.checkpoint.checkpoint
        return ckpt_func(fn, *args, **ckpt_kwargs)

    # LoRA / PEFT adapter lifecycle

    def add_adapter(self, adapter_config, adapter_name: str = "default") -> None:
        """Inject a PEFT LoRA adapter and store its config."""
        from peft import inject_adapter_in_model

        if not hasattr(self, "peft_config"):
            self.peft_config: dict[str, object] = {}
        self.peft_config[adapter_name] = adapter_config
        inject_adapter_in_model(adapter_config, self, adapter_name)

    def load_lora_adapter(self, adapter_path: str, adapter_name: str = "default") -> None:
        """Load a pre-trained LoRA adapter from *adapter_path*.

        Reads ``adapter_config.json`` and ``adapter_model.safetensors``,
        injects the adapter, then copies weights in.  Mismatched keys are
        warned about but do not raise.
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
        """Activate a named PEFT adapter across all submodules."""
        for module in self.modules():
            if module is self:
                continue
            set_adapter_fn = getattr(module, "set_adapter", None)
            if callable(set_adapter_fn):
                set_adapter_fn(adapter_name)

    def disable_adapters(self) -> None:
        """Disable all PEFT adapters (base weights only)."""
        for module in self.modules():
            if module is self:
                continue
            disable_adapters_fn = getattr(module, "disable_adapters", None)
            if callable(disable_adapters_fn):
                disable_adapters_fn()

    def enable_adapters(self) -> None:
        """Re-enable all PEFT adapters after ``disable_adapters``."""
        for module in self.modules():
            if module is self:
                continue
            enable_adapters_fn = getattr(module, "enable_adapters", None)
            if callable(enable_adapters_fn):
                enable_adapters_fn()

    # Checkpoint persistence

    def _save_config(self, save_directory: str) -> None:
        """Save ``self.config`` to *save_directory*, if supported."""
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
        """Save config and weights to *save_directory*."""
        from safetensors.torch import save_file as safetensors_save_file

        os.makedirs(save_directory, exist_ok=True)
        self._save_config(save_directory)

        if safe_serialization:
            weights_path = os.path.join(save_directory, "model.safetensors")
            state_dict = self.state_dict()
            clean_state = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
            safetensors_save_file(clean_state, weights_path)
        else:
            weights_path = os.path.join(save_directory, "pytorch_model.bin")
            torch.save(self.state_dict(), weights_path)

        logger.info("Model saved to %s", save_directory)
