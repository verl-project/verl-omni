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
"""Reusable PEFT/LoRA adapter lifecycle helpers for training engines."""

import logging
from contextlib import contextmanager, nullcontext

import torch
from peft import LoraConfig
from verl.utils.py_functional import convert_to_regular_types

from verl_omni.utils.config import resolve_lora_config

logger = logging.getLogger(__name__)

# peft stores adapter weights in per-adapter containers on each tuner layer.
_ADAPTER_MODULE_CONTAINERS = ("lora_A", "lora_B", "lora_magnitude_vector")
_ADAPTER_PARAMETER_CONTAINERS = ("lora_embedding_A", "lora_embedding_B")


def cast_adapter_params_to_base(module: torch.nn.Module) -> None:
    """Cast every PEFT adapter parameter to its base layer's device and dtype.

    peft applies the base layer's device and dtype only when it first wraps a
    module into a tuner layer; adapters added to an already-wrapped layer are
    created on CPU in the default dtype. On modules built on the meta device
    that mixes dtypes inside FSDP wrapping units, so normalize before sharding.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer

    for submodule in module.modules():
        if not isinstance(submodule, BaseTunerLayer):
            continue
        base_param = next(submodule.get_base_layer().parameters(), None)
        if base_param is None:
            continue
        for attr in _ADAPTER_MODULE_CONTAINERS:
            container = getattr(submodule, attr, None)
            if isinstance(container, torch.nn.ModuleDict):
                for adapter_name, layer in container.items():
                    container[adapter_name] = layer.to(device=base_param.device, dtype=base_param.dtype)
        for attr in _ADAPTER_PARAMETER_CONTAINERS:
            container = getattr(submodule, attr, None)
            if isinstance(container, torch.nn.ParameterDict):
                for adapter_name, param in container.items():
                    container[adapter_name] = torch.nn.Parameter(
                        param.to(device=base_param.device, dtype=base_param.dtype),
                        requires_grad=param.requires_grad,
                    )


def apply_lora_dtype(module: torch.nn.Module, target_dtype: torch.dtype) -> None:
    """Cast every LoRA adapter parameter to ``target_dtype`` and pin adapter inputs.

    Covers inactive adapters too: peft marks only the active adapter trainable,
    so a requires_grad-filtered cast leaves "old" policy adapters in the base
    dtype and mixes dtypes inside FSDP wrapping units and optimizer states.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer

    for name, param in module.named_parameters():
        if ".lora_" in name or name.startswith("lora_"):
            param.data = param.data.to(target_dtype)
    for submodule in module.modules():
        if isinstance(submodule, BaseTunerLayer):
            submodule.cast_input_dtype_enabled = False


class LoRAAdapterMixin:
    """Backend-agnostic helpers for named PEFT/LoRA policy adapters."""

    def _build_lora_module(self, module):
        lora_settings = resolve_lora_config(self.model_config)
        lora_adapter_path = lora_settings.adapter_path
        policy_state_adapters = lora_settings.adapters
        extra_adapters = tuple(adapter for adapter in policy_state_adapters if adapter not in ("default", "reference"))
        primary_adapter = policy_state_adapters[0]
        if lora_adapter_path is not None:
            from verl.utils.fs import copy_to_local

            print(f"Loading pre-trained LoRA adapter to from: {lora_adapter_path}")
            local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.model_config.use_shm)

            # diffusers auto-names an unnamed first adapter "default_0"; name it explicitly.
            module.load_lora_adapter(local_adapter_path, adapter_name=primary_adapter)
            peft_config = getattr(module, "peft_config", {}).get(primary_adapter, None)
            for adapter_name in extra_adapters:
                if peft_config is not None and adapter_name not in getattr(module, "peft_config", {}):
                    module.add_adapter(peft_config, adapter_name=adapter_name)
        else:
            lora_config = {
                "r": lora_settings.rank,
                "lora_alpha": lora_settings.alpha,
                "init_lora_weights": self.model_config.lora_init_weights,
                "target_modules": convert_to_regular_types(self.model_config.target_modules),
                "target_parameters": convert_to_regular_types(self.model_config.target_parameters),
                "exclude_modules": convert_to_regular_types(self.model_config.exclude_modules),
                "bias": "none",
            }
            module.add_adapter(LoraConfig(**lora_config), adapter_name="default")
            for adapter_name in extra_adapters:
                module.add_adapter(LoraConfig(**lora_config), adapter_name=adapter_name)

        # Adapters beyond the first skip peft's base-dtype cast; align them before sharding.
        cast_adapter_params_to_base(module)

        if "default" in policy_state_adapters and hasattr(module, "set_adapter"):
            module.set_adapter("default")

        lora_dtype = getattr(self.model_config, "lora_dtype", None)
        if lora_dtype is not None:
            from verl.utils.torch_dtypes import PrecisionType

            apply_lora_dtype(module, PrecisionType.to_dtype(lora_dtype))

        return module

    @contextmanager
    def _adapter_state_context(self):
        """Open writable adapter parameter access (FSDP summon when applicable)."""
        from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
        from verl.utils.memory_utils import aggressive_empty_cache

        from verl_omni.utils.fsdp_utils import fsdp_summon_full_params

        is_fsdp_module = fsdp_version(self.module) in (1, 2)
        is_offload_param = getattr(self, "_is_offload_param", False)
        origin_module_device = next(self.module.parameters()).device.type
        uses_cpu_offload_policy = getattr(self, "_uses_fsdp2_cpu_offload_policy", False)
        if is_fsdp_module and (is_offload_param or origin_module_device == "cpu") and not uses_cpu_offload_policy:
            load_fsdp_model_to_gpu(self.module)

        ctx = fsdp_summon_full_params(self.module, writeback=True) if is_fsdp_module else nullcontext()
        try:
            with ctx:
                try:
                    yield
                finally:
                    self._set_adapter("default")
        finally:
            if is_offload_param:
                offload_fsdp_model_to_cpu(self.module)
                aggressive_empty_cache(force_sync=True)

    def _set_adapter(self, name: str):
        module = getattr(self.module, "_fsdp_wrapped_module", self.module)
        if not hasattr(module, "set_adapter"):
            raise AttributeError(f"Module does not support set_adapter({name!r})")
        module.set_adapter(name)

    @contextmanager
    def use_adapter(self, name: str):
        """Temporarily select a named PEFT adapter.

        ``"reference"`` is a logical policy state (see ``policy_state_adapters``)
        that runs with all LoRA adapters disabled, not a registered PEFT adapter.
        """
        if name == "reference":
            with self.disable_adapter():
                yield
        else:
            self._set_adapter(name)
            try:
                yield
            finally:
                self._set_adapter("default")

    def _active_adapter_trainable_params(self, adapter_name: str) -> list[torch.nn.Parameter]:
        peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
        if not hasattr(peft_model, "set_adapter"):
            raise AttributeError("Module does not support PEFT adapter selection.")
        peft_model.set_adapter(adapter_name)
        return list(filter(lambda param: param.requires_grad, peft_model.parameters()))

    def copy_adapter(self, source: str = "default", target: str = "old") -> None:
        """Copy LoRA state between named policy adapters."""
        with self._adapter_state_context(), torch.no_grad():
            source_params = self._active_adapter_trainable_params(source)
            target_params = self._active_adapter_trainable_params(target)
            if len(source_params) != len(target_params) or not source_params:
                raise ValueError(
                    f"Adapter copy {source!r} -> {target!r} found mismatched params: "
                    f"{len(source_params)} vs {len(target_params)}"
                )
            for source_param, target_param in zip(source_params, target_params, strict=True):
                target_param.copy_(source_param)

    def ema_update_adapter(self, source: str = "default", target: str = "old", decay: float = 0.0) -> None:
        """EMA-update target adapter parameters from source adapter parameters."""
        if not 0.0 <= decay <= 1.0:
            raise ValueError(f"Adapter EMA decay must be in [0, 1], got {decay}.")
        with self._adapter_state_context(), torch.no_grad():
            source_params = self._active_adapter_trainable_params(source)
            target_params = self._active_adapter_trainable_params(target)
            if len(source_params) != len(target_params) or not source_params:
                raise ValueError(
                    f"Adapter EMA {source!r} -> {target!r} found mismatched params: "
                    f"{len(source_params)} vs {len(target_params)}"
                )
            for source_param, target_param in zip(source_params, target_params, strict=True):
                target_param.lerp_(source_param, 1.0 - decay)

    @contextmanager
    def disable_adapter(self):
        """Temporarily disable all PEFT adapters."""
        try:
            self.module.disable_adapters()
            yield
        finally:
            self.module.enable_adapters()
