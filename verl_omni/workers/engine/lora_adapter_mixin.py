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

from contextlib import contextmanager, nullcontext

import torch
from peft import LoraConfig
from verl.utils.py_functional import convert_to_regular_types


class LoRAAdapterMixin:
    """Backend-agnostic helpers for named PEFT/LoRA policy adapters."""

    def _build_lora_module(self, module):
        lora_adapter_path = getattr(self.model_config, "lora_adapter_path", None)
        policy_state_adapters = tuple(getattr(self.model_config, "policy_state_adapters", ("default",)))
        extra_adapters = tuple(adapter for adapter in policy_state_adapters if adapter not in ("default", "reference"))
        if lora_adapter_path is not None:
            from verl.utils.fs import copy_to_local

            print(f"Loading pre-trained LoRA adapter to from: {lora_adapter_path}")
            local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.model_config.use_shm)

            module.load_lora_adapter(local_adapter_path)
            peft_config = getattr(module, "peft_config", {}).get("default", None)
            for adapter_name in extra_adapters:
                if peft_config is not None and adapter_name not in getattr(module, "peft_config", {}):
                    module.add_adapter(peft_config, adapter_name=adapter_name)
        else:
            lora_config = {
                "r": self.model_config.lora_rank,
                "lora_alpha": self.model_config.lora_alpha,
                "init_lora_weights": self.model_config.lora_init_weights,
                "target_modules": convert_to_regular_types(self.model_config.target_modules),
                "target_parameters": convert_to_regular_types(self.model_config.target_parameters),
                "exclude_modules": convert_to_regular_types(self.model_config.exclude_modules),
                "bias": "none",
            }
            module.add_adapter(LoraConfig(**lora_config), adapter_name="default")
            for adapter_name in extra_adapters:
                module.add_adapter(LoraConfig(**lora_config), adapter_name=adapter_name)

        if "default" in policy_state_adapters and hasattr(module, "set_adapter"):
            module.set_adapter("default")

        return module

    @property
    def _peft_module(self):
        """PEFT model that owns adapter state (unwraps FSDP when applicable)."""
        return getattr(self.module, "_fsdp_wrapped_module", self.module)

    @contextmanager
    def _adapter_state_context(self):
        """Open writable adapter parameter access (FSDP summon when applicable)."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
        from verl.utils.memory_utils import aggressive_empty_cache

        is_fsdp_module = fsdp_version(self.module) == 1
        is_fsdp2_module = fsdp_version(self.module) == 2
        is_offload_param = getattr(self, "_is_offload_param", False)
        origin_module_device = next(self.module.parameters()).device.type
        if (is_fsdp_module or is_fsdp2_module) and (is_offload_param or origin_module_device == "cpu"):
            load_fsdp_model_to_gpu(self.module)

        ctx = FSDP.summon_full_params(self.module, writeback=True, recurse=True) if is_fsdp_module else nullcontext()
        try:
            with ctx:
                yield
        finally:
            self._set_adapter("default")
            if is_offload_param:
                offload_fsdp_model_to_cpu(self.module)
                aggressive_empty_cache(force_sync=True)

    def _set_adapter(self, name: str):
        peft_module = self._peft_module
        if not hasattr(peft_module, "set_adapter"):
            raise AttributeError(f"Module does not support set_adapter({name!r})")
        peft_module.set_adapter(name)

    @contextmanager
    def use_adapter(self, name: str):
        """Temporarily select a named PEFT adapter."""
        self._set_adapter(name)
        try:
            yield
        finally:
            self._set_adapter("default")

    def _active_adapter_trainable_params(self, adapter_name: str) -> list[torch.nn.Parameter]:
        peft_model = self._peft_module
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
