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

    @contextmanager
    def disable_adapter(self):
        """Temporarily disable all PEFT adapters."""
        try:
            self.module.disable_adapters()
            yield
        finally:
            self.module.enable_adapters()
