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

"""Manual LoRA hooks for LingBot ``torch.nn.Linear`` layers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

_LORA_WEIGHT_RE = re.compile(r"^(?P<module>.+)\.lora_(?P<side>[AB])(?:\.[^.]+)?\.weight$")
_PREFIXES = (
    "base_model.model.",
    "model.",
    "module.",
    "_fsdp_wrapped_module.",
    "transformer.",
)


@dataclass(frozen=True)
class LinearLoRALayer:
    """A single LoRA pair for an ``nn.Linear`` module."""

    lora_a: torch.Tensor
    lora_b: torch.Tensor
    scale: float


@dataclass
class LinearLoRAAdapter:
    """LoRA tensors grouped by transformer module name."""

    adapter_id: int
    layers: dict[str, LinearLoRALayer]


class ManualLinearLoRAManager:
    """Apply PEFT LoRA tensors to standard ``torch.nn.Linear`` modules via hooks."""

    def __init__(self, root_module: torch.nn.Module):
        self.root_module = root_module
        self._modules = dict(root_module.named_modules())
        self._adapters: dict[int, LinearLoRAAdapter] = {}
        self._active_adapter_id: int | None = None
        self._active_scale = 1.0
        self._hooks: dict[str, torch.utils.hooks.RemovableHandle] = {}
        self._pinned: set[int] = set()

    @property
    def active_adapter_id(self) -> int | None:
        return self._active_adapter_id

    def add_adapter(self, lora_request: Any) -> bool:
        adapter_id = int(lora_request.lora_int_id)
        lora_tensors = getattr(lora_request, "lora_tensors", None)
        if not lora_tensors:
            raise ValueError(
                "LingBot rollout can load LoRA only from in-memory tensors. "
                "Use the colocated trainer LoRA sync path, not a filesystem LoRA path."
            )
        peft_config = getattr(lora_request, "peft_config", None) or {}
        adapter = self._build_adapter(adapter_id, lora_tensors, peft_config)
        self._adapters[adapter_id] = adapter
        for module_name in adapter.layers:
            self._ensure_hook(module_name)
        return True

    def remove_adapter(self, adapter_id: int) -> bool:
        existed = int(adapter_id) in self._adapters
        self._adapters.pop(int(adapter_id), None)
        self._pinned.discard(int(adapter_id))
        if self._active_adapter_id == int(adapter_id):
            self._active_adapter_id = None
            self._active_scale = 1.0
        return existed

    def list_adapters(self) -> list[int]:
        return sorted(self._adapters)

    def pin_adapter(self, adapter_id: int) -> bool:
        adapter_id = int(adapter_id)
        if adapter_id not in self._adapters:
            return False
        self._pinned.add(adapter_id)
        return True

    def set_active_adapter(self, lora_request: Any | None, lora_scale: float = 1.0) -> None:
        if lora_request is None or math.isclose(float(lora_scale), 0.0):
            self._active_adapter_id = None
            self._active_scale = 1.0
            return
        adapter_id = int(getattr(lora_request, "lora_int_id", lora_request))
        if adapter_id not in self._adapters:
            self.add_adapter(lora_request)
        self._active_adapter_id = adapter_id
        self._active_scale = float(lora_scale)

    def _build_adapter(
        self,
        adapter_id: int,
        lora_tensors: dict[str, torch.Tensor],
        peft_config: dict[str, Any],
    ) -> LinearLoRAAdapter:
        grouped: dict[str, dict[str, torch.Tensor]] = {}
        for name, tensor in lora_tensors.items():
            parsed = self._parse_lora_weight_name(name)
            if parsed is None:
                continue
            module_name, side = parsed
            grouped.setdefault(module_name, {})[side] = tensor

        layers: dict[str, LinearLoRALayer] = {}
        missing_modules: list[str] = []
        incomplete_modules: list[str] = []
        for module_name, pair in grouped.items():
            if "A" not in pair or "B" not in pair:
                incomplete_modules.append(module_name)
                continue
            module = self._modules.get(module_name)
            if not isinstance(module, torch.nn.Linear):
                missing_modules.append(module_name)
                continue
            lora_a = pair["A"].detach().to(device=module.weight.device, dtype=module.weight.dtype).contiguous()
            lora_b = pair["B"].detach().to(device=module.weight.device, dtype=module.weight.dtype).contiguous()
            if lora_a.ndim != 2 or lora_b.ndim != 2:
                raise ValueError(f"LoRA tensors for {module_name!r} must be matrices.")
            rank = int(lora_a.shape[0])
            if rank <= 0 or lora_b.shape[1] != rank:
                raise ValueError(
                    f"LoRA tensor shapes for {module_name!r} are incompatible: "
                    f"A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}."
                )
            if lora_a.shape[1] != module.in_features or lora_b.shape[0] != module.out_features:
                raise ValueError(
                    f"LoRA tensor shapes for {module_name!r} do not match Linear "
                    f"({module.in_features}, {module.out_features}): "
                    f"A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}."
                )
            alpha = self._resolve_lora_alpha(module_name, peft_config, rank)
            if peft_config.get("use_rslora", False):
                base_scale = alpha / math.sqrt(rank)
            else:
                base_scale = alpha / rank
            layers[module_name] = LinearLoRALayer(lora_a=lora_a, lora_b=lora_b, scale=float(base_scale))

        if incomplete_modules:
            raise ValueError(f"Incomplete LoRA A/B tensors for LingBot modules: {sorted(incomplete_modules)[:5]}.")
        if not layers:
            detail = f" Missing/non-Linear modules include: {sorted(missing_modules)[:5]}." if missing_modules else ""
            raise ValueError(f"No LingBot nn.Linear LoRA tensors matched the rollout transformer.{detail}")
        return LinearLoRAAdapter(adapter_id=adapter_id, layers=layers)

    def _parse_lora_weight_name(self, name: str) -> tuple[str, str] | None:
        match = _LORA_WEIGHT_RE.match(name)
        if match is None:
            return None
        module_name = self._normalize_module_name(match.group("module"))
        return module_name, match.group("side")

    def _normalize_module_name(self, module_name: str) -> str:
        normalized = module_name
        changed = True
        while changed:
            changed = False
            for prefix in _PREFIXES:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix) :]
                    changed = True
        normalized = normalized.replace(".base_layer", "")
        if normalized in self._modules:
            return normalized
        # PEFT/FSDP wrappers can insert prefixes before the actual transformer
        # path.  Prefer the longest suffix that resolves to a known module.
        parts = normalized.split(".")
        for idx in range(1, len(parts)):
            candidate = ".".join(parts[idx:])
            if candidate in self._modules:
                return candidate
        return normalized

    @staticmethod
    def _resolve_lora_alpha(module_name: str, peft_config: dict[str, Any], rank: int) -> float:
        alpha_pattern = peft_config.get("alpha_pattern") or {}
        if isinstance(alpha_pattern, dict):
            for suffix, alpha in alpha_pattern.items():
                if module_name == suffix or module_name.endswith(f".{suffix}"):
                    return float(alpha)
        return float(peft_config.get("lora_alpha", rank))

    def _ensure_hook(self, module_name: str) -> None:
        if module_name in self._hooks:
            return
        module = self._modules[module_name]

        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
            return self._apply_lora(module_name, inputs, output)

        self._hooks[module_name] = module.register_forward_hook(hook)

    def _apply_lora(self, module_name: str, inputs: tuple[Any, ...], output: Any) -> Any:
        if self._active_adapter_id is None or not isinstance(output, torch.Tensor) or not inputs:
            return output
        adapter = self._adapters.get(self._active_adapter_id)
        if adapter is None:
            return output
        layer = adapter.layers.get(module_name)
        if layer is None:
            return output
        hidden_states = inputs[0]
        if not isinstance(hidden_states, torch.Tensor):
            return output
        lora_input = hidden_states.to(device=layer.lora_a.device, dtype=layer.lora_a.dtype)
        delta = F.linear(F.linear(lora_input, layer.lora_a), layer.lora_b)
        delta = delta * (layer.scale * self._active_scale)
        return output + delta.to(device=output.device, dtype=output.dtype)
