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
"""Map PEFT LoRA checkpoint tensors onto a transformer schema and fold one adapter."""

import re
from collections.abc import Mapping
from dataclasses import dataclass

import torch

LORA_METADATA_NAME = "lora_train_meta.json"
_LORA_TENSOR = re.compile(r"(?P<module>.+)\.lora_(?P<side>[AB])\.(?P<adapter>[^.]+)\.weight")


@dataclass(frozen=True)
class LoRAPlan:
    """Checkpoint sources for every schema tensor plus the selected adapter's updates.

    Args:
        sources: Schema key to checkpoint key. Empty when base weights come from the base model.
        updates: Schema weight key to the selected ``(lora_A, lora_B)`` checkpoint keys.
        shapes: Global shapes of the selected adapter tensors.
        scaling: ``lora_alpha / r`` applied to every update.
        record: Portable manifest description of the fusion.
    """

    sources: dict[str, str]
    updates: dict[str, tuple[str, str]]
    shapes: dict[str, tuple[int, ...]]
    scaling: float
    record: dict


def plan_lora_fusion(
    shapes: Mapping[str, tuple[int, ...]],
    expected_shapes: Mapping[str, tuple[int, ...]],
    metadata: Mapping,
    adapter_name: str,
) -> LoRAPlan:
    """Validate a full-state or LoRA-only checkpoint and select one linear LoRA adapter."""
    rank, alpha = metadata.get("r"), metadata.get("lora_alpha")
    if type(rank) is not int or rank < 1 or type(alpha) is not int or alpha < 1:
        raise ValueError(f"{LORA_METADATA_NAME} must declare a positive integer r and a positive integer lora_alpha")

    adapters: dict[str, dict[str, dict[str, str]]] = {}
    sources: dict[str, str] = {}
    for key in shapes:
        match = _LORA_TENSOR.fullmatch(key)
        if match:
            adapters.setdefault(match["adapter"], {}).setdefault(match["module"], {})[match["side"]] = key
            continue
        if "lora_" in key or ".adapter_" in key:
            raise ValueError(f"Unsupported adapter tensor (only linear LoRA A/B weights are supported): {key}")
        name = key.replace(".base_layer.", ".", 1)
        if ".base_layer." in name or name in sources:
            raise ValueError(f"Ambiguous LoRA base tensor: {key}")
        sources[name] = key
    if adapter_name not in adapters:
        raise ValueError(f"LoRA adapter {adapter_name!r} not found in checkpoint; available: {sorted(adapters)}")
    if sources and set(sources) != set(expected_shapes):
        raise ValueError(
            f"Incomplete transformer state: missing={sorted(set(expected_shapes) - set(sources))[:8]}, "
            f"unexpected={sorted(set(sources) - set(expected_shapes))[:8]}"
        )

    updates = {}
    for name, modules in sorted(adapters.items()):
        for module, sides in sorted(modules.items()):
            if set(sides) != {"A", "B"}:
                raise ValueError(f"Unpaired LoRA tensors for {module} (adapter {name!r})")
            target = f"{module}.weight"
            a, b = shapes[sides["A"]], shapes[sides["B"]]
            if target not in expected_shapes or len(expected_shapes[target]) != 2 or len(a) != 2 or len(b) != 2:
                raise ValueError(f"LoRA target is not a linear weight: {module}")
            if sources and sources[target] != f"{module}.base_layer.weight":
                raise ValueError(f"LoRA target lacks a base_layer weight: {module}")
            if a[0] != rank or b[1] != rank:
                raise ValueError(f"LoRA rank of {module} (adapter {name!r}) differs from {LORA_METADATA_NAME} r={rank}")
            if (b[0], a[1]) != expected_shapes[target]:
                raise ValueError(f"LoRA update shape does not match the base weight: {module}")
            if name == adapter_name:
                updates[target] = (sides["A"], sides["B"])

    record = {
        "adapter_name": adapter_name,
        "r": rank,
        "lora_alpha": alpha,
        "scaling": alpha / rank,
        "base_weights": "checkpoint" if sources else "base_model",
        "fused_modules": sorted(adapters[adapter_name]),
        "excluded_adapters": sorted(set(adapters) - {adapter_name}),
    }
    selected = {key: shapes[key] for pair in updates.values() for key in pair}
    return LoRAPlan(sources, updates, selected, alpha / rank, record)


def fuse_lora(base: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor, scaling: float) -> torch.Tensor:
    """Return ``base + scaling * lora_b @ lora_a`` computed in fp32 and cast to the base dtype."""
    if not base.is_floating_point():
        raise ValueError("LoRA targets must be floating-point weights")
    fused = (base.float() + scaling * (lora_b.float() @ lora_a.float())).to(base.dtype)
    if not torch.isfinite(fused).all():
        raise ValueError("LoRA fusion produced non-finite weights")
    return fused
