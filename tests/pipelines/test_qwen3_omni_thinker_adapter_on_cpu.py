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
"""Regression: upstream fixes that made patches unnecessary for Qwen3-Omni.

# TODO (mike): remove this comment once the patches are dropped.
Patches dropped from the adapter:
- ``_apply_tie_embeddings_fix``   → v5 config defaults ``tie_word_embeddings=False``
- ``_install_moe_unfuse_hook``    → PEFT >= 0.19.0 handles MoE natively
- ``module._no_split_modules``    → thinker class already correct
"""

import importlib.metadata

import pytest
import torch
import torch.nn as nn
from packaging.version import parse as parse_version


def _require_version(pkg_name: str, min_version: str):
    """Raise ``AssertionError`` if *pkg_name* is below *min_version*."""
    ver = importlib.metadata.version(pkg_name)
    assert parse_version(ver) >= parse_version(min_version), f"{pkg_name} >= {min_version} is required, got {ver}"


def _has_lora(module: nn.Module) -> bool:
    """Return True if *module* was wrapped with LoRA by PEFT."""
    return hasattr(module, "lora_A") and hasattr(module, "lora_B")


class _FusedMoEExperts(nn.Module):
    """Minimal Qwen3-Omni-style fused expert group.

    ``gate_up_proj`` is a 3D ``nn.Parameter``, not ``nn.Linear``.
    """

    def __init__(self, num_experts=4, hidden=64, intermediate=128):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * intermediate, hidden))
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden, intermediate))


class _MoEModel(nn.Module):
    """Minimal model with ``config.model_type`` for PEFT conversion routing.

    Contains one dense ``nn.Linear`` block and one fused MoE expert
    block so we can verify PEFT correctly routes LoRA to both.
    """

    def __init__(self, hidden=64, intermediate=128, num_experts=4):
        super().__init__()
        self.config = type("C", (), {"model_type": "qwen3_omni_moe"})()

        # Dense layers — standard nn.Linear, PEFT wraps these directly.
        self.dense = nn.Module()
        self.dense.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.dense.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.dense.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.dense.o_proj = nn.Linear(hidden, hidden, bias=False)

        # MoE experts — fused parameters, PEFT maps via target_parameters
        # with doubled rank.
        self.experts = _FusedMoEExperts(num_experts, hidden, intermediate)


def test_peft_lora_attaches_to_fused_moe_natively():
    """PEFT converts gate_proj+up_proj → gate_up_proj with doubled rank."""
    pytest.importorskip("peft")
    _require_version("peft", "0.19.0")
    from peft import LoraConfig, get_peft_model

    model = _MoEModel()
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    peft_model = get_peft_model(model, lora_config)

    # 1. Dense nn.Linear layers get standard LoRA.
    assert _has_lora(peft_model.dense.q_proj), "dense q_proj missing LoRA"
    assert _has_lora(peft_model.dense.k_proj), "dense k_proj missing LoRA"
    assert _has_lora(peft_model.dense.v_proj), "dense v_proj missing LoRA"
    assert _has_lora(peft_model.dense.o_proj), "dense o_proj missing LoRA"

    # 2. PEFT fused gate_proj+up_proj → gate_up_proj, and down_proj, in target_parameters.
    cfg = peft_model.peft_config["default"]
    assert "gate_up_proj" in cfg.target_parameters, (
        f"PEFT should move gate_proj+up_proj to target_parameters, got {cfg.target_parameters}"
    )
    assert "down_proj" in cfg.target_parameters, (
        f"PEFT should move down_proj to target_parameters, got {cfg.target_parameters}"
    )

    # 3. PEFT doubled rank and alpha via rank_pattern (base .r stays unchanged).
    gate_up_r = next((v for k, v in cfg.rank_pattern.items() if "gate_up_proj" in k), None)
    assert gate_up_r == 16, f"PEFT should double rank (8 → 16) in rank_pattern for gate_up_proj, got {gate_up_r}"

    # 4. gate_up_proj parameter is preserved through PEFT wrapping.
    obj = peft_model.experts
    while hasattr(obj, "base_layer"):
        obj = obj.base_layer
    assert hasattr(obj, "gate_up_proj"), "gate_up_proj parameter should still be reachable through PEFT wrapping"


def test_tie_word_embeddings_is_false_by_default():
    """v5 config: ``tie_word_embeddings=False`` on the thinker sub-config."""
    pytest.importorskip("transformers")
    _require_version("transformers", "5.0.0")
    from transformers.models.qwen3_omni_moe import Qwen3OmniMoeConfig

    cfg = Qwen3OmniMoeConfig()
    # The umbrella config delegates to sub-configs; the thinker sub-config
    # is what FSDP uses after adapter strips non-thinker modules.
    assert cfg.thinker_config.tie_word_embeddings is False, (
        "thinker_config.tie_word_embeddings should default to False in transformers >= 5.0"
    )


def test_thinker_class_no_split_modules_is_correct():
    """Thinker subclass already uses the right FSDP layer class name."""
    pytest.importorskip("transformers")
    _require_version("transformers", "5.0.0")
    from transformers.models.qwen3_omni_moe import (
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )

    expected = ["Qwen3OmniMoeAudioEncoder", "Qwen3OmniMoeVisionEncoder"]
    actual = Qwen3OmniMoeThinkerForConditionalGeneration._no_split_modules
    assert actual == expected, f"_no_split_modules should be {expected} in transformers >= 5.0, got {actual}"
