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

Patches dropped from the adapter:
- ``_apply_tie_embeddings_fix``   → v5 default is ``False``
- ``_install_moe_unfuse_hook``    → PEFT handles MoE natively
- ``module._no_split_modules``    → thinker class already correct
"""

import pytest
import torch
import torch.nn as nn


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

    # 2. PEFT doubled the rank — gate_proj + up_proj fused into gate_up_proj.
    cur_r = peft_model.peft_config["default"].r
    assert cur_r != 8, f"PEFT should have doubled LoRA rank (8 → 16) for fused gate_up_proj, got r={cur_r}"

    # 3. gate_up_proj parameter exists and is not an nn.Linear module.
    assert hasattr(peft_model.experts, "gate_up_proj"), "gate_up_proj parameter should still exist after PEFT wrapping"


def test_adapter_source_has_no_peft_references():
    """The adapter source must not import or reference ``peft``."""
    import verl_omni.pipelines.qwen3_omni.thinker_training_adapter as ta

    source = open(ta.__file__).read()
    err = "adapter source references 'peft' — PEFT handles MoE natively"
    assert "peft" not in source, err
    assert "get_peft_model" not in source, err


def test_tie_word_embeddings_is_false_by_default():
    """v5 config defaults ``tie_word_embeddings=False``."""
    pytest.importorskip("transformers")
    from transformers.models.qwen3_omni_moe import Qwen3OmniMoeConfig

    cfg = Qwen3OmniMoeConfig()
    assert cfg.tie_word_embeddings is False, "tie_word_embeddings should default to False in transformers >= 5.0"


def test_thinker_class_no_split_modules_is_correct():
    """Thinker subclass already uses the right FSDP layer class name."""
    pytest.importorskip("transformers")
    from transformers.models.qwen3_omni_moe import (
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )

    expected = ["Qwen3OmniMoeThinkerTextDecoderLayer"]
    actual = Qwen3OmniMoeThinkerForConditionalGeneration._no_split_modules
    assert actual == expected, f"_no_split_modules should be {expected} in transformers >= 5.0, got {actual}"
