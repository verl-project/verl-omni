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
"""Qwen3-Omni Thinker MoE expert unfuse patch for PEFT LoRA.

Import this module as a verl ``external_lib`` when training Qwen3-Omni with
FSDP + PEFT LoRA on MoE expert layers. It only installs the expert unfuse hook;
it does not patch tokenizer or processor loading.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_EXPERTS_UNFUSE_APPLIED = False


class _Expert(nn.Module):
    def __init__(self, hidden: int, intermediate: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)


class _Qwen3OmniMoeThinkerTextExpertsUnfused(nn.Module):
    """Per-expert nn.Linear replacement for the tf5 fused Qwen3OmniMoeThinkerTextExperts."""

    def __init__(self, n: int, hidden: int, intermediate: int, act_fn) -> None:
        super().__init__()
        self.num_experts = n
        self.act_fn = act_fn
        self.experts = nn.ModuleList([_Expert(hidden, intermediate) for _ in range(n)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, self.num_experts).permute(2, 1, 0)
            hits = mask.sum(dim=(-1, -2)).gt(0).nonzero()
        for row in hits:
            i = row[0].item()
            if i >= self.num_experts:
                continue
            top_k_pos, tok_idx = torch.where(mask[i])
            x = hidden_states[tok_idx]
            e = self.experts[i]
            out = e.down_proj(self.act_fn(e.gate_proj(x)) * e.up_proj(x))
            out = out * top_k_weights[tok_idx, top_k_pos, None]
            final.index_add_(0, tok_idx, out.to(final.dtype))
        return final


def unfuse_qwen3_omni_thinker_experts(model) -> int:
    """Replace fused Qwen3-Omni thinker experts with per-expert Linear modules.

    This is intentionally callable outside ``get_peft_model`` so adapter reload
    paths can build the same module structure used during training.
    """
    converted = 0
    for path, module in list(model.named_modules()):
        if type(module).__name__ != "Qwen3OmniMoeThinkerTextExperts":
            continue
        gate_up = module.gate_up_proj.data  # (n, 2*intermediate, hidden)
        down = module.down_proj.data  # (n, hidden, intermediate)
        n = gate_up.shape[0]
        di = gate_up.shape[1] // 2
        h = gate_up.shape[2]

        new_mod = _Qwen3OmniMoeThinkerTextExpertsUnfused(n, h, di, module.act_fn)
        for i, e in enumerate(new_mod.experts):
            e.gate_proj.weight = nn.Parameter(gate_up[i, :di, :].clone())
            e.up_proj.weight = nn.Parameter(gate_up[i, di:, :].clone())
            e.down_proj.weight = nn.Parameter(down[i].clone())

        parent_path, _, child_name = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, child_name, new_mod)
        converted += 1

    if converted:
        logger.info("verl_omni: unfused %d Qwen3-Omni thinker expert module(s)", converted)
    return converted


def _patch_unfuse_qwen3_omni_thinker_experts() -> None:
    """Hook peft.get_peft_model to unfuse tf5 fused MoE experts before LoRA (tf5+ only).

    Converts Qwen3OmniMoeThinkerTextExperts (fused 3D params) to per-expert nn.Linear.
    """
    global _EXPERTS_UNFUSE_APPLIED
    if _EXPERTS_UNFUSE_APPLIED:
        return

    # tf5 sentinel: transformers.integrations.moe only exists in transformers >= 5.x
    try:
        import transformers.integrations.moe  # noqa
        import peft as _peft
    except ImportError:
        return

    _orig_get_peft_model = _peft.get_peft_model

    # No-op PEFT's gate_proj/up_proj -> gate_up_proj remap for Qwen3-Omni, else expert LoRA won't attach.
    try:
        import peft.utils.transformers_weight_conversion as _twc

        _orig_get_mapping = _twc.get_model_conversion_mapping
        _orig_convert = _twc.convert_peft_config_for_transformers

        def _patched_get_mapping(model):
            if type(model).__name__ == "Qwen3OmniMoeForConditionalGeneration":
                return []
            return _orig_get_mapping(model)

        def _patched_convert(peft_config, model=None, conversions=None):
            if model is not None and type(model).__name__ == "Qwen3OmniMoeForConditionalGeneration":
                return
            return _orig_convert(peft_config, model=model, conversions=conversions)

        _twc.get_model_conversion_mapping = _patched_get_mapping
        _twc.convert_peft_config_for_transformers = _patched_convert
    except (ImportError, AttributeError) as e:
        logger.warning("verl_omni: could not patch PEFT tf5 name remapping (%s); MoE expert LoRA may not attach", e)

    def _patched_get_peft_model(model, peft_config, **kwargs):
        if type(model).__name__ == "Qwen3OmniMoeForConditionalGeneration":
            unfuse_qwen3_omni_thinker_experts(model)
            # verl passes target_modules as a comma-separated string; PEFT treats it as regex, so split to a set.
            if isinstance(peft_config.target_modules, str) and "," in peft_config.target_modules:
                peft_config.target_modules = set(peft_config.target_modules.split(","))
        return _orig_get_peft_model(model, peft_config, **kwargs)

    _peft.get_peft_model = _patched_get_peft_model
    # Also update verl's module-level binding if it was already imported before us.
    import sys as _sys

    _vi = _sys.modules.get("verl.workers.engine.fsdp.transformer_impl")
    if _vi is not None:
        _vi.get_peft_model = _patched_get_peft_model
    _EXPERTS_UNFUSE_APPLIED = True
    logger.info("verl_omni: installed get_peft_model hook for Qwen3-Omni MoE expert unfusing (tf5+)")


# Apply on import so this module works as a verl ``external_lib`` target.
_patch_unfuse_qwen3_omni_thinker_experts()
