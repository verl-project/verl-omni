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
"""MoE expert unfusing helpers for Qwen3-Omni Thinker LoRA training."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

FUSED_EXPERT_CLASS_NAME = "Qwen3OmniMoeThinkerTextExperts"


def _tf5_moe_available() -> bool:
    try:
        import transformers.integrations.moe  # noqa: F401

        return True
    except ImportError:
        return False


def unfuse_qwen3_omni_thinker_moe_experts(model) -> int:
    """Replace fused ``Qwen3OmniMoeThinkerTextExperts`` with per-expert ``nn.Linear``.

    tf5 stores MoE experts as fused 3D parameters; PEFT LoRA needs per-expert
    ``nn.Linear`` modules to attach adapters to ``gate_proj`` / ``up_proj`` /
    ``down_proj``.

    Returns:
        int: Number of fused expert modules converted.
    """
    if not _tf5_moe_available():
        return 0

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class _Expert(nn.Module):
        def __init__(self, hidden: int, intermediate: int) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
            self.up_proj = nn.Linear(hidden, intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    class _Qwen3OmniMoeThinkerTextExpertsUnfused(nn.Module):
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

    converted = 0
    for path, module in list(model.named_modules()):
        if type(module).__name__ != FUSED_EXPERT_CLASS_NAME:
            continue
        gate_up = module.gate_up_proj.data
        down = module.down_proj.data
        n = gate_up.shape[0]
        di = gate_up.shape[1] // 2
        h = gate_up.shape[2]

        new_mod = _Qwen3OmniMoeThinkerTextExpertsUnfused(n, h, di, module.act_fn)
        for i, expert in enumerate(new_mod.experts):
            expert.gate_proj.weight = nn.Parameter(gate_up[i, :di, :].clone())
            expert.up_proj.weight = nn.Parameter(gate_up[i, di:, :].clone())
            expert.down_proj.weight = nn.Parameter(down[i].clone())

        parent_path, _, child_name = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, child_name, new_mod)
        converted += 1

    if converted:
        logger.info("verl_omni: unfused %d Qwen3-Omni Thinker MoE expert module(s) for LoRA", converted)
    return converted


def model_uses_lora(model_config) -> bool:
    lora_rank = getattr(model_config, "lora_rank", 0)
    if lora_rank > 0:
        return True
    lora = getattr(model_config, "lora", None)
    if isinstance(lora, dict) and lora.get("rank", 0) > 0:
        return True
    return getattr(model_config, "lora_adapter_path", None) is not None
