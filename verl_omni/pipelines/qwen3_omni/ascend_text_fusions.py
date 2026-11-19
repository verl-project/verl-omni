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
"""RMSNorm and RoPE benchmark variant from verl-omni PR #819."""

import torch
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as modeling


def enable_ascend_rmsnorm_rope() -> None:
    """Enable the two text fusions in the benchmark's training process."""
    from verl.models.transformers.npu_patch import apply_rotary_pos_emb_npu, rms_norm_forward_npu

    if modeling.Qwen3OmniMoeThinkerTextRMSNorm.forward is rms_norm_forward_npu:
        return

    eager_rope = modeling.apply_rotary_pos_emb

    def apply_rotary_pos_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.device.type == "npu" and q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16:
            return apply_rotary_pos_emb_npu(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
        return eager_rope(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    modeling.Qwen3OmniMoeThinkerTextRMSNorm.forward = rms_norm_forward_npu
    modeling.Qwen3OmniMoeTextRMSNorm.forward = rms_norm_forward_npu
    modeling.apply_rotary_pos_emb = apply_rotary_pos_emb
    print("AVQA Ascend text fusions enabled (RMSNorm, RoPE)", flush=True)
