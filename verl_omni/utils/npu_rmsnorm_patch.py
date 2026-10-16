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
"""Patch diffusers ``RMSNorm`` to support affine-less norms on NPU.

LTX-2.x transformers set ``norm_elementwise_affine=false``, so every block norm
(``norm1``/``audio_norm1``/...) is built with ``weight=None``. diffusers'
``RMSNorm.forward`` NPU branch feeds that ``None`` unguarded into
``torch_npu.npu_rms_norm`` (``Expected a proper Tensor but got None ... 'gamma'``),
crashing the first trainer-side forward on Ascend. The CPU/CUDA branch handles
``weight is None`` by plain RMS normalization without scaling; this patch
reproduces that behavior on NPU only, and delegates every other path to the
upstream implementation untouched.
"""

import torch

_patched = False


def _is_npu_available() -> bool:
    # Resolved at call time (not captured) so it follows the same symbol
    # ``diffusers.models.normalization`` uses and stays CPU-testable.
    from diffusers.utils import is_torch_npu_available

    return is_torch_npu_available()


def patch_diffusers_npu_rmsnorm() -> None:
    """Make diffusers ``RMSNorm.forward`` handle ``elementwise_affine=False`` on NPU.

    Idempotent. Class-level patch, so already-constructed LTX blocks pick it up at
    call time. Zero behavior change on non-NPU backends and for affine norms.
    """
    global _patched
    if _patched:
        return

    import diffusers.models.normalization as normalization

    original = normalization.RMSNorm.forward

    def forward(self, hidden_states):
        if self.weight is None and _is_npu_available():
            # Mirrors the upstream CPU/CUDA ``weight is None`` branch: plain RMS
            # normalization in float32, then cast back to the input dtype.
            input_dtype = hidden_states.dtype
            variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
            return hidden_states.to(input_dtype)
        return original(self, hidden_states)

    normalization.RMSNorm.forward = forward
    _patched = True
