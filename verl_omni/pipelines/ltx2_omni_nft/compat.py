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
"""Runtime compatibility fixes scoped to the LTX-2.3 adapter."""

from types import MethodType

import torch
from verl.utils.device import get_device_name


def ltx_affine_free_rms_norm_forward(rms_norm, hidden_states: torch.Tensor) -> torch.Tensor:
    """Normalize the last dimension without affine parameters, preserving input dtype.

    Accumulate the mean square in FP32, multiply by its reciprocal square root
    with ``rms_norm.eps``, and cast back to the input dtype on the same device.
    The input is not modified and gradients remain connected.
    """
    input_dtype = hidden_states.dtype
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return (hidden_states * torch.rsqrt(variance + rms_norm.eps)).to(input_dtype)


def apply_ltx_npu_rms_norm_workaround(module: torch.nn.Module) -> torch.nn.Module:
    """Replace affine-free Diffusers RMSNorm forwards in this module tree on NPU.

    When get_device_name resolves to NPU, bind the eager normalization helper to
    each RMSNorm instance whose weight is ``None`` and return the same module.
    Other layers and the global RMSNorm class are unchanged. Bindings persist
    for these instances; this helper provides no undo operation.

    Notes:
        Diffusers issue #14380 describes the fused NPU path receiving an
        unsupported null gamma for affine-free layers.
    """
    if get_device_name() != "npu":
        return module

    from diffusers.models.normalization import RMSNorm

    for submodule in module.modules():
        if isinstance(submodule, RMSNorm) and submodule.weight is None:
            submodule.forward = MethodType(ltx_affine_free_rms_norm_forward, submodule)
    return module


__all__ = ["apply_ltx_npu_rms_norm_workaround", "ltx_affine_free_rms_norm_forward"]
