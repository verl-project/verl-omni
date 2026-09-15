# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Complete the CPU probe's Diffusers-to-FA3 failure chain with real FA3.

The CPU tests establish that compiled Diffusers preparation can pass an
unbacked ``max_seqlen_k`` to an FA3-shaped custom op. This file loads the real
Hub FA3 registration and invokes its fake backward implementation in
``FakeTensorMode``; no CUDA kernel is launched. The concrete case guards the
test setup, while the unbacked case records the FA3 behavior that makes the
eager metadata boundary necessary.

If the concrete case fails, repair the test setup for the new Hub API or
environment before drawing conclusions about the workaround. If only the
unbacked case starts passing, revalidate the end-to-end compiled FA3 path and
remove only that rationale. Keep the shared eager boundary while the
independent Inductor cumsum failure remains.
"""

import pytest
import torch
from diffusers.models import attention_dispatch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode, ShapeEnv


def _run_fa3_backward_fake(*, symbolic_max_seqlen_k: bool):
    assert torch.cuda.is_available(), "The FA3 contract test requires a CUDA runner."
    backend = attention_dispatch.AttentionBackendName._FLASH_3_VARLEN_HUB
    attention_dispatch._maybe_download_kernel_for_backend(backend)
    backward = attention_dispatch._HUB_KERNELS_REGISTRY[backend].wrapped_backward_fn
    assert backward is not None, "The Hub FA3 kernel no longer exposes its wrapped backward operation."

    shape_env = ShapeEnv()
    fake_mode = FakeTensorMode(shape_env=shape_env)
    with fake_mode:
        query = torch.empty((16, 2, 64), device="cuda", dtype=torch.bfloat16)
        key = torch.empty((8, 2, 64), device="cuda", dtype=torch.bfloat16)
        value = torch.empty_like(key)
        cumulative_lengths_q = torch.empty((3,), device="cuda", dtype=torch.int32)
        cumulative_lengths_k = torch.empty((3,), device="cuda", dtype=torch.int32)
        max_seqlen_k = shape_env.create_unbacked_symint() if symbolic_max_seqlen_k else 5

        return backward(
            torch.empty_like(query),
            query,
            key,
            value,
            torch.empty_like(query),
            torch.empty((2, 16), device="cuda", dtype=torch.float32),
            cumulative_lengths_q,
            cumulative_lengths_k,
            None,
            None,
            8,
            max_seqlen_k,
            torch.empty_like(query),
            torch.empty_like(key),
            torch.empty_like(value),
            0.125,
            False,
            -1,
            -1,
            0.0,
            False,
            0,
        )


def test_fa3_backward_fake_accepts_concrete_maximum_sequence_length():
    result = _run_fa3_backward_fake(symbolic_max_seqlen_k=False)

    assert result.device.type == "cuda", (
        "The real FA3 fake-backward baseline no longer accepts a concrete max_seqlen_k. Repair this test for the "
        "new Hub API or environment before interpreting the unbacked-symbol result."
    )


def test_fa3_backward_fake_still_rejects_unbacked_maximum_sequence_length():
    try:
        _run_fa3_backward_fake(symbolic_max_seqlen_k=True)
    except GuardOnDataDependentSymNode:
        pass
    else:
        pytest.fail(
            "FA3's backward fake implementation now accepts an unbacked max_seqlen_k. Revalidate the real "
            "compiled masked and unmasked paths and remove only this rationale if they succeed. Keep the shared "
            "eager boundary while the independent Inductor cumsum failure remains."
        )
