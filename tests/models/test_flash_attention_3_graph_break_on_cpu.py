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
"""CPU tests for the intentional FA3 varlen metadata graph break.

The workaround addresses two failures at the boundary between Diffusers and
FA3. These tests keep that boundary observable without requiring FA3 or CUDA:

* A custom op stands in for FA3 and records the symbolic ``max_seqlen_k`` seen
  by a kernel during tracing. The surrounding preparation and dispatch remain
  the real Diffusers implementation, so this checks behavior rather than its
  source text.
* The same probe after installing the workaround verifies that the eager
  boundary materializes the value before tracing resumes.
* A separate Inductor canary preserves evidence for the independent cumsum
  lowering failure that originally required the eager boundary.

The GPU contract tests then complete the chain by checking that the real Hub
FA3 fake implementation rejects the unbacked value observed here.

Failure triage:

* If an unpatched canary starts passing, verify the real masked and unmasked
  FA3 paths. This single eager boundary protects against both failures, so
  keep it while either remains. If only one was fixed, update that rationale
  and its canary; remove the workaround only if neither failure remains.
* If a patched-path assertion fails while its unpatched canary still fails,
  update the workaround for the new Diffusers/PyTorch behavior.
* If a private helper disappears, port both the workaround and these behavior
  tests to its replacement. A renamed helper alone does not make the
  workaround obsolete.
"""

from types import SimpleNamespace

import pytest
import torch
from diffusers.models import attention_dispatch
from torch._inductor.exc import InductorError
from torch.fx.experimental.symbolic_shapes import has_free_unbacked_symbols

from verl_omni.utils.diffusion_compile import _keep_varlen_attention_metadata_eager

_HELPER_NAMES = (
    "_prepare_for_flash_attn_or_sage_varlen_with_mask",
    "_prepare_for_flash_attn_or_sage_varlen_without_mask",
)
_FA3_MAX_SEQLEN_K_VALUES = []


@torch.library.custom_op("verl_omni_test::fa3_max_seqlen_k_probe", mutates_args=())
def _fa3_max_seqlen_k_probe(query: torch.Tensor, max_seqlen_k: int) -> torch.Tensor:
    """Model the compiler-visible tensor/scalar boundary of the FA3 custom op."""
    return query.clone()


@_fa3_max_seqlen_k_probe.register_fake
def _(query, max_seqlen_k):
    _FA3_MAX_SEQLEN_K_VALUES.append(max_seqlen_k)
    return torch.empty_like(query)


@pytest.fixture
def original_helpers(monkeypatch):
    helpers = {}
    for helper_name in _HELPER_NAMES:
        if not hasattr(attention_dispatch, helper_name):
            pytest.fail(
                f"Diffusers no longer exposes {helper_name}. Port the workaround and behavior probe to the "
                "replacement helper, then revalidate both upstream failures; a renamed helper alone does not "
                "make the workaround obsolete."
            )
        helpers[helper_name] = getattr(attention_dispatch, helper_name)
        monkeypatch.setattr(attention_dispatch, helper_name, helpers[helper_name])
    return helpers


def _install_fa3_probe(monkeypatch):
    backend = attention_dispatch.AttentionBackendName._FLASH_3_VARLEN_HUB
    monkeypatch.setitem(
        attention_dispatch._HUB_KERNELS_REGISTRY,
        backend,
        SimpleNamespace(kernel_fn=lambda **kwargs: _fa3_max_seqlen_k_probe(kwargs["q"], kwargs["max_seqlen_k"])),
    )


@pytest.mark.parametrize("with_mask", [False, True])
def test_diffusers_varlen_preparation_passes_unbacked_max_to_fa3(monkeypatch, original_helpers, with_mask):
    _install_fa3_probe(monkeypatch)
    _FA3_MAX_SEQLEN_K_VALUES.clear()
    torch._dynamo.reset()
    query = torch.ones((1, 4, 1, 8))
    mask = torch.tensor([[True, False, True, False]]) if with_mask else None

    try:
        # Keep scalar extraction in one graph so the probe sees the value that
        # FA3 would receive, rather than one materialized by an earlier break.
        with torch._dynamo.config.patch(capture_scalar_outputs=True):
            compiled_backend = torch.compile(
                attention_dispatch._flash_attention_3_varlen_hub,
                backend="aot_eager",
                fullgraph=True,
                dynamic=True,
            )
            compiled_backend(query, query, query, attn_mask=mask)
    finally:
        torch._dynamo.reset()

    assert _FA3_MAX_SEQLEN_K_VALUES, "The Diffusers varlen path no longer reaches the FA3 kernel interface."
    assert all(has_free_unbacked_symbols(value) for value in _FA3_MAX_SEQLEN_K_VALUES), (
        "Diffusers no longer passes an unbacked max_seqlen_k to FA3 when scalar outputs are captured; "
        "retest the real FA3 path and remove only this rationale if FA3 now succeeds. The independent Inductor "
        "cumsum failure may still require the eager boundary."
    )


def test_eager_boundary_materializes_max_before_fa3(monkeypatch, original_helpers):
    _install_fa3_probe(monkeypatch)
    _keep_varlen_attention_metadata_eager()
    _FA3_MAX_SEQLEN_K_VALUES.clear()
    torch._dynamo.reset()
    query = torch.ones((1, 4, 1, 8))

    try:
        # fullgraph=False is required because the workaround intentionally
        # creates a graph break around metadata preparation.
        with torch._dynamo.config.patch(capture_scalar_outputs=True):
            compiled_backend = torch.compile(
                attention_dispatch._flash_attention_3_varlen_hub,
                backend="aot_eager",
                fullgraph=False,
                dynamic=True,
            )
            compiled_backend(query, query, query)
    finally:
        torch._dynamo.reset()

    assert _FA3_MAX_SEQLEN_K_VALUES, "The patched Diffusers varlen path no longer reaches the FA3 kernel interface."
    assert not any(has_free_unbacked_symbols(value) for value in _FA3_MAX_SEQLEN_K_VALUES), (
        "The eager metadata boundary no longer materializes max_seqlen_k before FA3. Update the workaround for "
        "the new tracing behavior unless the unpatched real FA3 path now succeeds."
    )


@pytest.mark.parametrize("with_mask", [False, True])
def test_fa3_varlen_preparation_runs_outside_compiled_region(monkeypatch, original_helpers, with_mask):
    helper_name = (
        "_prepare_for_flash_attn_or_sage_varlen_with_mask"
        if with_mask
        else "_prepare_for_flash_attn_or_sage_varlen_without_mask"
    )
    original_prepare = original_helpers[helper_name]
    preparation_compile_states = []

    def observed_prepare(*args, **kwargs):
        preparation_compile_states.append(torch.compiler.is_compiling())
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(attention_dispatch, helper_name, observed_prepare)
    backend = attention_dispatch.AttentionBackendName._FLASH_3_VARLEN_HUB
    monkeypatch.setitem(
        attention_dispatch._HUB_KERNELS_REGISTRY,
        backend,
        SimpleNamespace(kernel_fn=lambda **kwargs: kwargs["q"]),
    )
    _keep_varlen_attention_metadata_eager()

    compiled_backend = torch.compile(
        attention_dispatch._flash_attention_3_varlen_hub,
        backend="inductor",
        fullgraph=False,
        dynamic=True,
    )
    query = torch.ones((1, 4, 1, 8))
    mask = torch.tensor([[True, False, True, False]]) if with_mask else None

    output = compiled_backend(query, query, query, attn_mask=mask)

    assert preparation_compile_states == [False], (
        "Varlen metadata preparation ran inside the compiled region. Update the workaround unless both "
        "unpatched failure canaries now pass with the real FA3 path."
    )
    torch.testing.assert_close(output, query)


def test_unpatched_varlen_path_still_needs_eager_boundary(monkeypatch, original_helpers):
    # Use the real Inductor backend here: aot_eager exercises the symbolic
    # interface above but cannot reproduce the faulty cumsum lowering pass.
    backend = attention_dispatch.AttentionBackendName._FLASH_3_VARLEN_HUB
    monkeypatch.setitem(
        attention_dispatch._HUB_KERNELS_REGISTRY,
        backend,
        SimpleNamespace(kernel_fn=lambda **kwargs: kwargs["q"]),
    )
    torch._dynamo.reset()
    compiled_backend = torch.compile(
        attention_dispatch._flash_attention_3_varlen_hub,
        backend="inductor",
        fullgraph=False,
        dynamic=True,
    )
    query = torch.ones((1, 4, 1, 8))

    try:
        compiled_backend(query, query, query)
    except InductorError as error:
        assert "FakeTensor" in str(error) and "Node" in str(error), (
            "The unpatched Diffusers varlen path now fails differently. Determine whether the cumsum bug changed "
            "or a new regression masks it, then update the workaround or canary accordingly."
        )
    else:
        pytest.fail(
            "The unpatched Diffusers varlen path now passes the Inductor cumsum canary. Revalidate the real "
            "masked and unmasked FA3 paths. Keep the shared eager boundary if the unbacked-symbol failure "
            "remains, and remove it only if both failures are fixed."
        )
    finally:
        torch._dynamo.reset()


def test_varlen_preparation_wrappers_are_installed_once(original_helpers):
    _keep_varlen_attention_metadata_eager()
    wrapped_helpers = tuple(getattr(attention_dispatch, helper_name) for helper_name in _HELPER_NAMES)
    _keep_varlen_attention_metadata_eager()

    assert tuple(getattr(attention_dispatch, helper_name) for helper_name in _HELPER_NAMES) == wrapped_helpers
