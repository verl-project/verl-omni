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
"""Regional compilation helpers for diffusion models."""

import logging

import torch
from diffusers.models import attention_dispatch
from verl.workers.config import FSDPEngineConfig

from verl_omni.workers.config import DiffusionModelConfig

logger = logging.getLogger(__name__)


def _keep_varlen_attention_metadata_eager() -> None:
    """Keep data-dependent varlen metadata preparation out of compiled graphs.

    With PyTorch 2.13 and Diffusers 0.40, compiling Qwen-Image's real FA3
    varlen path exposes two independent failures:

    1. For dense queries, Diffusers builds cumulative lengths as
       ``full((batch_size,), seq_len_q).cumsum()``. With dynamic shapes,
       PyTorch 2.13's ``pointless_cumsum_replacement`` Inductor pass mistakes
       the symbolic fill value for a regular scalar while rewriting that
       expression, then fails while multiplying a FakeTensor by a symbolic
       Node.
    2. For masked keys, Diffusers derives ``max_seqlen_k`` from Tensor data
       with ``seqlens_k.max().item()``. When Dynamo captures scalar outputs to
       keep this preparation in the graph, that value becomes an unbacked
       SymInt. FA3's fake backward uses it in Python control flow to select
       window, kernel, and workspace behavior, which raises
       ``GuardOnDataDependentSymNode``. With ``fullgraph=False`` Dynamo may
       instead break earlier, but that incidental break is not an FA3 contract.

    ``fullgraph=False`` does not avoid the Inductor lowering failure by
    itself. This eager boundary bypasses the faulty cumsum rewrite and
    materializes ``max_seqlen_q/k`` as concrete Python integers. Dynamo then
    resumes tracing, so token packing, the FA3 custom op, and the rest of the
    repeated transformer block remain eligible for regional compilation.
    The same eager boundary protects against both failures and must remain
    while either one is reproducible.

    The patched names are private Diffusers APIs. Remove this workaround once
    upstream fixes both symbolic cumsum lowering and FA3's handling or
    contract for data-dependent maximum sequence lengths, then revalidate the
    real masked and unmasked FA3 paths without this boundary, including with
    ``fullgraph=True``. Otherwise revisit it whenever those helpers change.
    """
    for helper_name in (
        "_prepare_for_flash_attn_or_sage_varlen_with_mask",
        "_prepare_for_flash_attn_or_sage_varlen_without_mask",
    ):
        prepare_varlen = getattr(attention_dispatch, helper_name)
        if getattr(prepare_varlen, "_verl_omni_compiler_disabled", False):
            continue

        prepare_varlen = torch.compiler.disable(prepare_varlen)
        prepare_varlen._verl_omni_compiler_disabled = True
        setattr(attention_dispatch, helper_name, prepare_varlen)


def _maybe_compile_repeated_blocks(
    module: torch.nn.Module,
    model_config: DiffusionModelConfig,
    engine_config: FSDPEngineConfig,
) -> None:
    """Regionally compile repeated diffusion blocks before FSDP2 mutates them."""
    if not model_config.use_regional_compile:
        return
    if engine_config.strategy == "fsdp":
        # Regional compilation has not been validated with this engine's FSDP1
        # path, so reject the combination as a whole. ``use_orig_params=True``
        # is a necessary prerequisite for eventually enabling FSDP1 with
        # torch.compile, but satisfying it alone would not make the currently
        # untested integration supported.
        raise NotImplementedError(
            "Diffusion regional torch.compile does not yet support FSDP1 because that integration has not "
            "been validated. FSDP1 would also require use_orig_params=True. Use strategy=fsdp2 or disable "
            "model.use_regional_compile."
        )
    if engine_config.strategy != "fsdp2":
        raise NotImplementedError(
            f"Diffusion regional torch.compile does not support strategy={engine_config.strategy!r}; use FSDP2."
        )

    # Regional compilation has not been validated with Ulysses SP, so reject
    # the combination as a whole. Ulysses places SP collectives inside each
    # compiled block. Enabling it requires distributed validation that Dynamo,
    # Inductor, and AOTAutograd preserve the SP all-to-all participation and
    # ordering in the original forward, activation-checkpoint recomputation,
    # and backward while FSDP2 independently schedules parameter all-gathers
    # and gradient reduce-scatters. Rank-specific graph breaks, guard misses,
    # or recompilation could otherwise make ranks enter SP and DP collectives
    # in different orders and hang the job.
    if engine_config.ulysses_sequence_parallel_size != 1:
        raise NotImplementedError(
            "Diffusion regional torch.compile does not yet support Ulysses SP because that distributed "
            "integration has not been validated. Use ulysses_sequence_parallel_size=1 or disable "
            "model.use_regional_compile."
        )
    _keep_varlen_attention_metadata_eager()
    options = dict(model_config.regional_compile_options or {})
    logger.info("Compiling repeated %s blocks with options=%s", type(module).__name__, options)
    module.compile_repeated_blocks(**options)
