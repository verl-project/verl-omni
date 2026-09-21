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
    """Keep data-dependent varlen-attention metadata out of compiled graphs.

    This avoids an Inductor symbolic-cumsum lowering failure and prevents
    unbacked maximum sequence lengths from reaching FA3 fake backward. Remove
    the boundary only after both paths work, then revalidate masked and
    unmasked FA3 with ``fullgraph=True``. The patched Diffusers helper names
    are private and must be rechecked when Diffusers changes.
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
        raise NotImplementedError(
            "Diffusion regional torch.compile does not yet support FSDP1 because that integration has not "
            "been validated. FSDP1 would also require use_orig_params=True. Use strategy=fsdp2 or disable "
            "model.use_regional_compile."
        )
    if engine_config.strategy != "fsdp2":
        raise NotImplementedError(
            f"Diffusion regional torch.compile does not support strategy={engine_config.strategy!r}; use FSDP2."
        )

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
