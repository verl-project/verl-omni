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

"""Native (in-process, torch-only) rollout helpers for BAGEL UniGRPO.

The training actor's live ``BagelForSFT`` module IS the sampler -- no vLLM, no
weight-sync server. Sampling on the FSDP-sharded model pays an all-gather (and, under
offload, a H2D copy) on every one of the thousands of small forwards a variable-length
AR decode issues, which is the full-FT throughput wall. Instead each rank drives a flat
full-param **bf16 replica** (``build_replica``) it re-syncs from the FSDP master once per
step (``sync_replica_from_master``); the update still runs on the FSDP model. Ported from
UniRL actor-side rollout implementation plus the standalone ``train_bagel_unigrpo`` trainer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from ..bagel_flow_grpo.bagel_sft_model import BagelForSFT

# Understanding-vision submodules unused by the reasoning->image recipe; kept frozen.
_FROZEN_VISION_SUBMODULES = ("vit_model", "connector", "vit_pos_embed")


def build_unigrpo_pipeline_kwargs(model_config, module=None) -> dict[str, Any]:
    """Assemble ``BagelUniPipeline`` kwargs from the diffusion model config.

    Reads geometry/step count from ``model_config.pipeline`` and the SDE window from
    ``model_config.algo``. The AR-decode knobs (``max_new_tokens``, ``temperature``,
    ``top_k``/``top_p``, ``shift``, ``sde_fraction``) are not part of the diffusion
    config schema and fall back to ``BagelUniPipeline`` defaults, which match the
    reference recipe. ``stop_token_ids`` is derived from the module's ``text_end_id``
    when a module is provided so the thinking chain terminates on EOS.
    """
    pipeline_cfg = model_config.pipeline
    algo = model_config.algo
    kwargs: dict[str, Any] = {
        "height": pipeline_cfg.height,
        "width": pipeline_cfg.width,
        "num_inference_steps": pipeline_cfg.num_inference_steps,
        "eta": float(getattr(algo, "noise_level", 0.8)),
        "num_sde_steps": int(getattr(algo, "sde_window_size", 3) or 3),
    }
    if module is not None:
        text_end_id = getattr(getattr(module, "config", None), "text_end_id", None)
        if text_end_id is not None:
            kwargs["stop_token_ids"] = [int(text_end_id)]
    return kwargs


def build_replica(model_path: str, device: torch.device, *, compute_dtype: torch.dtype = torch.bfloat16):
    """Plain full-param **bf16** rollout replica -- NOT FSDP-wrapped, all params frozen.

    A collective-free local sampler each rank drives on its own prompt slice;
    ``sync_replica_from_master`` refreshes it from the FSDP master once per step. It only
    ever samples (frozen, eval); the FSDP model does the training.
    """
    from ..bagel_flow_grpo.bagel_sft_model import BagelForSFT

    replica = BagelForSFT.from_pretrained(model_path, torch_dtype=compute_dtype)
    for unused in _FROZEN_VISION_SUBMODULES:
        sub = getattr(replica, unused, None)
        if sub is not None:
            for param in sub.parameters():
                param.requires_grad_(False)
    for param in replica.parameters():
        param.requires_grad_(False)
    return replica.eval().to(device)


@torch.no_grad()
def sync_replica_from_master(replica: BagelForSFT, model) -> None:
    """Copy the FSDP master's current trainable weights into the flat bf16 replica.

    Runs once per step: a bounded set of all-gathers (one per sharded trainable param via
    ``DTensor.full_tensor``) instead of the per-forward all-gathers a sharded rollout
    triggers. Frozen params are plain replicated tensors that never change after load, so
    only the ``fully_shard``-ed (trainable) params -- now ``DTensor``s -- are refreshed.
    Under CPU offload the sharded master lives on CPU, so the shard is moved to the
    replica's device before the (NCCL) all-gather; the reconstructed full param is transient.
    """
    from torch.distributed.tensor import DTensor

    rep = dict(replica.named_parameters())
    for name, param in model.named_parameters():
        if not isinstance(param.data, DTensor):
            continue
        target = rep.get(name)
        if target is None:
            continue
        shard = param.data
        if shard.device.type != target.device.type:
            shard = shard.to(target.device)
        target.data.copy_(shard.full_tensor().to(target.dtype))


__all__ = ["build_replica", "sync_replica_from_master", "build_unigrpo_pipeline_kwargs"]
