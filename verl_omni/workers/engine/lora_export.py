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
"""The one LoRA export contract shared by the FSDP engines.

verl-omni keeps the policy decisions (which adapter, which mode, when to sync);
this module owns the export mechanics both engines used to duplicate with
silently different dtype handling. ``get_per_tensor_param(base_sync_done=...)``
stays the frozen public entry on each engine; it delegates here.

Wire contract returned to the sync orchestrator:
- ``merge`` mode: merged full weights, ``peft_config=None`` — the rollout gets a
  plain full-weight update.
- adapter mode: ``base_sync_done=True`` yields adapter-only A/B tensors (training
  dtype preserved) plus a ``peft_config`` dict; ``base_sync_done=False`` yields
  full base weights (``replace_lora_wrapper`` names on the AR path) plus the dict.
- no LoRA: full weights, ``peft_config=None``.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, Iterator

import torch
from torch.distributed.tensor import DTensor
from verl.utils.device import get_device_id
from verl.utils.fsdp_utils import merged_lora_context, normalize_peft_param_name, replace_lora_wrapper
from verl.utils.model import convert_weight_keys

from verl_omni.utils.config import resolve_lora_config
from verl_omni.utils.fsdp_utils import collect_lora_params


def _cast_for_sync(tensor: torch.Tensor) -> torch.Tensor:
    """The single dtype policy for full-weight exports: gather DTensors to one
    device and cast floating-point gathers to bf16; leave everything else alone.
    Adapter-only exports never pass through here (training dtype is preserved)."""
    if isinstance(tensor, DTensor):
        tensor = tensor.to(get_device_id(), non_blocking=True).full_tensor()
        if tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
            tensor = tensor.to(torch.bfloat16, non_blocking=True)
    return tensor


def _merged_lora_stream(module, *, key_prefix: str = "", offload_fn: Callable[[], None] | None = None):
    """Stream merged (base + LoRA) weights for rollout weight sync.

    ``state_dict()`` returns tensors that alias the live FSDP parameter storage,
    and ``merged_lora_context`` restores the un-merged base weights when it
    exits. The context therefore must stay open until the consumer has
    materialized every tensor: DTensor gathers and the plain-tensor ``.clone()``
    produce copies, so yielded tensors remain valid after the restore. Consuming
    a state_dict captured inside the context after the context has exited would
    silently send base weights without the adapters.
    """
    try:
        with merged_lora_context(module, backup_adapters=True):
            params = normalize_peft_param_name(module.state_dict())
            params = convert_weight_keys(params, getattr(module, "_fsdp_wrapped_module", module))
            for name, param in params.items():
                # clone: plain tensors also alias module storage, and bucketed
                # senders may flush after the restore has already run
                tensor = _cast_for_sync(param) if isinstance(param, DTensor) else param.detach().clone()
                yield f"{key_prefix}{name}", tensor
    finally:
        if offload_fn is not None:
            offload_fn()


def export_lora_for_sync(
    module: torch.nn.Module,
    *,
    model_config,
    base_sync_done: bool,
    adapter_name: str | None = None,
    layered_summon: bool = False,
    is_diffusers: bool,
    key_prefix: str = "",
    adapter_context: Callable[[], object] | None = None,
    offload_fn: Callable[[], None] | None = None,
    stream_transform: Callable[[Iterator], Iterator] | None = None,
) -> tuple[Iterator[tuple[str, torch.Tensor]], dict | None]:
    """Yield ``(name, tensor)`` for a rollout sync plus ``peft_config``-or-``None``.

    Args:
        module: The (possibly FSDP-wrapped) training module.
        model_config: Model config carrying ``lora`` (the ``merge`` flag) and,
            for diffusers engines, ``fsdp_layer_prefixes``.
        base_sync_done: ``False`` only until the rollout holds the base weights.
        adapter_name: PEFT adapter to export (``"default"``, ``"old"``); ``None``
            means ``"default"``.
        layered_summon: Summon one FSDP unit at a time during collect.
        is_diffusers: Diffusion-pipeline export: skips the AR base-layer rename,
            selects the diffusers layered walker, and honors
            ``fsdp_layer_prefixes``.
        key_prefix: Prefix prepended to every exported name (``"transformer."``
            for diffusion pipelines, ``""`` for AR models).
        adapter_context: Engine hook opening writable adapter selection around
            the collect (diffusers ``use_adapter``).
        offload_fn: Engine memory management. Called after the params are
            materialized on the non-merged paths, and when the merged stream
            finishes.
        stream_transform: Engine post-processing (QAT quantization) applied to
            the non-merged export stream; merged exports bypass it, matching
            upstream verl.

    Returns:
        ``(iterator, peft_config_dict)`` where ``peft_config_dict`` is ``None``
        unless adapter-mode tensors are being shipped.
    """
    peft_model = getattr(module, "_fsdp_wrapped_module", module)
    has_lora = hasattr(peft_model, "peft_config")
    # Resolved unconditionally (like the pre-refactor engines did): the resolver
    # fail-fast-validates the LoRA config surface even for full-weight runs.
    merge_lora = resolve_lora_config(model_config).merge
    adapter = adapter_name or "default"
    peft_config = None

    if has_lora:
        if merge_lora:
            if adapter_name not in (None, "default"):
                # merged_lora_context merges the active ("default") adapter only;
                # silently exporting it for a named rollout_adapter would sync the
                # wrong policy.
                raise ValueError(
                    f"model.lora.merge=True exports the active 'default' adapter only; "
                    f"got rollout_adapter={adapter_name!r}."
                )
            return _merged_lora_stream(module, key_prefix=key_prefix, offload_fn=offload_fn), None
        peft_config = peft_model.peft_config.get(adapter, None)
        adapter_ctx = adapter_context() if adapter_context is not None else nullcontext()
        if is_diffusers:
            # Passed through exactly as the engine did: an empty list empties the
            # prefix walker, which then falls back to the full LoRA dump.
            layer_prefixes = model_config.fsdp_layer_prefixes
        else:
            layer_prefixes = ("transformer_blocks.",)
        with adapter_ctx:
            params = collect_lora_params(
                module=module,
                layered_summon=layered_summon,
                base_sync_done=base_sync_done,
                is_diffusers=is_diffusers,
                adapter_name=adapter,
                layer_prefixes=layer_prefixes,
            )
        if not base_sync_done and not is_diffusers:
            # The rollout model is PEFT-wrapped: plain leaf names must land inside
            # the LoRA-wrapped module. Diffusion pipelines load plain names.
            params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
    else:
        params = module.state_dict()

    params = convert_weight_keys(params, peft_model)

    if offload_fn is not None:
        offload_fn()

    if peft_config is not None and base_sync_done:
        per_tensor_param = params.items()
    else:
        per_tensor_param = ((name, _cast_for_sync(param)) for name, param in params.items())

    if key_prefix:
        per_tensor_param = ((f"{key_prefix}{name}", tensor) for name, tensor in per_tensor_param)

    if stream_transform is not None:
        per_tensor_param = stream_transform(per_tensor_param)

    peft_config_dict = peft_config.to_dict() if peft_config is not None else None
    return per_tensor_param, peft_config_dict
