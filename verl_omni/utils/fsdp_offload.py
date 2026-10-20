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
"""Order host transfers before FSDP2 reads CPU shards to rebuild padding."""

from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.nn import Module


def _may_repad(tensor, local) -> bool:
    from torch.distributed.tensor import Replicate, Shard

    shards = [(i, p) for i, p in enumerate(tensor.placements) if isinstance(p, Shard)]
    # The omni engine uses one FSDP shard dimension, optionally with HSDP
    # replication. Be conservative for other layouts instead of depending on
    # private FSDP parameter state or guessing combined TP/FSDP padding.
    if len(shards) != 1 or any(type(p) not in (Shard, Replicate) for p in tensor.placements):
        return True
    mesh_dim, placement = shards[0]
    size = tensor.shape[placement.dim]
    shard_count = tensor.device_mesh.size(mesh_dim)
    padded_size = (size + shard_count - 1) // shard_count
    return local.shape[placement.dim] != padded_size


def install_fsdp2_cpu_transfer_guard(model: Module, device_type: str, *, pin_memory: bool = False) -> bool:
    """Wait before CPU repadding/pinning in this model's recursive ``_apply``.

    Install after FSDP2 wrapping and before loading state. ``pin_memory`` must
    match the wrapping CPUOffloadPolicy. This protects root-model conversions,
    including inherited offload/checkpoint calls, without patching PyTorch's
    classes or replacing parameter storage. Direct child conversions are not
    intercepted. Conversion callbacks must obey the caller's current stream.

    Only nonempty accelerator DTensors that may need CPU repadding or pinning
    query a stream; only their CPU conversions wait. Uniform shards retain
    asynchronous D2H unless pinning requires a CPU copy. Returns whether a
    guard was installed; repeated installation is a no-op.
    """
    if device_type in ("cpu", "meta") or getattr(model, "_verl_omni_cpu_transfer_guard", False):
        return False

    import torch
    from torch.distributed.fsdp import FSDPModule
    from torch.distributed.tensor import DTensor

    if not isinstance(model, FSDPModule):
        raise TypeError("CPU transfer guard requires an FSDP2 root module")
    original_apply = model._apply

    def guarded_apply(self, fn, *args, **kwargs):
        def convert(tensor):
            if not isinstance(tensor, DTensor):
                return fn(tensor)
            # Keep the actual local source alive until the conversion and any
            # required wait finish, before Module._apply replaces its storage.
            source = tensor.to_local()
            if source.device.type in ("cpu", "meta") or source.numel() == 0:
                return fn(tensor)
            repad = _may_repad(tensor, source)
            if not (repad or pin_memory):
                return fn(tensor)
            stream = torch.get_device_module(source.device).current_stream(source.device)
            result = fn(tensor)
            local_result = result.to_local()
            if local_result.device.type == "cpu" and (repad or (pin_memory and not local_result.is_pinned())):
                # Waiting after model.to() is too late: each child FSDP _apply
                # has already reset its shards by then. Wait on the copy's
                # source stream here, before the recursive callback returns.
                stream.synchronize()
            return result

        return original_apply(convert, *args, **kwargs)

    model._apply = MethodType(guarded_apply, model)
    model._verl_omni_cpu_transfer_guard = True
    return True
