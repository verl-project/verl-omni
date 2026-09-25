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

"""Reusable parameter grouping, explicit FSDP2 wrapping and shard-aware clipping."""

import math

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from verl.utils.device import get_device_id


def optimizer_parameters(module, config):
    """Build ordered name-substring LR groups, retaining the default ungrouped path.

    Only trainable parameters enter explicit groups. The first matching key wins;
    unmatched trainable parameters keep the base learning rate. Optimizer class,
    betas and other options remain the responsibility of verl's optimizer builder.
    """
    overrides = getattr(config, "param_group_lrs", None)
    if not overrides:
        return module.parameters()
    for key, lr in overrides.items():
        if not key or not math.isfinite(float(lr)) or float(lr) < 0:
            raise ValueError(f"Invalid parameter LR group {key!r}: {lr}")
    buckets = {key: [] for key in overrides}
    base = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        key = next((key for key in overrides if key in name), None)
        (base if key is None else buckets[key]).append(param)
    groups = [{"params": base, "lr": float(config.lr)}] if base else []
    groups.extend({"params": params, "lr": float(overrides[key])} for key, params in buckets.items() if params)
    return groups


def shard_fsdp2_units(module, units, fsdp_kwargs):
    """Apply engine-owned FSDP2 policies to adapter-selected units in their given order."""
    from torch.distributed.fsdp import fully_shard

    if not units or len({id(unit) for unit in units}) != len(units):
        raise ValueError("Explicit FSDP2 units must be nonempty and unique")
    owned = {id(submodule) for submodule in module.modules()}
    if any(id(unit) not in owned for unit in units):
        raise ValueError("Explicit FSDP2 units must belong to the model")
    # Frozen buffers/leaves may be used directly and are not moved by fully_shard.
    module.to(get_device_id())
    for unit in units:
        fully_shard(unit, **fsdp_kwargs)
    # A missed leaf would silently leave trainable replicated weights unsynchronized.
    unsharded = [
        name for name, param in module.named_parameters() if param.requires_grad and not isinstance(param, DTensor)
    ]
    if unsharded:
        raise ValueError(f"Explicit FSDP2 units leave trainable parameters unsharded: {unsharded}")


def clip_grad_norm_sharded_(parameters, max_norm, mesh):
    """Clip leaf-sharded gradients using all_gather over the FSDP mesh communicator.

    Avoid per-DTensor reductions on additional communicators. Replicated dimensions
    contribute once to the global squared norm, including unsharded replicated grads.
    Partial placements are rejected: these must be reduced before clipping.
    """
    parameters = list(parameters)
    device = next((p.device for p in parameters if p.requires_grad), torch.device("cpu"))
    local_sq = torch.zeros((), dtype=torch.float32, device=device)
    grads = [p.grad for p in parameters if p.requires_grad and p.grad is not None]
    for grad in grads:
        replication = mesh.size()
        local = grad
        if isinstance(grad, DTensor):
            if grad.device_mesh != mesh:
                raise ValueError("Gradient mesh does not match the FSDP engine mesh")
            replication = 1
            for dim, placement in enumerate(grad.placements):
                if placement.is_partial():
                    raise ValueError("Partial gradients must be reduced before clipping")
                if placement.is_replicate():
                    replication *= mesh.size(dim)
            local = grad.to_local()
        local_sq += local.detach().float().square().sum() / replication
    # Reduce one mesh dimension at a time (supports both FSDP and hybrid sharding).
    for dim in range(mesh.ndim):
        group = mesh.get_group(dim)
        if dist.get_world_size(group) > 1:
            gathered = [torch.empty_like(local_sq) for _ in range(dist.get_world_size(group))]
            dist.all_gather(gathered, local_sq, group=group)
            local_sq = torch.stack(gathered).sum()
    total_norm = local_sq.sqrt()
    if torch.isfinite(total_norm):
        coefficient = (float(max_norm) / (total_norm + 1e-6)).clamp(max=1.0)
        for grad in grads:
            local = grad.to_local() if isinstance(grad, DTensor) else grad
            local.mul_(coefficient.to(local.device))
    return total_norm
