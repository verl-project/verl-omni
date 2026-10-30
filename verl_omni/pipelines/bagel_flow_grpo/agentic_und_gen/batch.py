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
"""Row ownership and valid-unit reduction for the Bagel joint actor."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu


def episode_numeric_id(episode_uid: str) -> int:
    """Return a stable int64 transport ID; reserve -1 for padding."""
    return int.from_bytes(hashlib.blake2b(str(episode_uid).encode(), digest_size=8).digest(), "big") & ((1 << 63) - 1)


def configure_corl_trainability(module: torch.nn.Module, *, freeze_und: bool, freeze_gen: bool) -> None:
    """Freeze an ablation lane before FSDP wrapping, including conditioning gradients."""
    if freeze_und and freeze_gen:
        raise ValueError("freeze both lanes is unsupported; use corl_no_optimizer_step for the frozen control")
    for name, param in module.named_parameters():
        if freeze_gen if "moe_gen" in name else freeze_und:
            param.requires_grad_(False)


def validate_episode_ownership(
    episode_ids: torch.Tensor, parents: torch.Tensor, padding: list[bool] | None = None
) -> torch.Tensor:
    """Validate the global gather before row IDs are split across DP ranks."""
    ids = episode_ids.detach().cpu().reshape(-1).clone()
    if padding is not None:
        if len(padding) != len(ids):
            raise ValueError("episode padding flags must match the global batch")
        ids[torch.tensor(padding, dtype=torch.bool)] = -1
    valid = ids[ids >= 0]
    if valid.unique().numel() != valid.numel():
        raise ValueError("global episode IDs must be unique")
    if not torch.isin(parents.cpu(), valid).all():
        raise ValueError("GEN parent episode is absent from the global UND batch")
    return ids


def gradient_snapshot(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Snapshot local trainable gradient shards between the two backwards."""
    result = {}
    for name, param in module.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad.to_local() if hasattr(param.grad, "to_local") else param.grad
            result[name] = grad.detach().clone()
    return result


def gradient_delta_norm(
    module: torch.nn.Module, before: dict[str, torch.Tensor], *, dp_group, collective_device: torch.device
) -> float:
    """Full FSDP gradient norm for this lane's increment (before global clipping)."""
    squared = torch.zeros((), dtype=torch.float64, device=collective_device)
    for name, param in module.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad.to_local() if hasattr(param.grad, "to_local") else param.grad
            delta = grad.detach().float()
            if name in before:
                delta = delta - before[name].float()
            squared += delta.double().square().sum().to(collective_device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(squared, group=dp_group)
    return float(squared.sqrt().item())


def select_gen_rows_for_episodes(gen_batch: TensorDict, episode_ids: torch.Tensor) -> TensorDict:
    """Keep only the current optimizer minibatch's episodes and their prompt rows."""
    ids = episode_ids.detach().cpu().reshape(-1)
    ids = ids[ids >= 0]
    if ids.unique().numel() != ids.numel():
        raise ValueError("UND episode IDs must be unique in an optimizer minibatch")
    row_ids = gen_batch["gen_row_id"]
    if row_ids.unique().numel() != row_ids.numel():
        raise ValueError("GEN row IDs must be unique")
    indices = torch.isin(gen_batch["parent_episode_id"].cpu(), ids).nonzero().flatten()
    out = gen_batch[indices.to(row_ids.device)].clone()
    for key in gen_batch.keys():
        value = tu.get_non_tensor_data(gen_batch, key, default=None)
        if isinstance(value, list | tuple | np.ndarray) and len(value) == len(gen_batch):
            selected = np.empty(len(indices), dtype=object)
            for j, idx in enumerate(indices.tolist()):
                selected[j] = value[idx]
            tu.assign_non_tensor(out, **{key: selected})
    return out


def pad_gen_shard(local_gen: TensorDict, *, target_rows: int, template: TensorDict | None = None) -> TensorDict:
    """Pad with finite, scheduler-safe rows that contribute zero valid transitions."""
    if target_rows < len(local_gen):
        raise ValueError("target_rows cannot discard GEN rows")
    if target_rows == len(local_gen):
        return local_gen
    template = local_gen[:1] if len(local_gen) else template
    if template is None or not len(template):
        raise ValueError("empty GEN rank needs a real transition template")
    real_n = len(local_gen)
    # TensorDict concatenation of non-tensor arrays is not row-aware. Build explicitly.
    out = TensorDict({}, batch_size=[target_rows])
    for key, value in local_gen.items():
        if isinstance(value, torch.Tensor):
            pad = template[key][:1].expand(target_rows - real_n, *value.shape[1:]).clone().to(value.device)
            out[key] = torch.cat([value, pad])
        else:
            values = tu.get_non_tensor_data(local_gen, key, default=None)
            base = tu.get_non_tensor_data(template, key, default=None)
            if isinstance(values, list | tuple | np.ndarray) and len(values) == real_n:
                arr = np.empty(target_rows, dtype=object)
                for i in range(target_rows):
                    arr[i] = values[i] if i < real_n else base[0]
                tu.assign_non_tensor(out, **{key: arr})
            else:
                tu.assign_non_tensor(out, **{key: values})
    out["gen_valid"][real_n:] = 0
    out["gen_transition_valid"][real_n:] = 0
    out["gen_row_id"][real_n:] = -1
    return out


def prepare_gen_shard(
    gen_batch: TensorDict, episode_ids: torch.Tensor, *, dp_group, micro_size: int, collective_device: torch.device
) -> TensorDict | None:
    """Select unique ownership then equalize FSDP work and globally normalize valid units."""
    local = select_gen_rows_for_episodes(gen_batch, episode_ids)
    count = float((local["gen_valid"][:, None] * local["gen_transition_valid"]).sum().item())
    totals = torch.tensor([count], dtype=torch.float64, device=collective_device)
    maximum = torch.tensor([len(local)], dtype=torch.long, device=collective_device)
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    dp_size = torch.distributed.get_world_size(dp_group) if distributed else 1
    if distributed:
        torch.distributed.all_reduce(totals, group=dp_group)
        torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX, group=dp_group)
    if totals.item() == 0:
        return None
    target = ((int(maximum.item()) + micro_size - 1) // micro_size) * micro_size
    out = pad_gen_shard(local, target_rows=target, template=gen_batch[:1])
    tu.assign_non_tensor(
        out, corl_loss_normalized=True, loss_normalizer=float(totals.item()) / dp_size, corl_dp_size=dp_size
    )
    return out
