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
"""Dual-lane batch gather for ``bagel_corl_sync``: UND rows from the bus, GEN rows via ``child_gen_keys``."""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.episode_metrics import (
    aggregate_episode_metrics,
    aggregate_reward_breakdown_metrics,
    und_record_field_keys,
)

logger = logging.getLogger(__name__)


def extra_info(batch) -> dict:
    """The batch's ``extra_info`` carrier dict, created on the batch when it is missing."""
    extra = getattr(batch, "extra_info", None)
    if extra is None:
        extra = {}
        if hasattr(batch, "extra_info"):
            batch.extra_info = extra
    return extra


def bagel_role_histogram(tags) -> dict[str, int]:
    """Count a ``KVBatchMeta`` row-tag list by ``bagel_role`` (+ padding), for step diagnostics."""
    counts: dict[str, int] = {}
    for tag in tags or []:
        if not isinstance(tag, dict):
            continue
        role = str(tag.get("bagel_role") or "untagged")
        if tag.get("is_padding"):
            role += ":pad"
        counts[role] = counts.get(role, 0) + 1
    return dict(sorted(counts.items()))


def _index_column(col, i: int):
    """Index a stacked TQ column (Tensor / NonTensorStack / list) at position ``i``."""
    if col is None:
        return None
    try:
        value = col[i]
    except (TypeError, IndexError, KeyError):
        return None
    try:
        from tensordict import NonTensorData
    except ImportError:  # pragma: no cover - tensordict is a hard dependency here
        return value
    return value.data if isinstance(value, NonTensorData) else value


def _rows_from_columnar_td(td, keys: list[str]) -> dict[str, dict] | None:
    """Rows of a columnar TensorDict, or ``None`` when it cannot be read that way."""
    if not (hasattr(td, "items") and hasattr(td, "get") and not isinstance(td, dict | list)):
        return None
    if not callable(getattr(td, "keys", None)):
        return None
    try:
        field_names = [str(k) for k in td.keys()]
        out = {
            str(key): {"fields": {name: _index_column(td[name], i) for name in field_names}}
            for i, key in enumerate(keys)
        }
    except (TypeError, KeyError, IndexError):
        return None
    return out or None


def _rows_from_dict(data: dict, keys: list[str]) -> dict[str, dict]:
    """Rows of a keyed ``{key: row}`` or columnar ``{field: [values]}`` dict."""
    if keys and all(k in keys for k in data.keys()):
        out: dict[str, dict] = {}
        for key in keys:
            row = data.get(key)
            if row is None:
                continue
            if isinstance(row, dict) and "fields" in row:
                out[str(key)] = row
            else:
                out[str(key)] = {"fields": row if isinstance(row, dict) else {"value": row}}
        return out
    got_keys = [str(k) for k in (data.get("keys") if data.get("keys") is not None else keys)]
    out = {}
    for i, key in enumerate(got_keys):
        row = {}
        for field_name, col in data.items():
            if field_name == "keys":
                continue
            try:
                row[field_name] = col[i]
            except (TypeError, IndexError, KeyError):
                row[field_name] = None
        out[key] = {"fields": row}
    return out


def normalize_tq_kv_get_result(data, keys: list[str]) -> dict[str, dict]:
    """Normalize ``transfer_queue.kv_batch_get`` output into ``{key: {"fields": row}}``.

    Fail loud on any shape not recognized so a silent ``{}`` can never drop the GEN lane.
    """
    if data is None:
        return {}
    columnar = _rows_from_columnar_td(data, keys)
    if columnar is not None:
        return columnar
    if isinstance(data, dict):
        return _rows_from_dict(data, keys)
    if isinstance(data, list | tuple):
        return {
            str(key): {"fields": row if isinstance(row, dict) else {"value": row}}
            for key, row in zip(keys, data, strict=False)
            if row is not None
        }
    raise ValueError(
        f"bagel_corl: unrecognized transfer_queue.kv_batch_get return type {type(data)!r}; "
        "refusing to silently drop the GEN lane"
    )


def merge_child_gen_keys(row, child_keys):
    """Give a fetched row the ``child_gen_keys`` column when its own blob omitted it."""
    if not isinstance(row, dict) or not child_keys or row.get("child_gen_keys"):
        return row
    return {**row, "child_gen_keys": list(child_keys)}


def rows_in_batch_key_order(batch, data: dict):
    """Normalize a per-key mapping (``{tq_key: row}``) into batch-key order."""
    keys = [str(k) for k in (getattr(batch, "keys", None) or [])]
    if keys and all(k in data for k in keys):
        return [data[k] for k in keys]
    return list(data.values())


def extra_fields_from_tq(batch):
    """Fetch ``extra_fields`` (+ sibling ``child_gen_keys``) for a ``KVBatchMeta``; ``None`` otherwise."""
    if not hasattr(batch, "keys") or not hasattr(batch, "partition_id"):
        return None
    try:
        import transfer_queue as tq

        data = tq.kv_batch_get(
            keys=list(batch.keys), partition_id=batch.partition_id, select_fields=["extra_fields", "child_gen_keys"]
        )
    except (KeyError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("bagel_corl UND extra_fields TQ fetch failed") from exc
    if isinstance(data, dict):
        return data["extra_fields"] if "extra_fields" in data else rows_in_batch_key_order(batch, data)
    if "extra_fields" in data.keys():
        rows = list(data["extra_fields"])
        if "child_gen_keys" in data.keys():
            child_col = list(data["child_gen_keys"])
            rows = [
                merge_child_gen_keys(row, child_col[i] if i < len(child_col) else None) for i, row in enumerate(rows)
            ]
        return rows
    if "child_gen_keys" in data.keys():
        return list(data["child_gen_keys"])
    return data


def _non_padding_indices(batch, n_rows: int) -> list[int] | range:
    """Row indices excluding ``is_padding`` tags; padding rows are deep copies of a real episode."""
    tags = list(getattr(batch, "tags", None) or [])
    if len(tags) == n_rows:
        return [i for i, tag in enumerate(tags) if not tag.get("is_padding", False)]
    return range(n_rows)


def _record_from_extra_row(extra_row, child_keys=None) -> dict:
    """One ``{"fields": ...}`` record from an ``extra_fields`` blob, flattening a nested blob."""
    mapping = extra_row if isinstance(extra_row, dict) else {}
    fields = dict(mapping)
    if "child_gen_keys" not in fields and isinstance(mapping.get("extra_fields"), dict):
        fields.update(mapping["extra_fields"])
    if child_keys is not None:
        fields = merge_child_gen_keys(fields, child_keys)
    return {"fields": fields}


def und_records_from_batch(batch, extra: dict) -> list[dict]:
    """UND episode rows as ``{"fields": ...}`` records, whatever carrier the bus handed over."""
    if extra.get("und_batch"):
        return [{"fields": row} for row in extra["und_batch"]]
    ntb = getattr(batch, "non_tensor_batch", None) or {}
    if ntb.get("child_gen_keys") is not None:
        records = []
        for i in range(len(ntb["child_gen_keys"])):
            fields = {}
            for key, col in ntb.items():
                try:
                    fields[key] = col[i]
                except (TypeError, IndexError, KeyError):
                    continue
            records.append({"fields": fields})
        return records
    if ntb.get("extra_fields") is not None:
        extras = list(ntb["extra_fields"])
        child_col = ntb.get("child_gen_keys")
        child_col = list(child_col) if child_col is not None else None
        return [
            _record_from_extra_row(
                extras[idx], child_col[idx] if child_col is not None and idx < len(child_col) else None
            )
            for idx in _non_padding_indices(batch, len(extras))
        ]
    extras = extra_fields_from_tq(batch)
    if extras is None:
        return []
    keys = list(getattr(batch, "keys", []) or [])
    if keys and len(extras) != len(keys):
        raise RuntimeError(
            f"bagel_corl UND extra_fields fetch returned {len(extras)} row(s) for {len(keys)} key(s); "
            "row<->key alignment is what the GEN gather and the J/K aggregation rely on"
        )
    return [_record_from_extra_row(extras[idx]) for idx in _non_padding_indices(batch, len(extras))]


def fetch_gen_records_by_keys(batch, extra: dict, und_records: list[dict]) -> dict[str, dict]:
    """Lookup GEN TQ rows by ``child_gen_keys``. Returns key -> ``{"fields": ...}``."""
    if isinstance(extra.get("gen_by_key"), dict):
        return {str(k): ({"fields": v} if "fields" not in v else v) for k, v in extra["gen_by_key"].items()}
    keys: list[str] = []
    for rec in und_records:
        fields = rec.get("fields") or rec
        for child in fields.get("child_gen_keys") or []:
            if str(child) not in keys:
                keys.append(str(child))
    if not keys:
        return {}
    if not hasattr(batch, "partition_id"):
        raise RuntimeError(
            f"bagel_corl: batch has child_gen_keys={len(keys)} but no partition_id; cannot fetch GEN TQ rows"
        )
    import transfer_queue as tq

    out = normalize_tq_kv_get_result(tq.kv_batch_get(keys=keys, partition_id=batch.partition_id), keys)
    missing = [k for k in keys if k not in out]
    if missing:
        raise RuntimeError(
            f"bagel_corl GEN TQ fetch incomplete: {len(out)}/{len(keys)} rows (first missing: {missing[:3]})"
        )
    return out


def _publish_episode_metrics(extra: dict, und_records: list[dict]) -> None:
    """Fold the batch's J/K aggregation and reward breakdown onto ``extra`` (first writer wins)."""
    aggregated = aggregate_episode_metrics(und_records)
    breakdown = aggregate_reward_breakdown_metrics(und_records)
    if not breakdown:
        logger.warning(
            "bagel_corl reward breakdown EMPTY: und_records=%d row_field_keys=%s",
            len(und_records),
            und_record_field_keys(und_records, limit=24),
        )
    aggregated.update(breakdown)
    for key, value in aggregated.items():
        if extra.get(key) is None:
            extra[key] = value


def gen_batch_from_step(batch, extra: dict) -> list:
    """GEN rows for FlowGRPO: dual-lane ``child_gen_keys`` or ``extra['gen_batch']`` only."""
    if extra.get("gen_batch") is not None:
        return list(extra["gen_batch"])
    from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.transfer_queue import split_und_gen_metas

    und_records = und_records_from_batch(batch, extra)
    has_children = bool(und_records) and any("child_gen_keys" in (r.get("fields") or r) for r in und_records)
    if has_children:
        gen_by_key = fetch_gen_records_by_keys(batch, extra, und_records)
        _, gen_batch = split_und_gen_metas(und_records, gen_by_key)
        _publish_episode_metrics(extra, und_records)
        extra["gen_batch"] = gen_batch
        return gen_batch
    bagel = extra.get("bagel_corl") if isinstance(extra.get("bagel_corl"), dict) else {}
    meta = getattr(batch, "meta_info", None) or {}
    bagel_meta = meta.get("bagel_corl") if isinstance(meta, dict) else {}
    if bagel.get("gen_batch") or (isinstance(bagel_meta, dict) and bagel_meta.get("gen_batch")):
        raise RuntimeError(
            "bagel_corl: GEN rows found only on nested bagel_corl.gen_batch / meta_info; "
            "dual-lane extra['gen_batch'] or child_gen_keys is required (legacy nested path removed)."
        )
    logger.info(
        "bagel_corl gen_batch empty (GEN lane will skip): und_records=%d has_child_gen_keys=%s "
        "has_tq_handle=%s ntb_keys=%s batch=%s",
        len(und_records),
        has_children,
        hasattr(batch, "keys") and hasattr(batch, "partition_id"),
        sorted((getattr(batch, "non_tensor_batch", None) or {}).keys()),
        type(batch).__name__,
    )
    return []


def gen_dispatch_divisor(actor_rollout_wg: Any, config) -> int:
    """Row-count divisor the GEN lane's worker-group dispatch requires (``lcm(dp_size, mini_batch*n)``)."""
    dp_size = 1
    if actor_rollout_wg is not None and hasattr(actor_rollout_wg, "_query_dispatch_info"):
        info = actor_rollout_wg._query_dispatch_info("actor")
        if isinstance(info, dict):
            dp_size = max(info.values()) + 1 if info else 1
        elif isinstance(info, list | tuple | set):
            dp_size = max(info) + 1 if info else 1
        else:
            dp_size = int(info) + 1 if info is not None else 1
    rollout_n = int(config.actor_rollout_ref.rollout.n)
    mini = int(config.actor_rollout_ref.actor.ppo_mini_batch_size) * rollout_n
    return max(1, math.lcm(int(dp_size), int(mini)))


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
