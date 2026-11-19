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

"""Batch aggregation of per-episode UND rows into step-level metrics (J/K, cond cache, reward breakdown)."""

from __future__ import annotations

from typing import Any

import numpy as np

#: Reward dims republished on every *training* update (validation already logs them as ``val-aux/*``).
TRAIN_REWARD_DIM_KEYS: tuple[str, ...] = (
    "reward_reflect",
    "reward_plan",
    "reward_format",
    "reward_protocol",
    "reward_tool",
    "reward_result",
    "reward_image",
    "reward_ca",
    "reward_done",
    "reward_tool_call",
)

_COND_SUM_KEYS = ("gen/cond_cache_hits", "gen/cond_cache_misses", "gen/cond_cache_bypassed")
_COND_MEAN_KEYS = ("gen/cond_recompute_ratio", "gen/prompt_embed_cache_hit_rate", "gen/cond_amortization")
_RATIO_FLAG_KEYS = ("und/no_image_credit", "gen/skipped_no_groups", "episode/K_zero")


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def record_fields(record: Any) -> dict[str, Any]:
    """Fields of one TQ row, accepting the ``{"fields": ...}`` wrapper or a bare dict."""
    if not isinstance(record, dict):
        return {}
    fields = record.get("fields")
    if isinstance(fields, dict):
        return fields
    return {} if "fields" in record else record


def und_record_field_keys(records: list[Any], *, limit: int | None = None) -> list[str]:
    """Field names the first UND row carries, for diagnosing an empty reward breakdown."""
    if not records:
        return []
    keys = sorted(record_fields(records[0]))
    return keys[:limit] if limit is not None else keys


def aggregate_episode_metrics(records: list[Any]) -> dict[str, float]:
    """Batch-aggregate per-episode metrics: counts are summed, ratios are batch means (RFC §4.4.4 R2)."""
    j_values: list[float] = []
    k_values: list[float] = []
    dropped = 0.0
    flags: dict[str, list[float]] = {key: [] for key in _RATIO_FLAG_KEYS}
    cond_sums: dict[str, float] = dict.fromkeys(_COND_SUM_KEYS, 0.0)
    cond_means: dict[str, list[float]] = {key: [] for key in _COND_MEAN_KEYS}
    for rec in records:
        fields = rec.get("fields") if isinstance(rec, dict) else rec
        if not isinstance(fields, dict):
            fields = {}
        metrics = fields.get("bagel_corl_metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        j = fields.get("episode_J", metrics.get("episode/J"))
        k = fields.get("episode_K", metrics.get("episode/K"))
        if j is not None:
            j_values.append(float(j))
        if k is not None:
            k_values.append(float(k))
        dropped += float(metrics.get("gen/dropped_incomplete_groups", 0.0) or 0.0)
        for key, values in flags.items():
            if key in metrics:
                values.append(float(metrics[key] or 0.0))
        for key in cond_sums:
            if key in metrics:
                cond_sums[key] += float(metrics[key] or 0.0)
        for key, values in cond_means.items():
            if key in metrics:
                values.append(float(metrics[key] or 0.0))
    aggregated: dict[str, float] = {}
    if j_values:
        aggregated["episode/J"] = _mean(j_values)
    if k_values:
        aggregated["episode/K"] = _mean(k_values)
    aggregated["gen/dropped_incomplete_groups"] = float(dropped)
    for key, values in flags.items():
        if values:
            aggregated[key] = _mean(values)
    if any(value > 0.0 for value in cond_sums.values()):
        aggregated.update(cond_sums)
    for key, values in cond_means.items():
        if values:
            aggregated[key] = _mean(values)
    return aggregated


def _reward_breakdown(fields: dict[str, Any]) -> dict[str, Any] | None:
    """The episode's per-dim reward blob, flat on the row or nested under ``extra_fields``."""
    nested = fields.get("extra_fields")
    nested_blob = nested.get("reward_extra_info") if isinstance(nested, dict) else None
    for candidate in (fields.get("reward_extra_info"), nested_blob):
        if isinstance(candidate, dict):
            return candidate
    return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _episode_score(value: Any) -> float | None:
    """The episode scalar from a per-token score vector, mirroring the trainer's ``sum(-1)``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return _as_float(value)
    if isinstance(value, str | bytes) or value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return float(array.sum()) if array.size else None


def _episode_reward(fields: dict[str, Any]) -> float | None:
    """The episode's trained reward, read the way the trainer reads it."""
    for key in ("rm_scores", "token_level_scores"):
        if key in fields:
            value = _episode_score(fields.get(key))
            if value is not None:
                return value
    return _as_float(fields.get("und_reward"))


def aggregate_reward_breakdown_metrics(records: list[Any]) -> dict[str, float]:
    """Batch-aggregate the per-episode reward breakdown and the ``used_image_credit`` split."""
    dim_values: dict[str, list[float]] = {key: [] for key in TRAIN_REWARD_DIM_KEYS}
    with_image: list[float] = []
    without_image: list[float] = []
    for record in records:
        fields = record_fields(record)
        if not fields:
            continue
        breakdown = _reward_breakdown(fields)
        if breakdown is not None:
            for key in TRAIN_REWARD_DIM_KEYS:
                value = _as_float(breakdown.get(key))
                if value is not None:
                    dim_values[key].append(value)
        score = _episode_reward(fields)
        credit = fields.get("used_image_credit")
        if score is None or credit is None:
            continue
        (with_image if bool(credit) else without_image).append(score)
    aggregated: dict[str, float] = {}
    for key, values in dim_values.items():
        if not values:
            continue
        metric = key.removeprefix("reward_")
        aggregated[f"bagel_corl/reward_dim/{metric}/mean"] = _mean(values)
        aggregated[f"bagel_corl/reward_dim/{metric}/min"] = float(min(values))
        aggregated[f"bagel_corl/reward_dim/{metric}/max"] = float(max(values))
    for name, values in (("valid", with_image), ("no_image", without_image)):
        if values:
            aggregated[f"bagel_corl/reward_{name}/mean"] = _mean(values)
            aggregated[f"bagel_corl/reward_{name}/count"] = float(len(values))
    return aggregated
