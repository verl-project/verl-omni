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
"""Advantage gates over a padded batch: per-token credit masks and baseline exclusion.

Two generic agent-loop ``extra_fields`` columns drive :func:`apply_advantage_gates`:
``advantage_mask`` (one weight per response position) scales a row's advantage,
and ``advantage_exclude`` drops a row from its GRPO group's baseline and zeroes
its advantage. MiniCPM-o 4.5 under duplex GRPO (``omni_duplex_agent``) is the
only pipeline that carries either; every other pipeline carries neither, and
the function leaves its batch untouched.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from verl.protocol import DataProto
from verl.trainer.ppo import core_algos

__all__ = ["apply_advantage_gates"]


def apply_advantage_gates(
    data: DataProto,
    extra_fields: Sequence[Mapping[str, Any] | None],
    *,
    adv_estimator: str,
    norm_adv_by_std_in_grpo: bool,
) -> dict[str, float]:
    """Apply the agent loop's advantage gates to a batch verl already scored.

    Args:
        data: The padded batch after verl's advantage computation:
            ``advantages``, ``returns``, ``response_mask`` and
            ``token_level_rewards`` of shape ``(bs, L)`` in ``batch``, and the
            ``uid`` group index in ``non_tensor_batch``.
        extra_fields: One ``extra_fields`` mapping per row of ``data``, or
            ``None`` where the row carries none (a padded row). A row with
            ``advantage_exclude`` leaves its group's baseline and takes
            advantage and return 0; the group's remaining rows are rescored
            against their own mean through verl's GRPO estimator. A row's
            ``advantage_mask`` scales its advantage per response position (a
            missing tail is 0); rows without one are untouched.
        adv_estimator: ``config.algorithm.adv_estimator``. The baseline
            exclusion is implemented for ``grpo`` only.
        norm_adv_by_std_in_grpo: verl's flag, forwarded to the rescored groups.

    Returns:
        ``advantage/gated_token_share``: the fraction of response-masked
        positions the per-token gate zeroed; ``advantage/excluded_sample_share``:
        the fraction of rows carrying extras that are excluded.

    Raises:
        NotImplementedError: When a row carries ``advantage_exclude`` under an
            estimator other than ``grpo``.
    """
    if len(extra_fields) != len(data):
        raise ValueError(f"{len(extra_fields)} extra_fields rows for a batch of {len(data)} rows")
    carried = [fields if isinstance(fields, Mapping) else None for fields in extra_fields]
    excluded = np.array(
        [bool(fields.get("advantage_exclude", False)) if fields is not None else False for fields in carried]
    )
    masks = [fields.get("advantage_mask") if fields is not None else None for fields in carried]
    returns = data.batch["returns"]
    response_mask = data.batch["response_mask"]
    advantages = data.batch["advantages"]
    if any(mask is not None for mask in masks):
        # verl's GRPO leaves advantages and returns the same tensor; only advantages take the gate.
        advantages = data.batch["advantages"] = advantages.clone()

    if excluded.any():
        if adv_estimator != core_algos.AdvantageEstimator.GRPO:
            raise NotImplementedError(
                f"advantage_exclude is implemented for grpo only, got adv_estimator={adv_estimator!r}"
            )
        uid = data.non_tensor_batch["uid"]
        clean = ~excluded
        excluded_rows = torch.as_tensor(excluded, device=advantages.device)
        advantages[excluded_rows] = 0
        returns[excluded_rows] = 0
        token_level_rewards = data.batch["token_level_rewards"]
        for group in np.unique(uid[excluded]):
            rows = np.flatnonzero((uid == group) & clean)
            if not rows.size:
                continue
            advantages[rows], returns[rows] = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards[rows],
                response_mask[rows],
                uid[rows],
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )

    gated = torch.zeros_like(response_mask, dtype=torch.bool)
    for row, mask in enumerate(masks):
        if mask is None:
            continue
        gate = torch.zeros(advantages.shape[1], dtype=advantages.dtype, device=advantages.device)
        length = min(len(mask), gate.shape[0])
        gate[:length] = torch.as_tensor(list(mask[:length]), dtype=gate.dtype, device=gate.device)
        advantages[row] *= gate
        gated[row] = (response_mask[row] > 0) & (gate == 0)
    masked_positions = int(response_mask.sum())
    sample_count = sum(fields is not None for fields in carried)
    return {
        "advantage/gated_token_share": int(gated.sum()) / masked_positions if masked_positions else 0.0,
        "advantage/excluded_sample_share": float(excluded.sum()) / sample_count if sample_count else 0.0,
    }
