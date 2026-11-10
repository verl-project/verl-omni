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
"""UND-lane GRPO advantage with a per-group std floor (``algorithm.adv_estimator=grpo_std_floor``).

Why a third option next to verl's ``norm_adv_by_std_in_grpo`` True/False:

* ``True`` divides every group by its own std, so an 8-sibling group whose rewards differ by
  0.012 (image seed + judge noise) becomes a full +/-1 push. That is why the Bagel Co-RL
  recipe turned it off.
* ``False`` (Dr.GRPO) leaves the raw spread. On ``bagel_corl_dualrm_3567`` that is
  ``critic/advantages`` in [-0.11, +0.08] per step. Adam is scale-invariant on the *total*
  gradient, but the actor loss is ``pg + kl_loss_coef * KL(pi || ref)``, and the raw scale
  shrinks the policy-gradient term ~10x against the KL anchor. The run sat at
  ``actor/und/ref_kl`` ~0.01 from step 51 to step 164 (first five steps 0.0071-0.0107, last
  five 0.0072-0.0168): a KL equilibrium, not learning.

``grpo_std_floor`` is ``(r - mean) / max(std, floor)``. A real spread (std above the floor)
gets the GRPO unit scale; a near-tie group is divided by the floor instead of its own tiny
std, so its push shrinks in proportion to how much of a tie it is. With ``floor=0.05`` the
0.012-spread example lands at |A| ~0.2 instead of 1.0, while the typical 0.06-0.08 spread
observed on this task lands at ~1.0.

verl dispatches any ``adv_estimator`` that is not ``gae``/``grpo`` through
``core_algos.get_adv_estimator_fn`` with ``token_level_rewards``, ``response_mask``,
``index`` and ``config``; this module registers under that name so the V1 trainer picks it
up with no trainer change. ``config.adv_std_floor`` is read from ``OmniAlgoConfig``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import numpy as np
import torch
from verl.trainer.ppo.core_algos import register_adv_est

GRPO_STD_FLOOR = "grpo_std_floor"
DEFAULT_ADV_STD_FLOOR = 0.05


def resolve_adv_std_floor(config: Any) -> float:
    """``algorithm.adv_std_floor`` as a non-negative float (missing -> default)."""
    raw = None
    if config is not None:
        getter = getattr(config, "get", None)
        raw = getter("adv_std_floor", None) if callable(getter) else getattr(config, "adv_std_floor", None)
    if raw is None:
        return DEFAULT_ADV_STD_FLOOR
    floor = float(raw)
    if floor < 0:
        raise ValueError(f"algorithm.adv_std_floor must be >= 0, got {floor}")
    return floor


@register_adv_est(GRPO_STD_FLOOR)
def compute_grpo_std_floor_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[Any] = None,
    **_: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Outcome GRPO advantage, normalised by ``max(group_std, floor)``.

    Singleton groups follow verl's GRPO: mean 0, std 1 (the raw score is the advantage).
    Returns ``(advantages, returns)`` with identical tensors, like the other outcome estimators.
    """
    floor = resolve_adv_std_floor(config)
    scores = token_level_rewards.sum(dim=-1)

    id2score: dict[Any, list[torch.Tensor]] = defaultdict(list)
    id2mean: dict[Any, torch.Tensor] = {}
    id2scale: dict[Any, torch.Tensor] = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx, group in id2score.items():
            if len(group) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2scale[idx] = torch.tensor(1.0)
                continue
            group_t = torch.stack(group)
            id2mean[idx] = torch.mean(group_t)
            id2scale[idx] = torch.clamp(torch.std(group_t), min=floor)
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2scale[index[i]] + epsilon)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores
