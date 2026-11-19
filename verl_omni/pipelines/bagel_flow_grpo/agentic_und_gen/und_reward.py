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

"""Episode-level UND reward from the GEN samples' trusted RM scores (v1 image credit, v2 envelope)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.samples import EpisodeRollout, GenSample

TRUSTED_RM_SOURCE = "bagel_corl_tool_rm"


def trusted_image_reward_fields(samples, *, num_gen_calls, autonomous_done=False, forced_reflection=False):
    """Build the harness-only V2 envelope from RM-marked samples, never transcript text.

    Group first appearance is call chronology; seed order is explicit and stable.
    A failed/missing image score rejects the episode instead of fabricating signal.
    """
    calls: dict[str, dict[int, float]] = {}
    call_indices: dict[str, Any] = {}
    for sample in samples:
        if (
            not sample.valid
            or sample.rm_score_source != TRUSTED_RM_SOURCE
            or sample.rm_score is None
            or not math.isfinite(float(sample.rm_score))
        ):
            raise ValueError("V2 requires finite trusted tool RM scores for every seed")
        uid = str(sample.gen_group_uid)
        call_index = getattr(sample, "gen_call_index", None)
        if uid in call_indices and call_indices[uid] != call_index:
            raise ValueError("inconsistent call index in V2 group")
        call_indices[uid] = call_index
        group = calls.setdefault(uid, {})
        if sample.seed_index in group:
            raise ValueError("duplicate seed index in V2 call")
        group[sample.seed_index] = float(sample.rm_score)
    if len(calls) != num_gen_calls:
        raise ValueError("V2 trusted call count does not match generated calls")
    if calls and all(index is not None for index in call_indices.values()):
        if sorted(call_indices.values()) != list(range(num_gen_calls)):
            raise ValueError("V2 call indices must be unique and contiguous")
        calls = dict(sorted(calls.items(), key=lambda pair: call_indices[pair[0]]))
    seed_scores = [[row[i] for i in sorted(row)] for row in calls.values()]
    means = [sum(row) / len(row) for row in seed_scores]
    first, final, best = (means[0], means[-1], max(means)) if means else (None, None, None)
    return {
        "origin": TRUSTED_RM_SOURCE,
        "version": "v2",
        "call_means": means,
        "call_seed_scores": seed_scores,
        "image_reward_first": first,
        "image_reward_final": final,
        "image_reward_best": best,
        "image_reward_gain": final - first if means else None,
        "image_reward_source": TRUSTED_RM_SOURCE,
        "num_gen_calls": num_gen_calls,
        "autonomous_done": bool(autonomous_done),
        "forced_reflection": bool(forced_reflection),
        "image_reward_objective": final - 0.01 * max(0, num_gen_calls - 1) if means else None,
    }


def image_credit_reward(samples: list[GenSample]) -> float | None:
    """RFC ``und/no_image_credit`` image credit: the **final** pass plus the rewrite gain."""
    ordered: dict[str, list[float]] = {}
    for sample in samples or []:
        if sample.valid and sample.rm_score is not None:
            ordered.setdefault(str(sample.gen_group_uid), []).append(float(sample.rm_score))
    if not ordered:
        return None
    calls = [float(np.mean(scores)) for scores in ordered.values()]
    final = calls[-1]
    improvement = max(0.0, final - calls[0]) if len(calls) > 1 else 0.0
    return float(final + min(1.0, improvement))


def episode_und_reward(episode: EpisodeRollout) -> tuple[float, bool]:
    """RFC ``und/no_image_credit``: ``(reward, no_image_credit)`` carried on the UND row."""
    if episode.corl_reward_version == "v2" and episode.terminal_reward is not None:
        return float(episode.terminal_reward), not episode.used_image_credit
    if episode.corl_reward_version == "v2":
        value = episode.bagel_corl_reward.get("image_reward_objective")
        return (float(value), False) if value is not None else (float(episode.und_reward), True)
    image_credit = image_credit_reward(episode.gen_samples)
    if image_credit is not None:
        return image_credit, False
    return float(episode.und_reward), True


def und_episode_rm_scores(response_mask: Any, reward: float) -> list[float]:
    """Per-token UND episode reward, placed on the last *trainable* response position."""
    if hasattr(response_mask, "ndim") and response_mask.ndim != 1:
        raise ValueError(
            "Bagel Co-RL UND rm_scores expects a 1-D response mask for one episode, got shape "
            f"{tuple(response_mask.shape)}"
        )
    mask = list(response_mask)
    valid = [index for index, flag in enumerate(mask) if bool(flag)]
    if not valid:
        raise ValueError(
            "Bagel Co-RL UND rm_scores: response_mask has no trainable position, so the episode "
            f"reward {float(reward)} could never reach the advantage computation"
        )
    scores = [0.0] * len(mask)
    scores[valid[-1]] = float(reward)
    return scores
