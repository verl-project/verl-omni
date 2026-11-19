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

"""Episode records for Bagel Co-RL: RFC component-1 IDs, ``GenSample`` and ``EpisodeRollout``."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from typing import Any

from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.serving import serving_rollout_provenance


def bind_episode_ids(
    *,
    dataset_task_uid: str,
    episode_uid: str | None = None,
    policy_version: int = 0,
    gen_call_id: str | None = None,
) -> dict[str, str | int]:
    """RFC component-1 IDs. FlowGRPO groups by ``gen_group_uid``, never by semantic slot."""
    episode_uid = episode_uid or str(uuid.uuid4())
    gen_call_id = gen_call_id or str(uuid.uuid4())
    return {
        "dataset_task_uid": dataset_task_uid,
        "und_group_uid": dataset_task_uid,
        "episode_uid": episode_uid,
        "policy_version": int(policy_version),
        "gen_call_id": gen_call_id,
        "gen_group_uid": gen_call_id,
    }


def gen_sample_uid(gen_call_id: str, seed_index: int) -> str:
    """Stable per-seed id within one ``generate_image`` call."""
    return f"{gen_call_id}:{seed_index}"


def conditioning_uid(token_ids: list[int]) -> str:
    """RFC §4.4.2a ``cond_uid``: content hash of the conditioning token slice."""
    digest = hashlib.sha256()
    for token in token_ids:
        digest.update(int(token).to_bytes(8, "big", signed=True))
    return digest.hexdigest()


@dataclass
class GenSample:
    """One of S FlowGRPO seeds for a single ``generate_image`` turn.

    ``rm_axes`` (UnifiedReward 1-5 axes) and ``rm_judge`` (rubric verdict) feed the judge
    observation only; ``rm_score`` stays the reward. ``cond_uid``/``cond_len`` are the RFC
    §4.4.2a conditioning identity reported by the GEN serving path.
    """

    gen_sample_uid: str
    gen_group_uid: str
    seed_index: int
    valid: bool
    prompt_token_ids: list[int]
    all_latents: Any | None = None
    timesteps: Any | None = None
    rollout_log_probs: Any | None = None
    rm_score: float | None = None
    rm_score_source: str | None = None
    image_path: str | None = None
    call_role: str = "initial"
    gen_call_index: int | None = None
    good_enough: bool | None = None
    rm_axes: dict[str, float] | None = None
    rm_judge: dict[str, Any] | None = None
    cond_uid: str | None = None
    cond_len: int = 0
    rollout_policy_version: int | None = None
    rollout_replica_id: str | int | None = None
    serving_provenance_verified: bool = False


@dataclass
class EpisodeRollout:
    """One serial UND episode (zero or more GEN calls).

    ``turns`` is in-episode ``J`` (UND policy turns); ``num_gen_calls`` is in-episode ``K``
    (``generate_image`` invocations). Invariant ``J >= K``. ``rollout_log_probs`` is π_rollout
    per response position (empty on a replay that never sampled).
    """

    und_group_uid: str
    episode_uid: str
    policy_version: int | None
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    turns: int
    rollout_log_probs: list[float] = field(default_factory=list)
    requested_policy_version: int | None = None
    rollout_replica_id: str | int | None = None
    serving_provenance_verified: bool = False
    und_serving_provenance: list[dict[str, Any]] = field(default_factory=list)
    gen_samples: list[GenSample] = field(default_factory=list)
    used_image_credit: bool = False
    forced_reflection: bool = False
    autonomous_done: bool = False
    corl_reward_version: str = "v1"
    bagel_corl_reward: dict[str, Any] = field(default_factory=dict)
    und_reward: float = 0.0
    terminal_reward: float | None = None
    judge_text: str | None = None
    stop_required: bool = False
    num_gen_calls: int = 0
    turn_trace: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """π_rollout must stay positionally aligned with the response it describes."""
        if self.rollout_log_probs and len(self.rollout_log_probs) != len(self.response_ids):
            raise ValueError(
                "Bagel Co-RL rollout_log_probs must align with response_ids "
                f"(got {len(self.rollout_log_probs)} for {len(self.response_ids)} tokens)"
            )


def gen_sample_from_row(item: Any) -> GenSample | None:
    """Rebuild a ``GenSample`` from its dict form (TQ row); passes a ``GenSample`` through."""
    if isinstance(item, GenSample):
        return item
    if not isinstance(item, dict):
        return None
    return GenSample(
        gen_sample_uid=str(item.get("gen_sample_uid", "")),
        gen_group_uid=str(item.get("gen_group_uid", "")),
        seed_index=int(item.get("seed_index", 0)),
        valid=bool(item.get("valid", False)),
        prompt_token_ids=list(item.get("prompt_token_ids") or []),
        all_latents=item.get("all_latents"),
        timesteps=item.get("timesteps"),
        rollout_log_probs=item.get("rollout_log_probs"),
        rm_score=item.get("rm_score"),
        rm_score_source=item.get("rm_score_source"),
        image_path=item.get("image_path"),
        call_role=str(item.get("call_role", "initial")),
        gen_call_index=item.get("gen_call_index"),
        good_enough=item.get("good_enough"),
        cond_uid=str(item["cond_uid"]) if item.get("cond_uid") else None,
        cond_len=int(item.get("cond_len") or 0),
        **serving_rollout_provenance(item),
    )


def strip_pixels_for_actor(row: dict[str, Any]) -> dict[str, Any]:
    """Bagel GEN trains on ``prompt_token_ids``, not pixels or ``prompt_embeds``."""
    cleaned = dict(row)
    for key in ("images", "pixel_values", "prompt_embeds", "negative_prompt_embeds"):
        cleaned.pop(key, None)
    return cleaned
