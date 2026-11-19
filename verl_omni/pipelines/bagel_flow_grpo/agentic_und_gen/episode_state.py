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

"""Mutable per-episode state for the serial loop: the token transcript and the serving-provenance ledger."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.gen_tool import BagelGenerateImageTool
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.samples import GenSample
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.serving import serving_rollout_provenance
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.tool_calls import IM_END, TOOL_RESPONSE_TEMPLATE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpisodeSpec:
    """Immutable knobs of one ``run_serial_episode`` call."""

    und_decode: Callable[..., Any]
    generate_tool: BagelGenerateImageTool
    score_fn: Callable[[list[GenSample]], Any] | None
    build_judge_text_fn: Callable[[list[GenSample]], Any] | None
    max_und_turns: int
    non_image_reward: float | None
    good_enough_threshold: float | None
    good_enough_reduction: str
    corl_reward_version: str
    fixed_gen_prompt: str
    planned_images: int
    active_user_prompt: str


class EpisodeTranscript:
    """Response tokens, mask, π_rollout log-probs and the turn trace, kept in lockstep."""

    def __init__(
        self,
        *,
        prompt_ids: list[int],
        tokenizer: Any | None,
        max_context_tokens: int | None,
        max_response_tokens: int | None,
    ):
        self.prompt_ids = list(prompt_ids)
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens
        self.max_response_tokens = max_response_tokens
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.log_probs: list[float] = []
        self.trace: list[dict[str, Any]] = []

    @property
    def context_len(self) -> int:
        """Tokens the engine would be asked to prefill for the next turn."""
        return len(self.prompt_ids) + len(self.response_ids)

    def budget_spent(self) -> bool:
        """True once the episode has used up ``max_response_tokens`` or ``max_context_tokens``."""
        if self.max_response_tokens is not None and len(self.response_ids) >= self.max_response_tokens:
            return True
        return self.max_context_tokens is not None and self.context_len >= self.max_context_tokens

    def encode(self, text: str, *, fallback_ids: list[int] | None = None) -> list[int]:
        """Tokenize ``text``; without a tokenizer, the caller-provided fallback ids are used."""
        if self.tokenizer is not None:
            return list(self.tokenizer.encode(text, add_special_tokens=False))
        return list(fallback_ids or [])

    def append_response(self, tokens: Any, mask_value: int, log_probs: Any | None = None) -> None:
        """Extend the episode with ``tokens`` (mask 1 = policy, 0 = forced/environment)."""
        token_list = [int(t) for t in tokens]
        self.response_ids.extend(token_list)
        self.response_mask.extend([int(mask_value)] * len(token_list))
        score = None if log_probs is None else [float(v) for v in log_probs]
        if score is not None and len(score) != len(token_list):
            raise ValueError(
                "Bagel Co-RL UND pass: rollout_log_probs must align with response_ids "
                f"(got {len(score)} log-probs for {len(token_list)} tokens)"
            )
        self.log_probs.extend(score if score is not None else [0.0] * len(token_list))

    def append_env_turn(self, body: str, policy_text: str, fallback_ids: list[int], *, turn: int) -> None:
        """Answer the policy's tool call with a ``<tool_response>`` user turn (mask 0)."""
        close = "" if policy_text.rstrip().endswith(IM_END) else IM_END
        obs_ids = self.encode(close + TOOL_RESPONSE_TEMPLATE.format(body=body), fallback_ids=fallback_ids)
        if obs_ids:
            self.append_response(obs_ids, 0)
        self.trace.append({"record": "env_obs", "turn": turn, "text": body})

    def truncate_to_budget(self, episode_tag: str) -> None:
        """Drop tail tokens an environment turn landed past ``max_response_tokens``."""
        limit = self.max_response_tokens
        if limit is None or len(self.response_ids) <= limit:
            return
        logger.warning(
            "bagel_corl_episode episode=%s response %d > response_length %d; truncating",
            episode_tag,
            len(self.response_ids),
            limit,
        )
        del self.response_ids[limit:]
        del self.response_mask[limit:]
        del self.log_probs[limit:]


class ServingLedger:
    """Serving provenance of every UND decode; a synchronous episode may not mix policy versions."""

    def __init__(self):
        self.und_version: int | None = None
        self.replicas: set[str | int] = set()
        self.provenance: list[dict[str, Any]] = []
        self.all_verified = True

    def observe_und(self, step: dict[str, Any], *, response_start: int) -> None:
        """Record one UND decode's provenance and the response span it will occupy."""
        provenance = serving_rollout_provenance(step)
        version = provenance["rollout_policy_version"]
        if version is not None:
            if self.und_version is not None and version != self.und_version:
                raise ValueError("Synchronous episode mixed observed UND serving policy versions")
            self.und_version = version
        replica = provenance["rollout_replica_id"]
        if replica is not None:
            self.replicas.add(replica)
        self.all_verified &= provenance["serving_provenance_verified"]
        self.provenance.append(
            {
                **provenance,
                "response_start": response_start,
                "response_end": response_start + len(step.get("token_ids") or []),
            }
        )

    def check_gen(self, samples: list[GenSample]) -> None:
        """GEN samples must come from the same policy version the UND decodes did."""
        for sample in samples:
            version = sample.rollout_policy_version
            if version is not None and self.und_version is not None and version != self.und_version:
                raise ValueError("Synchronous episode mixed GEN and UND serving policy versions")

    @property
    def replica_id(self) -> str | int | None:
        """The single replica that served the episode, or ``None`` when ambiguous."""
        return next(iter(self.replicas)) if len(self.replicas) == 1 else None

    @property
    def verified(self) -> bool:
        """Every decode carried verified provenance from one replica."""
        return bool(self.provenance) and self.all_verified and len(self.replicas) == 1
