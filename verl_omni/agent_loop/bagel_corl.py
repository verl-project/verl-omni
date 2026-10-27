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

"""Serial Bagel UND→GEN Co-RL (Joint-Training) agent loop (RFC phase A / PR1)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
import zlib
from functools import partial
from pathlib import Path
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register
from verl.utils.tokenizer import normalize_token_ids

from verl_omni.agent_loop.bagel_corl_gen_serve import build_gen_sampling_params, stash_gen_row_from_diffusion_output
from verl_omni.agent_loop.bagel_corl_lib import (  # noqa: F401
    GENERATE_IMAGE_TOOL_SCHEMA,
    JUDGE_IMAGE_TOOL_SCHEMA,
    UND_STOP_SEQUENCES,
    BagelGenerateImageTool,
    EpisodeRollout,
    GenSample,
    cond_reuse_metrics,
    conditioning_uid,
    count_degenerate_und_turns,
    generate_retrying_engine_wake,
    resolve_task_type,
    run_serial_episode,
    serving_rollout_provenance,
    strip_reflection_tail,
    trusted_image_reward_fields,
    turn_histogram,
    und_turn_max_tokens,
)
from verl_omni.agent_loop.bagel_corl_lib import (
    with_reflection_system_prompt as _with_reflection_system_prompt,
)
from verl_omni.agent_loop.bagel_corl_rm import get_bagel_rm_gen_handle, make_rm_score_fn
from verl_omni.agent_loop.composite_agent_loop import CompositeAgentLoopWorker
from verl_omni.agent_loop.rpco_turn_protocol import derive_good_enough_from_scores
from verl_omni.agent_loop.utils import derive_rollout_seed
from verl_omni.tools.trajectory.artifacts import get_latest_generate_prompt_for_active_rollout
from verl_omni.tools.trajectory.context import (
    get_active_user_prompt,
    reset_active_trajectory_relpath,
    reset_active_user_prompt,
    set_active_trajectory_relpath,
    set_active_user_prompt,
)
from verl_omni.tools.trajectory.hydra_env import agentic_get, agentic_scorer_knobs_from_config
from verl_omni.tools.trajectory.paths import (
    build_trajectory_relpath,
    project_relpath,
    resolve_rollout_images_root,
    resolve_run_dir,
)
from verl_omni.utils.agentic.image_gen_rollout_parse import last_user_prompt

logger = logging.getLogger(__name__)


def _log_engine_asleep_retry(
    _logger, attempt: int, delay: float, exc: Exception, *, lane: str, step: Any = None
) -> None:
    """Report a re-issue of a decode the engine rejected mid-wake.

    Logged at WARNING, not DEBUG: this is a transient that the retry schedule is expected to
    absorb, but a *run* of these means wake latency is outgrowing the schedule and the rollout is
    stalling on it. ``lane`` separates the AR (``und``) and diffusion (``gen``) pools, which wake
    on different paths.
    """
    _logger.warning(
        "bagel_corl %s lane: engine rejected a decode mid-wake (step=%s attempt=%d, retrying in %.1fs) -- "
        "the wake call returns when the request is dispatched, not when the memory is mapped: %s",
        lane,
        step,
        attempt + 1,
        delay,
        exc,
    )


def _messages_are_text_only(messages: Any) -> bool:
    """True when every message carries plain text (no image/video/audio parts).

    The UND prompt is rendered from ``raw_prompt`` only when it is text: a multimodal
    message would need the dataset's own tokenizer run (image placeholder expansion),
    so those prompts keep the ids they came with.
    """
    if not isinstance(messages, list):
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        content = msg.get("content")
        if content is None or isinstance(content, str):
            continue
        if isinstance(content, list):
            if any(isinstance(part, dict) and part.get("type") not in (None, "text") for part in content):
                return False
            continue
        return False
    return True


def _template_renders_hermes_tools(tokenizer: Any) -> bool:
    """True when the checkpoint's chat template knows the Hermes tool-call grammar.

    Gating the schema injection on the template (rather than on a recipe flag) keeps a
    non-Hermes UND model from being handed a ``<tools>`` block it cannot render:
    the published Bagel tokenizer's template contains both ``<tools>`` and
    ``<tool_call>``, which is exactly the grammar ``parse_hermes_tool_call`` reads.
    """
    template = getattr(tokenizer, "chat_template", None) or ""
    return "<tools>" in template and "<tool_call>" in template


def _inject_tools_schema_enabled() -> bool:
    """``BAGEL_CORL_INJECT_TOOLS=0`` opts out (diagnostics / non-Hermes UND models)."""
    return os.getenv("BAGEL_CORL_INJECT_TOOLS", "1") not in ("0", "false", "False", "")


# Episode-level reduction over the S per-seed ``good_enough`` flags. SoT is
# ``actor_rollout_ref.rollout.agent.good_enough_reduction``; unknown values fail
# loud rather than silently degrading to best-of-S (see ``bagel_corl_lib``).
_GOOD_ENOUGH_REDUCTIONS = frozenset({"any", "all", "mean"})

# RFC §5: knobs the agent loop must see but must NOT invent a value for.
_BAGEL_AGENT_REQUIRED_KNOBS = ("gen_samples_per_call", "max_generate_passes", "max_und_turns")


def resolve_bagel_agent_knobs(agent_cfg: Any) -> dict[str, Any]:
    """Resolve the ``bagel_multiturn_agent`` knobs, failing loud on missing ones."""
    missing = [key for key in _BAGEL_AGENT_REQUIRED_KNOBS if agent_cfg.get(key) is None]
    if missing:
        raise ValueError(
            "bagel_multiturn_agent requires actor_rollout_ref.rollout.agent."
            f"{', '.join(missing)} (RFC §5 knob SoT, no code default)"
        )
    reduction = str(agent_cfg.get("good_enough_reduction", "any"))
    if reduction not in _GOOD_ENOUGH_REDUCTIONS:
        raise ValueError(
            "bagel_multiturn_agent actor_rollout_ref.rollout.agent.good_enough_reduction "
            f"must be one of {sorted(_GOOD_ENOUGH_REDUCTIONS)}, got {reduction!r}"
        )
    return {
        # S = FlowGRPO seeds under one generate_image turn. Not episode GEN-turn count K.
        "gen_samples_per_call": int(agent_cfg.get("gen_samples_per_call")),
        "max_generate_passes": int(agent_cfg.get("max_generate_passes")),
        "max_und_turns": int(agent_cfg.get("max_und_turns")),
        "good_enough_reduction": reduction,
    }


def episode_to_agent_output(
    episode: EpisodeRollout,
    *,
    timing: dict[str, Any] | None = None,
    relpath: str | None = None,
) -> AgentLoopOutput:
    """Project a finished episode onto the ``AgentLoopOutput`` the v1 TQ path expects."""
    extra = {
        "text_encoder_responses": "",
        "prompt_ids": episode.prompt_ids,
        "response_ids": episode.response_ids,
        "response_mask": episode.response_mask,
        "gen_samples": episode.gen_samples,
        "und_group_uid": episode.und_group_uid,
        "episode_uid": episode.episode_uid,
        "used_image_credit": episode.used_image_credit,
        "bagel_corl_reward": dict(episode.bagel_corl_reward),
        "corl_reward_version": episode.corl_reward_version,
        "autonomous_done": episode.autonomous_done,
        "forced_reflection": episode.forced_reflection,
        **{k: v for k, v in episode.bagel_corl_reward.items() if k.startswith("image_reward_")},
        "turns": episode.turns,
        "num_gen_calls": episode.num_gen_calls,
        "und_reward": episode.und_reward,
        "policy_version": episode.policy_version,
        "rollout_policy_version": episode.policy_version,
        "requested_policy_version": episode.requested_policy_version,
        "rollout_replica_id": episode.rollout_replica_id,
        "serving_provenance_verified": episode.serving_provenance_verified,
        "und_serving_provenance": list(episode.und_serving_provenance),
        "min_global_steps": episode.policy_version,
        "max_global_steps": episode.policy_version,
        "bagel_role_timing": dict(timing or {}),
        "bagel_corl_r2": dict(episode.r2_metrics),
        "llm_all_log_probs": None,
        # Stamped so ``dump_bagel_corl_episode_images`` can drop the GEN PNGs into the
        # same ``step_XXXXXX/sample_<id>.01`` folder this episode's trajectory was
        # dumped under. Without it those images were copied to a synthesised
        # ``sample_0.00`` path that matched no trajectory (or, before that, not
        # copied at all).
        "trajectory_relpath": relpath or "",
    }
    return AgentLoopOutput(
        prompt_ids=episode.prompt_ids,
        response_ids=episode.response_ids,
        response_mask=episode.response_mask,
        # ``None`` when the episode sampled nothing (replay/stub): the trainer then gets no
        # rollout distribution for that row instead of an all-zero one it would trust.
        response_logprobs=list(episode.rollout_log_probs) or None,
        num_turns=episode.turns,
        metrics=AgentLoopMetrics(),
        reward_score=None,
        extra_fields=extra,
    )


def _trace_dump_enabled() -> bool:
    """``BAGEL_CORL_TRACE_DUMP=0`` opts out of the per-episode trajectory dump.

    On by default: the dump is a few KB of text per episode and it is the only record
    of a rollout that never reached ``generate_image`` (K=0). Opting out matters for
    very long runs on a shared filesystem, not for correctness.
    """
    return os.getenv("BAGEL_CORL_TRACE_DUMP", "1") not in ("0", "false", "False", "")


def materialized_image_path(source: Any, image_relpath: str) -> str:
    """Name the ``rollout_images/<image_relpath>/`` copy of a GEN scratch PNG.

    ``dump_episode_images`` copies each scratch file to that folder under its own basename,
    so the destination is a pure function of the two and can be named before the copy runs.
    Rendered project-relative so the record stays valid when the checkout moves.
    """
    return project_relpath(resolve_rollout_images_root() / image_relpath / Path(str(source)).name)


def _materialized_turn_trace(turn_trace: list[dict[str, Any]], image_relpath: str) -> list[dict[str, Any]]:
    """Copy ``turn_trace`` with each ``gen_call`` naming the materialized images.

    The scratch locations move to ``source_image_paths`` for provenance. The episode's own
    records are left untouched: the GEN lane still reads the scratch files.
    """
    out = []
    for record in turn_trace:
        if record.get("record") == "gen_call":
            sources = list(record.get("image_paths") or [])
            record = {
                **record,
                "image_paths": [materialized_image_path(p, image_relpath) for p in sources],
                "source_image_paths": sources,
            }
        out.append(record)
    return out


def trace_ground_truth(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Copy only reward ground truth from dataset metadata, never policy prose.

    Do not export arbitrary rollout kwargs (which can include service secrets).
    Preserve missing labels/references as missing so offline diagnosis can detect
    them. ``resolve_task_type`` uses the same extra_info precedence as rollout.
    """
    extra = kwargs.get("extra_info")
    extra = extra if isinstance(extra, dict) else {}
    reward_model = kwargs.get("reward_model")
    reward_model = reward_model if isinstance(reward_model, dict) else {}
    raw = kwargs.get("ground_truth", reward_model.get("ground_truth", extra.get("ground_truth", {})))
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            raw = {}
    raw = raw if isinstance(raw, dict) else {}
    keys = {
        "task_type",
        "reference_subtasks",
        "reference_steps",
        "expected_num_images",
        "plan_expected",
        "user_request",
    }
    keys.update("w_" + dim for dim in ("reflect", "plan", "format", "protocol", "tool", "result", "image", "done"))
    ground_truth = {key: value for key, value in extra.items() if key in keys}
    ground_truth.update({key: value for key, value in raw.items() if key in keys})
    task_type = resolve_task_type(extra_info=extra, ground_truth=raw)
    if task_type is not None:
        ground_truth["task_type"] = task_type
    return ground_truth


def dump_episode_trace(
    *,
    episode: EpisodeRollout,
    relpath: str,
    step: Any = None,
    sample_index: Any = None,
    user_prompt: str = "",
    prompt_ids: list[int] | None = None,
    tokenizer: Any | None = None,
    image_relpath: str | None = None,
    ground_truth: dict[str, Any] | None = None,
) -> str | None:
    """Write one finished Co-RL episode to ``rollout_trajectories/<relpath>``."""
    if not _trace_dump_enabled():
        return None
    try:
        step_dir = Path(relpath).parent
        name = Path(relpath).name
        trajectory_dir = resolve_run_dir() / "rollout_trajectories" / step_dir
        trajectory_dir.mkdir(parents=True, exist_ok=True)

        turn_trace = _materialized_turn_trace(episode.turn_trace, image_relpath or relpath)
        turns = [r for r in turn_trace if r.get("record") == "und_turn"]
        gen_calls = [r for r in turn_trace if r.get("record") == "gen_call"]
        kind_counts: dict[str, int] = {}
        for record in turns:
            key = str(record.get("kind"))
            kind_counts[key] = kind_counts.get(key, 0) + 1
        # The artifact-level oracle for a corrupted AR replica. A healthy run shows 0 here and a
        # positive ``num_gen_calls``; a corrupted one loops every turn while K stays 0. See
        # ``count_degenerate_und_turns`` for why an HTTP canary cannot do this.
        degenerate_und_turns = count_degenerate_und_turns(episode.turn_trace)
        prompt_text = ""
        if tokenizer is not None and prompt_ids:
            try:
                prompt_text = tokenizer.decode(list(prompt_ids), skip_special_tokens=False)
            except Exception:  # noqa: BLE001 - decoding is best effort
                prompt_text = ""

        trusted_reward = dict(episode.bagel_corl_reward)
        trusted_capture_error = None
        if not trusted_reward:
            # Diagnostic capture for V1 comparisons only. This does not change
            # the episode's live reward or promote legacy unmarked RM scores.
            try:
                trusted_reward = trusted_image_reward_fields(
                    episode.gen_samples,
                    num_gen_calls=episode.num_gen_calls,
                    autonomous_done=episode.autonomous_done,
                    forced_reflection=episode.forced_reflection,
                )
            except ValueError as error:
                trusted_capture_error = str(error)

        payload = {
            "trajectory_relpath": relpath,
            "step": step,
            "sample_index": sample_index,
            "episode_uid": episode.episode_uid,
            "und_group_uid": episode.und_group_uid,
            "policy_version": episode.policy_version,
            "ground_truth": dict(ground_truth or {}),
            "corl_reward_version": episode.corl_reward_version,
            "bagel_corl_reward": trusted_reward,
            "trusted_reward_capture_error": trusted_capture_error,
            "autonomous_done": episode.autonomous_done,
            "response_mask": list(episode.response_mask),
            "trace_metadata_origin": {
                "ground_truth": "dataset_rollout_kwargs",
                "bagel_corl_reward": "episode_harness_envelope",
                "response_mask": "episode_response_mask",
            },
            "user_prompt": user_prompt,
            "prompt_text": prompt_text,
            "turns": episode.turns,
            "num_gen_calls": episode.num_gen_calls,
            "und_reward": episode.und_reward,
            "stop_required": episode.stop_required,
            "forced_reflection": episode.forced_reflection,
            "judge_text": episode.judge_text,
            "response_tokens": len(episode.response_ids),
            "response_mask_ones": int(sum(episode.response_mask)),
            # The single boolean that separates "GEN ran and produced nothing" from
            # "GEN was never asked": K == 0 means no GEN request was ever issued.
            "gen_lane_skipped": episode.num_gen_calls == 0,
            "turn_kind_counts": kind_counts,
            "degenerate_und_turns": degenerate_und_turns,
            "gen_calls": gen_calls,
            "turn_trace": turn_trace,
        }
        (trajectory_dir / f"{name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

        lines = [
            f"relpath={relpath}",
            f"episode_uid={episode.episode_uid}",
            f"turns={episode.turns} num_gen_calls={episode.num_gen_calls} "
            f"gen_lane_skipped={episode.num_gen_calls == 0} und_reward={episode.und_reward}",
            f"turn_kind_counts={kind_counts}",
            f"degenerate_und_turns={degenerate_und_turns}",
            f"user_prompt: {user_prompt}",
            "assistant_rollout:",
        ]
        for record in turn_trace:
            if record.get("record") == "gen_call":
                lines.extend(
                    [
                        f"  gen_call turn={record.get('turn')} role={record.get('call_role')} "
                        f"samples={record.get('num_samples')} valid={record.get('num_valid')}",
                        f"    prompt: {record.get('prompt')}",
                        *[f"    image: {p}" for p in (record.get("image_paths") or [])],
                    ]
                )
                continue
            if record.get("record") in ("env_obs", "reflection"):
                lines.append(f"  {record.get('record')} turn={record.get('turn')} (mask 0)")
                lines.extend(f"      {line}" for line in str(record.get("text") or "").splitlines() or [""])
                continue
            header = (
                f"  turn={record.get('turn')} kind={record.get('kind')} "
                f"req_ctx={record.get('req_ctx')} out_tokens={record.get('out_tokens')} "
                f"decode_s={record.get('decode_s')}"
            )
            lines.append(header)
            lines.append("    text:")
            lines.extend(f"      {line}" for line in str(record.get("text") or "").splitlines() or [""])
        (trajectory_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")
        return str(trajectory_dir / f"{name}.txt")
    except Exception:  # noqa: BLE001 - diagnostics must never fail a rollout
        logger.warning("bagel_corl_trace_dump_failed relpath=%s", relpath, exc_info=True)
        return None


def dump_episode_hermes_action(
    *,
    episode: EpisodeRollout,
    relpath: str,
    step: Any = None,
    sample_index: Any = None,
    user_prompt: str = "",
    image_relpath: str | None = None,
) -> str | None:
    """Append one compact action row to ``hermes_actions/step_XXXXXX.jsonl``."""
    if not _trace_dump_enabled():
        return None
    try:
        monitor_dir = resolve_run_dir() / "hermes_actions"
        monitor_dir.mkdir(parents=True, exist_ok=True)
        step_tag = f"step_{int(step):06d}" if step is not None else "step_unknown"
        path = monitor_dir / f"{step_tag}.jsonl"

        gen_calls = [r for r in episode.turn_trace if r.get("record") == "gen_call"]
        kind_counts: dict[str, int] = {}
        for record in episode.turn_trace:
            if record.get("record") == "und_turn":
                kind = str(record.get("kind"))
                kind_counts[kind] = kind_counts.get(kind, 0) + 1

        image_relpath = image_relpath or relpath
        source_paths = [p for call in gen_calls for p in (call.get("image_paths") or [])]
        image_paths = [materialized_image_path(p, image_relpath) for p in source_paths]

        row = {
            "trajectory_relpath": relpath,
            "image_dir": project_relpath(resolve_rollout_images_root() / image_relpath),
            "step": step,
            "sample_index": sample_index,
            "episode_uid": episode.episode_uid,
            "und_group_uid": episode.und_group_uid,
            "policy_version": episode.policy_version,
            "user_prompt": user_prompt,
            "turns": episode.turns,
            "turn_kind_counts": kind_counts,
            # The single boolean separating "GEN ran and produced nothing" from "GEN was
            # never asked" -- the first thing to check when the GEN lane looks idle.
            "num_gen_calls": episode.num_gen_calls,
            "gen_lane_skipped": episode.num_gen_calls == 0,
            "num_tool_calls_executed": len(gen_calls),
            "gen_calls": [
                {
                    "turn": call.get("turn"),
                    "call_role": call.get("call_role"),
                    "prompt": call.get("prompt"),
                    "num_samples": call.get("num_samples"),
                    "num_valid": call.get("num_valid"),
                    "image_paths": [materialized_image_path(p, image_relpath) for p in (call.get("image_paths") or [])],
                }
                for call in gen_calls
            ],
            "image_paths": image_paths,
            "source_image_paths": source_paths,
            "response_tokens": len(episode.response_ids),
            "stop_required": episode.stop_required,
            "forced_reflection": episode.forced_reflection,
            "judge_text": episode.judge_text,
            "reward_metrics": {"und_reward": episode.und_reward},
        }

        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
        finally:
            os.close(fd)
        return str(path)
    except Exception:  # noqa: BLE001 - diagnostics must never fail a rollout
        logger.warning("bagel_corl_hermes_dump_failed relpath=%s", relpath, exc_info=True)
        return None


def validation_artifact_relpath(relpath: str, sampling_params: dict | None) -> str:
    """Return ``relpath`` under a ``val/`` prefix for a validation episode."""
    return f"val/{relpath}" if (sampling_params or {}).get("logprobs") is False else relpath


def dump_episode_images(*, episode: EpisodeRollout, relpath: str, step: Any = None) -> list[str]:
    """Copy this episode's GEN PNGs into ``rollout_images/<relpath>/``."""
    try:
        from verl_omni.utils.agentic.image_gen_rollout_dump import dump_bagel_corl_episode_images

        return dump_bagel_corl_episode_images(
            [{"gen_samples": episode.gen_samples, "trajectory_relpath": relpath}],
            step=step,
        )
    except Exception:  # noqa: BLE001 - diagnostics must never fail a rollout
        logger.warning("bagel_corl_image_dump_failed relpath=%s", relpath, exc_info=True)
        return []


def episode_seed_index(episode_key: str) -> int:
    """Deterministic per-episode integer for the GEN seed base.

    Uses ``zlib.crc32`` rather than the builtin ``hash``: ``hash`` is salted per
    process (``PYTHONHASHSEED``), so a replayed step would draw *different* noise
    than the run it is meant to reproduce, and FlowGRPO's GEN keys are supposed to
    be reproducible per ``(episode, call)``.
    """
    return int(zlib.crc32(str(episode_key).encode("utf-8")))


def episode_artifact_keys(
    *,
    dataset_task_uid: str,
    session_id: Any,
    global_steps: Any,
    seed: Any = None,
) -> tuple[str, int]:
    """Return ``(relpath, seed_base)`` for one episode."""
    episode_key = f"{dataset_task_uid}:{session_id}"
    relpath = build_trajectory_relpath(
        step=global_steps,
        sample_index=dataset_task_uid,
        rollout_n=int(session_id or 0) + 1,
    )
    seed_base = derive_rollout_seed(
        int(seed or global_steps or 0),
        episode_seed_index(episode_key),
    )
    return relpath, seed_base


@register("bagel_multiturn_agent")
class BagelMultiturnAgentLoop(AgentLoopBase):
    """Per-episode serial UND→GEN→RM loop. Outer gather stays on the worker."""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # Stash for the GEN tool (``_generate_image``) so it can build diffusion
        # sampling params (seed / logprobs / num_inference_steps) per FlowGRPO seed.
        self._sampling_params = sampling_params
        dataset_task_uid = str(
            kwargs.get("dataset_task_uid")
            or kwargs.get("uid")
            or (kwargs.get("extra_info") or {}).get("dataset_task_uid")
            or "missing_task"
        )
        # This is the requested trainer step for episode IDs, not evidence of
        # which weights served the tokens. The decode outputs supply that version.
        requested_policy_version = int(
            sampling_params.get("global_steps", kwargs.get("global_steps", kwargs.get("policy_version", 0))) or 0
        )
        raw_prompt = kwargs.get("raw_prompt")
        if raw_prompt is None:
            raise ValueError("bagel_multiturn_agent requires raw_prompt")
        # Transformers 4/5: tokenize=True may return BatchEncoding; list(enc) would iterate keys ('input_ids', ...)
        # and crash normalize_token_ids on the server.
        prompt_ids = kwargs.get("prompt_ids")
        injected_tools = False
        if (
            _inject_tools_schema_enabled()
            and _messages_are_text_only(raw_prompt)
            and _template_renders_hermes_tools(self.tokenizer)
        ):
            # Re-render from the messages rather than patching the dataset's ids: the ``<tools>``
            # block belongs inside the system turn, ahead of its text.
            #
            # ``task_type`` decides whether the dataset's own system turn survives: plan rows must
            # keep ``PLAN_SYSTEM_PROMPT`` (see ``with_reflection_system_prompt``), whose
            # numbered-plan sentence is what makes ``R_plan`` / ``R_result`` reachable.
            task_type = resolve_task_type(extra_info=kwargs.get("extra_info"), ground_truth=kwargs.get("ground_truth"))
            encoded = self.tokenizer.apply_chat_template(
                _with_reflection_system_prompt(raw_prompt, task_type=task_type),
                tools=[GENERATE_IMAGE_TOOL_SCHEMA, JUDGE_IMAGE_TOOL_SCHEMA],
                add_generation_prompt=True,
                tokenize=True,
            )
            prompt_ids = normalize_token_ids(encoded)
            injected_tools = True
        elif prompt_ids is None:
            encoded = self.tokenizer.apply_chat_template(raw_prompt, add_generation_prompt=True, tokenize=True)
            prompt_ids = normalize_token_ids(encoded)
        else:
            prompt_ids = normalize_token_ids(prompt_ids)
        if os.getenv("BAGEL_CORL_DEBUG") == "1":
            logger.info(
                "bagel_corl_prompt uid=%s tools_injected=%s prompt_tokens=%d tail=%r",
                dataset_task_uid,
                injected_tools,
                len(prompt_ids),
                # The tail is the generation prompt (``<|im_start|>assistant\n``) -- cheap proof
                # that the rendered prompt ends where the engine expects it to.
                self.tokenizer.decode(prompt_ids[-24:], skip_special_tokens=False),
            )

        agent_cfg = self.config.actor_rollout_ref.rollout.agent
        # Response kind the RM worker's reward manager will dtype-check the in-loop
        # payload row against. ``output_type`` is a real ``DiffusionPipelineConfig``
        # field, so ``get`` + a literal fallback is only defensive; the fallback
        # matches the field's own default.
        rollout_pipeline = self.config.actor_rollout_ref.rollout.get("pipeline") or {}
        knobs = resolve_bagel_agent_knobs(agent_cfg)
        s = knobs["gen_samples_per_call"]
        max_passes = knobs["max_generate_passes"]
        max_und_turns = knobs["max_und_turns"]
        # good_enough SoT: agentic_image_gen.good_enough_threshold (yaml, bound at
        # worker init). Reduction over per-seed flags defaults to best-of-S ("any").
        good_enough_threshold = float(agentic_get("good_enough_threshold"))
        good_enough_reduction = knobs["good_enough_reduction"]
        # RFC §4.4.2b (R2): conditioning reuse only pays off when the whole S-group
        # is pinned to one replica, because each replica owns its own cache. Read the
        # affinity knob once per episode rather than per seed.
        rollout_cfg = self.config.actor_rollout_ref.rollout
        # Episode context budget: ``_und_decode`` feeds the engine ``prompt_ids +
        # response_ids``, the trainer pads every trajectory to
        # ``rollout.prompt_length + rollout.response_length``, and the AR engine's
        # ``max_model_len`` is sized against that same sum. The turn loop therefore has
        # to stop there -- see ``run_serial_episode``'s ``max_context_tokens``.
        max_context_tokens = int(rollout_cfg.prompt_length) + int(rollout_cfg.response_length)
        max_response_tokens = int(rollout_cfg.response_length)
        self._bagel_cond_affinity = bool(
            getattr(rollout_cfg, "enable_prompt_embed_cache", False)
            and getattr(rollout_cfg, "enable_prompt_embed_cache_routing_affinity", False)
        )
        # RFC §4.4.4 R2 accumulator: engine-measured counters for this episode's GEN requests, turned into ratios once
        # the episode ends.
        self._bagel_r2: dict[str, Any] = {
            "calls": 0,
            "hits": None,
            "misses": None,
            "bypassed": None,
        }
        # Role timing (RFC §8 KPIs): per-episode wall-clock split by lane.
        timing = {"und_decode_s": 0.0, "gen_s": 0.0, "rm_s": 0.0}
        self._bagel_timing = timing
        # In-loop RM (RFC §4.2): score each generate_image call mid-episode via the
        # GEN-side reward handle bound by the worker. UniCoT reference paths ride on
        # the row's extra_info (stamped by the dataset builder / driver).
        rm_handle = get_bagel_rm_gen_handle()
        if rm_handle is not None:
            extra_info = kwargs.get("extra_info") if isinstance(kwargs.get("extra_info"), dict) else {}
            ref_paths = extra_info.get("reference_image_path")
            self._rm_reference_paths = (
                [str(p) for p in ref_paths]
                if isinstance(ref_paths, list | tuple)
                else ([str(ref_paths)] if ref_paths else [])
            )
            self._rm_extra_info = dict(extra_info)
            self._rm_score_fn = make_rm_score_fn(
                rm_handle,
                get_reference_paths=lambda: list(self._rm_reference_paths),
                get_extra_info=lambda: {**self._rm_extra_info, "user_prompt": get_active_user_prompt() or ""},
                get_scorer_knobs=lambda: dict(agentic_scorer_knobs_from_config(self.config)),
                get_image_prompt=lambda: get_latest_generate_prompt_for_active_rollout() or "",
                # The RM worker's reward manager validates the row's ``responses``
                # against the rollout pipeline's response kind, so the un-used
                # placeholder must match it (image → uint8, latent → float).
                output_type=str(rollout_pipeline.get("output_type", "image")),
            )
        else:
            self._rm_score_fn = None

        async def _und_decode(**_decode_kwargs):
            t0 = time.perf_counter()
            try:
                und_params = dict(sampling_params)
                und_params["bagel_role"] = "und"
                # π_rollout for the UND lane: the recipe runs with ``rollout.calculate_log_probs=True``, so the v1
                # trainer reads ``rollout_log_probs`` off the TQ next to our recomputed ``old_log_probs``
                # (``trainer_base.py:1506-1509``); without it the step dies with ``KeyError: 'rollout_log_probs'``.
                und_params["logprobs"] = 0
                # Bound this turn, and cut it at the call boundary. See
                # ``und_turn_max_tokens`` / ``UND_STOP_SEQUENCES``: the AR strategy's default cap
                # is the whole episode response budget, so without an explicit ``max_tokens`` a
                # single degenerate turn (role-label repetition, or the invented ``<output>``
                # spiral after a call) consumes the episode and nothing is left to train on.
                _context_used = len(_decode_kwargs["prompt_ids"]) + len(_decode_kwargs["response_ids"])
                _turn_max = und_turn_max_tokens(
                    context_used=_context_used,
                    max_context_tokens=min(max_context_tokens, len(_decode_kwargs["prompt_ids"]) + max_response_tokens),
                )
                if _turn_max > 0:
                    und_params["max_tokens"] = _turn_max
                und_params.setdefault("stop", list(UND_STOP_SEQUENCES))
                und_params.setdefault("include_stop_str_in_output", True)
                # UND must hit the AR replica (BagelDualRoleLLMServerClient routes on role /
                # absence of diffusion keys). Never send num_inference_steps here.
                for _k in (
                    "num_inference_steps",
                    "noise_level",
                    "sde_window_size",
                    "sde_window_range",
                    "sde_type",
                    "height",
                    "width",
                ):
                    und_params.pop(_k, None)
                try:
                    output = await generate_retrying_engine_wake(
                        partial(
                            self.server_manager.generate,
                            request_id=str(uuid.uuid4()),
                            prompt_ids=list(_decode_kwargs["prompt_ids"]) + list(_decode_kwargs["response_ids"]),
                            sampling_params=und_params,
                        ),
                        on_retry=partial(
                            _log_engine_asleep_retry,
                            logger,
                            lane="und",
                            step=getattr(self, "global_steps", None),
                        ),
                    )
                except Exception as exc:  # noqa: BLE001 — map diffusion-replica errors to UND dual-role failure
                    err = str(exc)
                    if "num_inference_steps" in err or "Diffusion" in type(exc).__name__:
                        raise RuntimeError(
                            "Bagel Co-RL (Joint-Training) UND decode hit the GEN diffusion replica "
                            "(bagel_single_stage / DiffusionStrategy). UND needs AR TokenOutput "
                            "(Hermes tool-call); refuse soft-empty TQ. "
                            "Prove dual-role serving, then set agent.und_ar_serving_ready=True. "
                            f"Underlying error: {err}"
                        ) from exc
                    raise
                if hasattr(output, "diffusion_output"):
                    raise RuntimeError(
                        "Bagel Co-RL (Joint-Training) UND decode received DiffusionOutput: "
                        "rollout is on DiffusionStrategy (bagel_single_stage / output_mode≠ar). "
                        "UND needs AR token generation; "
                        "refuse empty token_ids that would leave TQ with no materializable trajectories."
                    )
                if not hasattr(output, "token_ids"):
                    raise RuntimeError(
                        "Bagel Co-RL (Joint-Training) UND decode expected TokenOutput with token_ids, "
                        f"got {type(output)!r}"
                    )
                token_ids = list(output.token_ids)
                text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
                step_log_probs = getattr(output, "log_probs", None)
                if step_log_probs is None or len(step_log_probs) != len(token_ids):
                    # Fail loud rather than publish a fabricated ratio: the trainer pairs these
                    # with ``old_log_probs`` for the π_rollout/π_θ correction, so a silent
                    # zero-fill would corrupt the objective instead of erroring.
                    raise RuntimeError(
                        "Bagel Co-RL (Joint-Training) UND decode got no per-token log-probs "
                        f"(log_probs={None if step_log_probs is None else len(step_log_probs)} for "
                        f"{len(token_ids)} tokens). The AR replica must be asked with "
                        "sampling_params['logprobs']=0 and its engine started with a logprobs_mode."
                    )
                provenance = serving_rollout_provenance(
                    dict(getattr(output, "extra_fields", None) or {}), required=True
                )
                return {
                    "token_ids": token_ids,
                    "text": text,
                    "log_probs": [float(v) for v in step_log_probs],
                    **provenance,
                }
            finally:
                timing["und_decode_s"] += time.perf_counter() - t0

        # Rollout-scoped artifact context: registrations inside the episode land under
        # this trajectory in the unified tools.trajectory registry (audit T1.6).
        global_steps = sampling_params.get("global_steps", kwargs.get("global_steps"))
        session_id = kwargs.get("session_id", kwargs.get("index", 0))
        # GEN group seed base: episode-scoped, so the S FlowGRPO seeds are not the constant ``range(S)`` they used to
        # be.
        relpath, seed_base = episode_artifact_keys(
            dataset_task_uid=dataset_task_uid,
            session_id=session_id,
            global_steps=global_steps,
            seed=sampling_params.get("seed"),
        )
        tool = BagelGenerateImageTool(
            gen_samples_per_call=s,
            max_generate_passes=max_passes,
            generate_fn=self._generate_image,
            seed_base=seed_base,
        )
        # Non-image UND scalar for pattern-3 (K=0) episodes. The reward model computes
        # the authoritative token-GRPO reward post-hoc (via the worker's ``_compute_score``);
        # this is only a fallback used when no RM is attached.
        non_image_reward = kwargs.get("non_image_reward")
        # ``relpath`` is precomputed beside ``seed_base`` so the folder and the noise
        # draw share one episode identity (see ``episode_artifact_keys``).
        path_tokens = set_active_trajectory_relpath(relpath)
        user_prompt_text = last_user_prompt(raw_prompt)
        prompt_token = set_active_user_prompt(user_prompt_text)
        episode_ground_truth = trace_ground_truth(kwargs)
        try:
            expected_num_images = max(1, int(episode_ground_truth.get("expected_num_images") or 1))
        except (TypeError, ValueError):
            expected_num_images = 1
        try:
            episode = await run_serial_episode(
                dataset_task_uid=dataset_task_uid,
                policy_version=requested_policy_version,
                prompt_ids=list(prompt_ids),
                und_decode=_und_decode,
                generate_tool=tool,
                score_fn=self._score_gen_samples,
                tokenizer=self.tokenizer,
                max_und_turns=max_und_turns,
                max_context_tokens=max_context_tokens,
                max_response_tokens=max_response_tokens,
                non_image_reward=float(non_image_reward) if non_image_reward is not None else None,
                good_enough_threshold=good_enough_threshold,
                good_enough_reduction=good_enough_reduction,
                corl_reward_version=str(agentic_scorer_knobs_from_config(self.config).get("corl_reward_version", "v1")),
                fixed_gen_prompt=str(self.config.actor_rollout_ref.rollout.agent.get("corl_fixed_gen_prompt", "")),
                task_type=episode_ground_truth.get("task_type"),
                expected_num_images=expected_num_images,
            )
        finally:
            reset_active_trajectory_relpath(path_tokens)
            reset_active_user_prompt(prompt_token)
        # Per-episode trajectory dump.
        image_relpath = validation_artifact_relpath(relpath, getattr(self, "_sampling_params", None))
        episode.trace_dump_path = dump_episode_trace(
            episode=episode,
            relpath=relpath,
            step=global_steps,
            sample_index=session_id,
            user_prompt=user_prompt_text,
            prompt_ids=list(prompt_ids),
            tokenizer=self.tokenizer,
            image_relpath=image_relpath,
            ground_truth=episode_ground_truth,
        )
        # Action-level index beside the trajectory (and the PNGs the image dumper copies
        # under the same ``relpath``). Written from the episode because the bagel TQ
        # worker never runs ``dump_raw_rollouts``, which is what produces
        # ``hermes_actions`` for the sibling agentic lane.
        dump_episode_hermes_action(
            episode=episode,
            relpath=relpath,
            step=global_steps,
            sample_index=session_id,
            user_prompt=user_prompt_text,
            image_relpath=image_relpath,
        )
        # Same reason: copy the GEN PNGs out of the scratch dir into the run's
        # ``rollout_images`` tree, next to this episode's trajectory.
        dump_episode_images(episode=episode, relpath=image_relpath, step=global_steps)
        # RFC §4.4.4 R2: engine-measured conditioning reuse for this episode. Empty
        # whenever the engine exposed no counters, so nothing is fabricated.
        r2 = getattr(self, "_bagel_r2", None) or {}
        episode.r2_metrics = cond_reuse_metrics(
            calls=int(r2.get("calls", 0)),
            hits=r2.get("hits"),
            misses=r2.get("misses"),
            bypassed=r2.get("bypassed", 0),
        )
        return episode_to_agent_output(episode, timing=self._bagel_timing, relpath=relpath)

    async def _generate_image(self, **kwargs) -> list[dict[str, Any]]:
        """Run **one** GEN turn: S same-conditioning FlowGRPO seeds, then return."""
        prompt = strip_reflection_tail(str(kwargs.get("prompt", "")))
        seeds = list(kwargs.get("seeds") or [])
        if not prompt:
            raise ValueError("_generate_image requires a non-empty diffusion prompt")

        rollout = self.config.actor_rollout_ref.rollout
        base = dict(getattr(self, "_sampling_params", None) or {})
        # ``logprobs is False`` is the validate marker the val overlay sets (see ``validation_artifact_relpath``).
        is_validate = base.get("logprobs") is False
        rows: list[dict[str, Any]] = []
        timing = getattr(self, "_bagel_timing", None)
        # BagelPipeline reads text from the request prompt; encode the diffusion prompt directly (not the whole UND
        # conversation) so the GEN replica denoises the right conditioning.
        gen_prompt_ids = normalize_token_ids(self.tokenizer.encode(prompt, add_special_tokens=False))
        if not gen_prompt_ids:
            raise ValueError("_generate_image requires a non-empty tokenized diffusion prompt")
        cond_uid = conditioning_uid(gen_prompt_ids)
        cond_len = len(gen_prompt_ids)
        # RFC §4.4.2b: one sticky key for the whole S-group. ``LLMServerClient.generate``
        # acquires its replica from the pool's load balancer with this id, so passing
        # the *same* key to every seed pins the group to the replica that then holds
        # the conditioning entry and lets seeds 2..S hit it. A per-seed ``uuid4()``
        # scattered the group across replicas and made a cold cache guaranteed.
        affinity = bool(getattr(self, "_bagel_cond_affinity", False))
        routing_key = self._cond_routing_key(kwargs.get("gen_call_id") or cond_uid) if affinity else None
        counters_before = await self._prompt_embed_cache_counters(routing_key)
        for seed in seeds:
            t0 = time.perf_counter()
            try:
                request_params = build_gen_sampling_params(rollout, base=base, seed=int(seed))
                request_params["bagel_role"] = "gen"
                # Same wake-window rejection as the UND lane, and the same fix: the engine
                # rejects before executing, so re-issuing is safe. The request id is the S-group's
                # sticky ``routing_key``, so reusing it is required -- a fresh uuid would scatter
                # the group across replicas and cold the conditioning cache (RFC §4.4.2b).
                output = await generate_retrying_engine_wake(
                    partial(
                        self.server_manager.generate,
                        request_id=routing_key or str(uuid.uuid4()),
                        prompt_ids=gen_prompt_ids,
                        sampling_params=request_params,
                    ),
                    on_retry=partial(
                        _log_engine_asleep_retry,
                        logger,
                        lane="gen",
                        step=getattr(self, "global_steps", None),
                    ),
                )
                # Training packs the FlowGRPO trajectory and fails closed without it;
                # validation decodes ODE (``noise_level=0.0``) so it has no SDE
                # log-probs and must not be required to carry one. See
                # ``stash_gen_row_from_diffusion_output`` for the split.
                row = stash_gen_row_from_diffusion_output(output, seed=int(seed), require_traj=not is_validate)
                # Live serving must identify its own GEN version/replica; a
                # parent UND or requested trainer step cannot supply these.
                row.update(serving_rollout_provenance(row, required=True))
                row["cond_uid"] = cond_uid
                row["cond_len"] = cond_len
                rows.append(row)
            finally:
                if timing is not None:
                    timing["gen_s"] += time.perf_counter() - t0
        await self._record_cond_reuse(routing_key, counters_before, calls=len(seeds))
        return rows

    def _cond_routing_key(self, group_key: str) -> str:
        """Sticky replica key for one GEN S-group (RFC §4.4.2b).

        Namespaced so it cannot collide with a routing id another pool minted.
        """
        return f"bagel_cond::{group_key}"

    async def _prompt_embed_cache_counters(self, routing_key: str | None) -> dict[str, Any] | None:
        """Read the conditioning-cache counters of the replica ``routing_key`` pins to.

        Returns ``None`` whenever affinity is off or the client/engine cannot
        answer. Metrics must never break a rollout, so every failure degrades to
        "no measurement" rather than an exception or a fabricated zero.
        """
        if routing_key is None:
            return None
        probe = getattr(self.server_manager, "prompt_embed_cache_stats", None)
        if probe is None:
            return None
        try:
            return await probe(routing_key)
        except Exception as exc:  # noqa: BLE001 — metrics only
            logger.warning("bagel_corl R2: prompt-embed cache probe failed: %s", exc)
            return None

    async def _record_cond_reuse(
        self,
        routing_key: str | None,
        before: dict[str, Any] | None,
        *,
        calls: int,
    ) -> None:
        """Accumulate one ``generate_image`` turn's engine-measured reuse (RFC §4.4.4)."""
        r2 = getattr(self, "_bagel_r2", None)
        if r2 is None:
            return
        r2["calls"] = int(r2.get("calls", 0)) + max(0, int(calls))
        if routing_key is None:
            return
        after = await self._prompt_embed_cache_counters(routing_key)
        if before is None or after is None:
            # Nothing measured: leave ``hits``/``misses`` at ``None`` so the episode
            # reports no R2 numbers instead of a fabricated perfect ratio.
            return
        if r2.get("hits") is None:
            # First successful probe: start counting from here.
            r2["hits"] = 0
            r2["misses"] = 0
            r2["bypassed"] = 0
        for key in ("hits", "misses", "bypassed"):
            delta = int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0)
            if delta > 0:
                r2[key] = int(r2.get(key, 0) or 0) + delta

    async def _score_gen_samples(self, samples: list[GenSample]) -> list[GenSample]:
        """Score GEN samples via an injected RM hook (C/A, UniCoT similarity, good_enough).

        When no hook is wired (no mid-episode RM), derive ``good_enough`` from any
        pre-populated ``rm_score`` and otherwise leave the samples unscored so the
        post-episode reward loop supplies the authoritative scalar. Never hardcode a
        zero score.
        """
        rm_fn = getattr(self, "_rm_score_fn", None)
        if rm_fn is None:
            for sample in samples:
                if sample.rm_score is not None and sample.good_enough is None:
                    sample.good_enough = derive_good_enough_from_scores(
                        correctness=float(sample.rm_score),
                        aesthetics=float(sample.rm_score),
                    )
            return samples
        t0 = time.perf_counter()
        try:
            result = rm_fn(samples)
            if asyncio.iscoroutine(result):
                result = await result
            return result
        finally:
            timing = getattr(self, "_bagel_timing", None)
            if timing is not None:
                timing["rm_s"] += time.perf_counter() - t0


class MultiturnAgentLoopWorker(CompositeAgentLoopWorker):
    """Composite worker that gathers J serial episodes, then flattens UND vs GEN.

    Prefer ``BagelCorlAgentLoopWorkerTQ`` for bagel_corl_sync (dual-lane TQ).
    """

    async def generate_sequences(self, batch):
        output = await super().generate_sequences(batch)
        turns = []
        if "metrics" in output.meta_info:
            for row in output.meta_info["metrics"]:
                if isinstance(row, dict) and "turns" in row:
                    turns.append(int(row["turns"]))
        hist = turn_histogram(turns)
        extra = output.meta_info.setdefault("bagel_corl", {})
        extra.update(hist)
        logger.info("bagel_corl turn histogram: %s", hist)
        agent_cfg = getattr(self.config.actor_rollout_ref.rollout, "agent", None)
        expected_s = int(getattr(agent_cfg, "gen_samples_per_call", None) or 4)
        from verl_omni.agent_loop.bagel_corl_lib import flatten_from_agent_output, strip_pixels_for_actor

        flat = flatten_from_agent_output(output, expected_s=expected_s)
        extra.update(flat.metrics)
        extra["gen_batch"] = [strip_pixels_for_actor(row) for row in flat.gen_batch]
        extra["und_batch"] = flat.und_batch
        extra["gen_episode_map"] = flat.gen_episode_map
        try:
            from verl_omni.utils.agentic.image_gen_rollout_dump import (
                dump_bagel_corl_episode_images,
                dump_raw_rollouts,
            )

            step = None
            if hasattr(batch, "meta_info") and isinstance(batch.meta_info, dict):
                step = batch.meta_info.get("global_steps") or batch.meta_info.get("step")
            if step is None and isinstance(getattr(output, "meta_info", None), dict):
                step = output.meta_info.get("global_steps") or output.meta_info.get("step")
            try:
                dump_raw_rollouts(
                    tokenizer=getattr(self, "tokenizer", None),
                    output=output,
                    step=step,
                )
            except (TypeError, KeyError, AttributeError) as exc:
                logger.debug("dump_raw_rollouts skipped for Bagel Co-RL (Joint-Training) batch shape: %s", exc)
                dump_bagel_corl_episode_images(output, step=step)
        except ImportError as exc:
            logger.debug("rollout dump unavailable: %s", exc)
        return output
