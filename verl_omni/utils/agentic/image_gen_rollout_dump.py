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

"""Dump / materialization / invalid-rollout masking for agentic monitoring."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from verl_omni.tools.trajectory import (
    build_trajectory_relpath,
    resolve_run_dir,
)
from verl_omni.utils.agentic.image_gen_rollout_parse import (
    extract_generate_image_prompts,
    last_user_prompt,
    split_rollout_turns,
    turn_kind,
    unpad_left_ids,
)
from verl_omni.utils.metrics_utils import AgenticRewardMetrics

logger = logging.getLogger(__name__)

_TOOL_CALL_PAT = (
    r"<tool_call>\s*(?:\{.*?\"name\"\s*:\s*\"[^\"]+\".*?\}|"
    r"<function=[^>\s]+\s*>.*?</function>)\s*</tool_call>"
)
_EXECUTED_TOOL_RESPONSE_PAT = r"\bagentic_(?:tool|reflect|judge)\s+ok=[01]\b"
_IMAGE_PATH_IN_OBS_PAT = r"path=((?:/|[A-Za-z]:\\)[^\s\"']+\.(?:png|jpg|jpeg|webp))"


def materialize_rollout_images(
    *,
    decoded_response: str,
    run_dir: Path,
    relpath: str,
    user_prompt: str,
) -> list[str]:
    """Index images already written by the live tool; never create empty folders."""
    target_dir = run_dir / "rollout_images" / relpath
    # ``agentic_tool._save_images`` creates this directory only after a real
    # generate_image execution. A rollout with no generated artifact must not
    # gain a confusing meta-only ``sample_*`` directory during post-processing.
    if not target_dir.is_dir():
        return []

    prompts = extract_generate_image_prompts(decoded_response)
    image_paths = [
        str(path)
        for path in sorted(target_dir.iterdir())
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ]
    if not image_paths:
        # Preserve any live failure artifact (for example STUB_NO_IMAGE), but do
        # not manufacture or update meta.json for a directory with no images.
        return []

    meta_path = target_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    except json.JSONDecodeError:
        meta = {}
    meta.update(
        {
            "trajectory_relpath": relpath,
            "user_prompt": user_prompt,
            "image_paths": image_paths,
            "tool_prompts": prompts,
            "source": "direct_tool_write",
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return image_paths


def discard_invalid_rollouts(output: Any) -> None:
    """Drop no-``generate_image`` rollouts from the policy update.

    Sets ``response_mask`` to 0 so GRPO/PPO give them no gradient. Their
    scalar reward is already 0 with ``rollout_valid=0`` from
    ``agentic_reward``; they can still slightly affect the GRPO group mean,
    which is acceptable (penalizes skip-gen relative to siblings).
    """
    valid = output.non_tensor_batch.get("rollout_valid")
    n_gen = output.non_tensor_batch.get("num_generate_image_prompts")
    response_mask = output.batch.get("response_mask")
    if response_mask is None:
        return
    n = int(response_mask.shape[0])
    # Keep a copy so we can restore if zeroing would empty the whole batch
    # (verl rollout_corr raises: "response_mask must contain at least one valid token").
    original_mask = response_mask.clone()
    dropped = 0
    for i in range(n):
        is_valid = True
        if valid is not None:
            try:
                is_valid = int(np.asarray(valid[i]).reshape(-1)[0]) == 1
            except (TypeError, ValueError, IndexError):
                is_valid = True
        elif n_gen is not None:
            try:
                is_valid = int(np.asarray(n_gen[i]).reshape(-1)[0]) >= 1
            except (TypeError, ValueError, IndexError):
                is_valid = True
        if is_valid:
            continue
        response_mask[i].zero_()
        dropped += 1
    if dropped and not bool(response_mask.any()):
        response_mask.copy_(original_mask)
        logger.warning(
            "All %d/%d rollouts lacked generate_image; kept response_mask intact "
            "to avoid empty-mask crash in rollout_corr (rewards stay 0)",
            dropped,
            n,
        )
    elif dropped:
        logger.info(
            "Discarded %d/%d rollouts with no generate_image (response_mask=0, rollout_valid=0)",
            dropped,
            n,
        )


def _resolve_sample_relpath(
    *,
    i: int,
    sample_index: Any,
    sample_key: str,
    rollout_counts: dict[str, int],
    live_relpaths: Any,
    step_i: int,
) -> tuple[str, str, int]:
    """Return ``(relpath, sample_key, rollout_n)`` for one batch row."""
    live_relpath = None
    if live_relpaths is not None:
        try:
            raw = live_relpaths[i]
            if raw:
                live_relpath = str(raw)
        except (TypeError, IndexError, KeyError):
            live_relpath = None
    if live_relpath:
        relpath = live_relpath
        # Parse sample_index.rollout_n from ``…/sample_6.03`` when present.
        name = Path(relpath).name
        m = re.fullmatch(r"sample_(.+)\.(\d+)$", name)
        if m:
            sample_key = m.group(1)
            rollout_n = int(m.group(2))
        else:
            rollout_n = rollout_counts.get(sample_key, 0)
    else:
        rollout_n = rollout_counts.get(sample_key, 0)
        relpath = build_trajectory_relpath(
            step=step_i,
            sample_index=sample_index,
            rollout_n=rollout_n,
        )
    return relpath, sample_key, rollout_n


def _annotate_ordered_turns(rollout_turns: list[dict[str, Any]], user_prompt: str) -> list[dict[str, Any]]:
    if rollout_turns and not rollout_turns[0].get("turn_prompt"):
        rollout_turns[0]["turn_prompt"] = user_prompt
    # Prefer the full chat-templated model input in ``turn_prompt`` (system +
    # Tools schema + history). Keep the short env delta in ``turn_obs``.
    for turn in rollout_turns:
        turn_obs = turn.get("turn_prompt") or ""
        turn_input = turn.get("turn_input") or ""
        turn["turn_obs"] = turn_obs
        if turn_input:
            turn["turn_prompt"] = turn_input
        turn["turn_kind"] = turn_kind(
            turn.get("decode") or "",
            turn_obs,
            turn.get("response") or "",
        )
    return [
        {
            "turn": t.get("turn"),
            "turn_kind": t.get("turn_kind"),
            "turn_prompt": t.get("turn_prompt") or "",
            "turn_obs": t.get("turn_obs") or "",
            "decode": t.get("decode") or "",
            "response": t.get("response") or "",
            "decode_has_tool_call": bool(t.get("decode_has_tool_call")),
        }
        for t in rollout_turns
    ]


def _build_trajectory_payload(
    *,
    relpath: str,
    image_dir: str,
    step_i: int,
    sample_index: Any,
    rollout_n: int,
    user_prompt: str,
    ordered_turns: list[dict[str, Any]],
    image_paths: list[str],
) -> dict[str, Any]:
    return {
        "trajectory_relpath": relpath,
        "image_dir": image_dir,
        "step": step_i,
        "sample_index": int(sample_index) if str(sample_index).lstrip("-").isdigit() else str(sample_index),
        "rollout_n": rollout_n,
        "user_prompt": user_prompt,
        "rollout_turns": ordered_turns,
        "image_paths": image_paths,
        "image_paths_in_obs": sorted(
            {
                m.group(1)
                for turn in ordered_turns
                for m in re.finditer(
                    _IMAGE_PATH_IN_OBS_PAT,
                    turn.get("turn_obs") or "",
                    flags=re.IGNORECASE,
                )
            }
        ),
        "num_tool_calls_executed": sum(
            len(re.findall(_EXECUTED_TOOL_RESPONSE_PAT, turn.get("turn_obs") or "", flags=re.IGNORECASE))
            for turn in ordered_turns
        ),
        "num_forced_tool_calls": 0,
        "num_voluntary_tool_calls": sum(
            len(re.findall(_TOOL_CALL_PAT, turn.get("decode") or "", flags=re.IGNORECASE | re.DOTALL))
            for turn in ordered_turns
        ),
        # Legacy field name retained for downstream dashboards.
        "num_voluntary_hermes": sum(
            len(re.findall(_TOOL_CALL_PAT, turn.get("decode") or "", flags=re.IGNORECASE | re.DOTALL))
            for turn in ordered_turns
        ),
    }


def _format_turn_text_block(turn: dict[str, Any]) -> list[str]:
    t = int(turn["turn"])
    turn_prompt = turn.get("turn_prompt") or ""
    response = turn.get("response") or ""
    decode = turn.get("decode") or ""
    kind = turn.get("turn_kind") or "other"
    header = f"  turn={t} kind={kind} decode_has_tool_call={turn['decode_has_tool_call']}"
    return [
        header,
        f"    turn_{t}_prompt:",
        *[f"      {line}" for line in (turn_prompt.splitlines() or [""])],
        f"    turn_{t}_response:",
        *[f"      {line}" for line in (response.splitlines() or [""])],
        "    decode:",
        *[f"      {line}" for line in (decode.splitlines() or [""])],
    ]


def dump_raw_rollouts(*, tokenizer: Any, output: Any, step: Any) -> None:
    """Write user prompt + raw assistant turns only."""
    try:
        responses = output.batch["responses"]
        response_masks = output.batch["response_mask"]
        raw_prompts = output.non_tensor_batch.get("raw_prompt")
        indices = output.non_tensor_batch.get("index", np.arange(len(responses)))
        step_i = int(step) if step is not None else -1
        step_tag = f"step_{step_i:06d}" if step_i >= 0 else "step_unknown"
        run_dir = resolve_run_dir()
        monitor_dir = run_dir / "hermes_actions"
        trajectory_dir = run_dir / "rollout_trajectories" / step_tag
        monitor_dir.mkdir(parents=True, exist_ok=True)
        trajectory_dir.mkdir(parents=True, exist_ok=True)

        rollout_counts: dict[str, int] = {}
        step_text: list[str] = []
        jsonl_rows: list[str] = []
        live_relpaths = output.non_tensor_batch.get("trajectory_relpath")
        for i in range(len(responses)):
            sample_index = indices[i]
            sample_key = str(int(sample_index)) if str(sample_index).lstrip("-").isdigit() else str(sample_index)
            # Prefer the live artifact id stamped by ImageGenToolAgentLoop so
            # trajectory JSON / image folders match path= markers in tool obs.
            # Fallback: renumber by batch order (legacy; can diverge if workers
            # reorder outputs relative to dispatch).
            relpath, sample_key, rollout_n = _resolve_sample_relpath(
                i=i,
                sample_index=sample_index,
                sample_key=sample_key,
                rollout_counts=rollout_counts,
                live_relpaths=live_relpaths,
                step_i=step_i,
            )
            rollout_counts[sample_key] = max(rollout_counts.get(sample_key, 0), int(rollout_n) + 1)
            user_prompt = last_user_prompt(raw_prompts[i]) if raw_prompts is not None else ""
            prompt_ids = None
            if "prompts" in output.batch:
                prompt_ids = unpad_left_ids(
                    output.batch["prompts"][i],
                    getattr(tokenizer, "pad_token_id", None),
                )
            rollout_turns = split_rollout_turns(
                responses[i],
                response_masks[i],
                tokenizer,
                prompt_ids=prompt_ids,
            )
            decoded_response = tokenizer.decode(
                responses[i].tolist(),
                skip_special_tokens=False,
            )
            image_paths = materialize_rollout_images(
                decoded_response=decoded_response,
                run_dir=run_dir,
                relpath=relpath,
                user_prompt=user_prompt,
            )
            image_dir = str(run_dir / "rollout_images" / relpath) if image_paths else ""
            ordered_turns = _annotate_ordered_turns(rollout_turns, user_prompt)
            payload = _build_trajectory_payload(
                relpath=relpath,
                image_dir=image_dir,
                step_i=step_i,
                sample_index=sample_index,
                rollout_n=rollout_n,
                user_prompt=user_prompt,
                ordered_turns=ordered_turns,
                image_paths=image_paths,
            )
            reward_metrics = AgenticRewardMetrics.for_rollout(output, i)

            name = Path(relpath).name
            (trajectory_dir / f"{name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
            trajectory_text = [
                f"relpath={relpath}",
                f"user_prompt: {user_prompt}",
                "assistant_rollout:",
            ]
            step_text.extend(
                [
                    f"=== {name}  sample={sample_index} rollout_n={rollout_n} ===",
                    f"user_prompt: {user_prompt}",
                    "assistant_rollout:",
                ]
            )
            for turn in ordered_turns:
                block = _format_turn_text_block(turn)
                trajectory_text.extend(block)
                step_text.extend(block)
            trajectory_text.append("")
            step_text.append("")
            (trajectory_dir / f"{name}.txt").write_text("\n".join(trajectory_text) + "\n")
            # ``rollout_trajectories`` is the canonical home of raw
            # decodes. Keep hermes_actions compact and focused on action
            # metadata plus the exact per-rollout reward outputs.
            # Hermes JSONL stays compact: short env obs only (not the full template).
            monitor_payload = {
                **payload,
                "rollout_turns": [
                    {
                        "turn": turn["turn"],
                        "turn_kind": turn["turn_kind"],
                        "turn_prompt": turn.get("turn_obs") or "",
                        "response": turn.get("response") or "",
                        "decode_has_tool_call": bool(turn.get("decode_has_tool_call")),
                    }
                    for turn in ordered_turns
                ],
                "reward_metrics": reward_metrics,
            }
            jsonl_rows.append(json.dumps(monitor_payload, ensure_ascii=False))

        (monitor_dir / f"{step_tag}.txt").write_text("\n".join(step_text) + "\n")
        (monitor_dir / f"{step_tag}.jsonl").write_text("\n".join(jsonl_rows) + "\n")
    except Exception as exc:  # noqa: BLE001
        # Monitoring must never fail or alter rollout generation.
        logger.warning("Failed to dump raw agent rollouts: %s", exc)
