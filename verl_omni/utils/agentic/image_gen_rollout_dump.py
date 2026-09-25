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
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from verl_omni.tools.trajectory import (
    build_trajectory_relpath,
    resolve_run_dir,
    traj_dir_exclusive,
    write_json_atomic,
)
from verl_omni.utils.agentic.image_gen_rollout_parse import (
    extract_generate_image_prompts,
    last_user_prompt,
    split_rollout_turns,
    turn_kind,
    unpad_left_ids,
)
from verl_omni.utils.metrics_utils import AgenticRewardMetrics

try:  # POSIX-only: serialises concurrent appends to one step monitor file.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

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
    """Index images already written by the live tool; never create empty folders.

    Args:
        decoded_response: Decoded assistant text (for tool prompts).
        run_dir: Artifact run directory.
        relpath: Trajectory-relative image folder.
        user_prompt: Dataset user request stored in ``meta.json``.

    Returns:
        Existing image paths under that folder, or ``[]`` if none.
    """
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
    # Same folder-wide lock as ``image_gen._save_images``: this rewrite races the
    # live tool whenever a worker post-processes a rollout while another tool
    # call still writes into the same trajectory folder.
    with traj_dir_exclusive(target_dir):
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
        write_json_atomic(meta_path, meta)
    return image_paths


def discard_invalid_rollouts(output: Any) -> None:
    """Zero ``response_mask`` for rows that never produced ``generate_image``.

    Args:
        output: Rollout ``DataProto``. Prefers stamps on ``non_tensor_batch``.

    Returns:
        None.
    """
    valid = output.non_tensor_batch.get("rollout_valid")
    has_gen = output.non_tensor_batch.get("rollout_has_generate")
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
        is_valid = _row_has_generate(valid=valid, has_gen=has_gen, n_gen=n_gen, index=i)
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


def _row_int(values: Any, index: int) -> int | None:
    if values is None:
        return None
    try:
        return int(np.asarray(values[index]).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return None


def _row_has_generate(*, valid: Any, has_gen: Any, n_gen: Any, index: int) -> bool:
    """Fail-closed when a stamp is present but unparsable; True only if generate count >= 1."""
    for values, predicate in (
        (valid, lambda v: v == 1),
        (has_gen, lambda v: v == 1),
        (n_gen, lambda v: v >= 1),
    ):
        parsed = _row_int(values, index)
        if parsed is None:
            if values is not None:
                # Key present for the batch but this row failed to parse → treat as invalid.
                try:
                    _ = values[index]
                except Exception:  # noqa: BLE001
                    continue
                return False
            continue
        return bool(predicate(parsed))
    # No validity stamps at all (non-agentic loop) → keep the row.
    return True


#: ``sample_<index>`` or ``sample_<index>.<rollout_n>``, the two relpath shapes this
#: module writes. The index is stamped by the agent loop from the *dataset* index, so
#: the relpath is the one place that still names the sample after ``output`` has lost
#: ``index`` (see :func:`_resolve_sample_relpath`).
_SAMPLE_RELPATH_RE = re.compile(r"sample_(-?\d+)(?:\.(\d+))?$")


def _sample_index_from_relpath(relpath: str) -> int | None:
    """Return the dataset index a trajectory relpath names, or ``None``.

    Args:
        relpath: ``…/sample_<index>[.<rollout_n>]``.

    Returns:
        The index, or ``None`` when the path is not in that shape.
    """
    match = _SAMPLE_RELPATH_RE.fullmatch(Path(relpath).name)
    return int(match.group(1)) if match else None


def _row_task_type(values: Any, index: int) -> str:
    """Return the ``agentic_task_type`` for one batch row, normalised or empty.

    Args:
        values: The output non-tensor batch entry, or ``None``.
        index: Row index.

    Returns:
        Lower-cased task type, or ``""`` when the row carries none.
    """
    if values is None:
        return ""
    try:
        raw = values[index]
    except (TypeError, IndexError, KeyError):
        return ""
    return str(raw or "").strip().lower()


def _resolve_sample_relpath(
    *,
    i: int,
    sample_index: Any,
    sample_key: str,
    rollout_counts: dict[str, int],
    live_relpaths: Any,
    step_i: int,
    validate: bool = False,
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
        # Parse sample_index[.rollout_n] from ``…/sample_6.03`` when present, and the
        # index alone from the val holdout form ``…/sample_9004``.
        match = _SAMPLE_RELPATH_RE.fullmatch(Path(relpath).name)
        if match:
            sample_key = match.group(1)
            rollout_n = int(match.group(2)) if match.group(2) is not None else rollout_counts.get(sample_key, 0)
        else:
            rollout_n = rollout_counts.get(sample_key, 0)
    else:
        rollout_n = 0 if validate else rollout_counts.get(sample_key, 0)
        relpath = build_trajectory_relpath(
            step=step_i,
            sample_index=sample_index,
            rollout_n=rollout_n,
            validate=validate,
        )
    return relpath, sample_key, rollout_n


def _annotate_ordered_turns(
    rollout_turns: list[dict[str, Any]], user_prompt: str, *, task_type: str = ""
) -> list[dict[str, Any]]:
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
            task_type=task_type,
            # The turn's ``response_mask`` composition, so the label is checked
            # against the mask the trainer used rather than sniffed from the text.
            advantage=str(turn.get("turn_advantage") or ""),
        )
    return [
        {
            "turn": t.get("turn"),
            "turn_kind": t.get("turn_kind"),
            # Who owns the tokens: mask=1 policy spans versus the mask=0 observation
            # and any harness-injected cue. This is the advantage view of the turn.
            "turn_advantage": t.get("turn_advantage") or "",
            "policy_tokens": t.get("policy_tokens"),
            "env_tokens": t.get("env_tokens"),
            "injected_cue": bool(t.get("injected_cue")),
            "turn_prompt": t.get("turn_prompt") or "",
            "turn_obs": t.get("turn_obs") or "",
            "tool_name": t.get("tool_name") or "",
            "tool_prompt": t.get("tool_prompt") or "",
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
    task_type: str = "",
    raw_prompt: str = "",
    raw_response: str = "",
) -> dict[str, Any]:
    return {
        "trajectory_relpath": relpath,
        "image_dir": image_dir,
        "step": step_i,
        "sample_index": int(sample_index) if str(sample_index).lstrip("-").isdigit() else str(sample_index),
        "rollout_n": rollout_n,
        # ``plan`` / ``reflect`` for this row. Recorded because it is what
        # ``turn_kind`` keys its plan labels off, so a dump that shows no ``plan``
        # label can be told apart from one whose row never claimed the plan protocol.
        "task_type": task_type,
        "user_prompt": user_prompt,
        # The literal episode the trainer held: the prompt tokens the policy read and the
        # response tokens it produced, decoded in order. ``rollout_turns`` below is a
        # mask-annotated *interpretation* of this; these two are the raw ``output`` row.
        "raw_prompt": raw_prompt,
        "raw_response": raw_response,
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
        # Advantage view of the rollout: the mask=1 tokens the optimizer actually saw.
        # A rollout whose ``num_policy_tokens`` is small relative to its length is
        # mostly harness scaffolding and observation, however long the transcript is.
        "num_policy_tokens": sum(int(turn.get("policy_tokens") or 0) for turn in ordered_turns),
        "num_injected_cue_turns": sum(1 for turn in ordered_turns if turn.get("injected_cue")),
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


def _compact_turn_record(turn: dict[str, Any]) -> dict[str, Any]:
    """Project a trajectory turn into the compact ``hermes_actions`` row shape.

    ``hermes_actions`` keeps the short env observation in ``turn_prompt`` (not the
    full chat template) and drops ``decode`` to stay readable, so ``tool_name`` and
    ``tool_prompt`` have to be carried over explicitly — otherwise the rewritten
    diffusion prompt exists only inside the escaped JSON in ``decode``.
    """
    return {
        "turn": turn["turn"],
        "turn_kind": turn["turn_kind"],
        # Kept next to the label: the dump's labels are mask-consistent, and this is
        # the mask the consistency was checked against.
        "turn_advantage": turn.get("turn_advantage") or "",
        "policy_tokens": turn.get("policy_tokens"),
        "env_tokens": turn.get("env_tokens"),
        "injected_cue": bool(turn.get("injected_cue")),
        "turn_prompt": turn.get("turn_obs") or "",
        "tool_name": turn.get("tool_name") or "",
        "tool_prompt": turn.get("tool_prompt") or "",
        "response": turn.get("response") or "",
        "decode_has_tool_call": bool(turn.get("decode_has_tool_call")),
    }


def _raw_text_block(label: str, text: str) -> list[str]:
    """Render one literal token-stream section, indented and never an empty body.

    Args:
        label: Section heading.
        text: Decoded token text.

    Returns:
        Lines to extend a dump with.
    """
    return [f"{label}:", *[f"  {line}" for line in (text.splitlines() or [""])]]


def _dump_header_lines(
    *,
    relpath: str,
    sample_index: Any,
    rollout_n: int,
    user_prompt: str,
    task_type: str,
    prefixed: bool = False,
) -> list[str]:
    """Return the identifying header for one rollout's text dump.

    Args:
        relpath: Trajectory relpath (or bare sample name for the step monitor).
        sample_index: Dataset sample index actually used for this row.
        rollout_n: Rollout group index.
        user_prompt: The user turn.
        task_type: ``plan`` / ``reflect`` / ``""``.
        prefixed: Wrap in a ``=== ===`` banner (the step monitor holds many rollouts).

    Returns:
        Header lines.
    """
    ident = f"{relpath}  sample={sample_index} rollout_n={rollout_n}" if prefixed else relpath
    first = f"=== {ident} ===" if prefixed else f"relpath={ident}"
    return [
        first,
        f"task_type={task_type or 'unspecified'} sample_index={sample_index} rollout_n={rollout_n}",
        f"user_prompt: {user_prompt}",
    ]


def _raw_sections(raw_prompt: str, raw_response: str) -> list[str]:
    """Return the literal prompt/response sections plus the annotated-turn preamble.

    These come first so the file can be read as the episode the trainer optimised. The
    per-turn blocks that follow re-present the same tokens with the ``response_mask``
    made explicit, which is what the raw decode cannot show.

    Args:
        raw_prompt: Decoded prompt token ids.
        raw_response: Decoded response token ids.

    Returns:
        Lines to extend a dump with.
    """
    return [
        "",
        *_raw_text_block("raw_prompt (decoded prompt tokens: the exact input)", raw_prompt),
        "",
        *_raw_text_block("raw_response (decoded response tokens: the exact episode output)", raw_response),
        "",
        "# annotated_turns: the same tokens split by response_mask. advantage=policy spans",
        "# are mask=1 (trainable); advantage=cue/env spans are mask=0 (observation or a",
        "# harness-injected cue, outside the advantage).",
        "annotated_turns:",
    ]


def _format_turn_text_block(turn: dict[str, Any]) -> list[str]:
    t = int(turn["turn"])
    # The full chat-templated model input is deliberately *not* printed per turn: it is
    # the whole system prompt plus every earlier turn, so printing it each time repeats
    # the same ~40 lines and buries the episode. It stays in the JSON payload as
    # ``turn_prompt``; the text dump leads with the literal raw prompt/response instead.
    turn_obs = turn.get("turn_obs") or ""
    response = turn.get("response") or ""
    decode = turn.get("decode") or ""
    kind = turn.get("turn_kind") or "other"
    tool_prompt = turn.get("tool_prompt") or ""
    advantage = turn.get("turn_advantage") or "?"
    header = (
        f"  turn={t} kind={kind} advantage={advantage} "
        f"policy_tokens={turn.get('policy_tokens')} decode_has_tool_call={turn['decode_has_tool_call']}"
    )
    lines = [header]
    if tool_prompt:
        # Only generate turns carry one; shown here so a rewrite chain is readable
        # without unescaping the JSON tool call out of ``decode``.
        lines += [f"    turn_{t}_tool_prompt:", *[f"      {line}" for line in tool_prompt.splitlines()]]
    # Mark the mask on the two text bodies a reader would otherwise take at face value:
    # ``response`` on a cue turn is harness text the optimizer never saw, and ``decode``
    # is the trainable span. Naming the mask here keeps the transcript honest without
    # changing the field names the JSON payload uses.
    response_label = (
        f"    turn_{t}_response_masked0_injected_cue:" if turn.get("injected_cue") else f"    turn_{t}_response:"
    )
    decode_label = "    decode_masked1_policy:" if turn.get("policy_tokens") else "    decode:"
    lines += [
        f"    turn_{t}_obs:",
        *[f"      {line}" for line in (turn_obs.splitlines() or [""])],
        response_label,
        *[f"      {line}" for line in (response.splitlines() or [""])],
        decode_label,
        *[f"      {line}" for line in (decode.splitlines() or [""])],
    ]
    return lines


def dump_raw_rollouts(
    *, tokenizer: Any, output: Any, step: Any, validate: bool = False, write_monitor: bool = True
) -> None:
    """Write hermes_actions JSONL and per-step trajectory dumps.

    Args:
        tokenizer: Tokenizer used to decode responses.
        output: Rollout ``DataProto``.
        step: Global step, or ``None`` for ``step_unknown``.
        validate: Validation batch — relpaths omit the ``.<nn>`` group suffix.
        write_monitor: Also (re)write ``hermes_actions/step_*.{jsonl,txt}``. Holdout
            viz passes ``False`` so it cannot truncate the step monitor that the
            V1 workers append to.

    Returns:
        None.
    """
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
        if write_monitor:
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
                validate=validate,
            )
            rollout_counts[sample_key] = max(rollout_counts.get(sample_key, 0), int(rollout_n) + 1)
            # The payload's ``sample_index`` must name the sample the artifacts name.
            # ``indices`` above reads ``output.non_tensor_batch["index"]``, which the
            # parent ``_postprocess`` only forwards when ``reward_loop_worker_handles``
            # is None; with the agent reward loop enabled the key is absent and the
            # lookup falls back to ``np.arange`` — so the 9001-9004 holdout was dumped
            # with ``sample_index`` 0,1,2,3 beside folders named 9001-9004. The relpath
            # is stamped by the loop from the real index, so it wins.
            relpath_index = _sample_index_from_relpath(relpath)
            if relpath_index is not None:
                sample_index = relpath_index
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
            # ``agentic_task_type`` is stamped into the loop's ``extra_fields`` and the
            # parent ``_postprocess`` copies every extra-field key into the output's
            # non-tensor batch, so it is available here even when the input non-tensor
            # batch (and with it ``extra_info``) is not forwarded. ``turn_kind`` no longer
            # branches on it — labels are read off the turn's own text — so this only
            # stamps the row's provenance onto the dump header.
            task_type = _row_task_type(output.non_tensor_batch.get("agentic_task_type"), i)
            ordered_turns = _annotate_ordered_turns(rollout_turns, user_prompt, task_type=task_type)
            # The literal token streams: ``raw_response`` is the whole episode exactly as
            # the agent loop produced it, before any turn splitting or relabelling.
            raw_prompt_text = "" if prompt_ids is None else tokenizer.decode(prompt_ids, skip_special_tokens=False)
            payload = _build_trajectory_payload(
                relpath=relpath,
                image_dir=image_dir,
                step_i=step_i,
                sample_index=sample_index,
                rollout_n=rollout_n,
                user_prompt=user_prompt,
                ordered_turns=ordered_turns,
                image_paths=image_paths,
                task_type=task_type,
                raw_prompt=raw_prompt_text,
                raw_response=decoded_response,
            )
            reward_metrics = AgenticRewardMetrics.for_rollout(output, i)

            name = Path(relpath).name
            (trajectory_dir / f"{name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
            trajectory_text = _dump_header_lines(
                relpath=relpath,
                sample_index=sample_index,
                rollout_n=rollout_n,
                user_prompt=user_prompt,
                task_type=task_type,
            )
            trajectory_text += _raw_sections(raw_prompt_text, decoded_response)
            step_text.extend(
                _dump_header_lines(
                    relpath=name,
                    sample_index=sample_index,
                    rollout_n=rollout_n,
                    user_prompt=user_prompt,
                    task_type=task_type,
                    prefixed=True,
                )
            )
            step_text += _raw_sections(raw_prompt_text, decoded_response)
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
                "rollout_turns": [_compact_turn_record(turn) for turn in ordered_turns],
                "reward_metrics": reward_metrics,
            }
            jsonl_rows.append(json.dumps(monitor_payload, ensure_ascii=False))

        if write_monitor:
            (monitor_dir / f"{step_tag}.txt").write_text("\n".join(step_text) + "\n")
            (monitor_dir / f"{step_tag}.jsonl").write_text("\n".join(jsonl_rows) + "\n")
    except Exception as exc:  # noqa: BLE001
        # Monitoring must never fail or alter rollout generation.
        logger.warning("Failed to dump raw agent rollouts: %s", exc)


def _append_step_monitor(monitor_dir: Path, step_tag: str, *, jsonl_line: str, text_block: str) -> None:
    """Append one rollout's row to the step-level ``hermes_actions`` files.

    V1 workers finish concurrently, so the step monitor is assembled by
    appending each rollout's block. ``flock`` keeps two writers from splicing a
    line into the middle of another's JSON object.
    """
    for suffix, content in ((".jsonl", f"{jsonl_line}\n"), (".txt", f"{text_block}\n")):
        path = monitor_dir / f"{step_tag}{suffix}"
        with open(path, "a", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _rollout_reward_metrics(extra_fields: Any, reward_score: Any) -> dict[str, float | int]:
    """Per-rollout scorer counters for one ``hermes_actions`` row (V1 path).

    Mirrors ``AgenticRewardMetrics.for_rollout`` for the data a TransferQueue
    worker holds (``reward_score`` + ``extra_fields``) instead of a DataProto row.

    Args:
        extra_fields: ``AgentLoopOutput.extra_fields`` (may nest
            ``reward_extra_info``).
        reward_score: Scalar reward written by ``_compute_score``.

    Returns:
        Compact dict of ``score`` and any ``ARTIFACT_KEYS`` present on the row.
    """
    metrics: dict[str, float | int] = {}
    if reward_score is not None:
        try:
            metrics["score"] = float(reward_score)
        except (TypeError, ValueError):
            pass
    sources: list[dict[str, Any]] = []
    if isinstance(extra_fields, dict):
        sources.append(extra_fields)
        reward_extra = extra_fields.get("reward_extra_info")
        if isinstance(reward_extra, dict):
            sources.append(reward_extra)
    for source in sources:
        for key in AgenticRewardMetrics.ARTIFACT_KEYS:
            if key in metrics or key not in source:
                continue
            value = source[key]
            try:
                metrics[key] = int(value) if key in AgenticRewardMetrics.INTEGER_KEYS else float(value)
            except (TypeError, ValueError):
                continue
    return metrics


def dump_rollout_artifacts(
    *,
    tokenizer: Any,
    step: Any,
    relpath: str,
    sample_index: Any,
    raw_prompt: Any = None,
    outputs: Any = None,
) -> None:
    """Dump one rollout on the V1 TransferQueue path.

    ``OmniAgentLoopManager.generate_sequences`` only dispatches on V1 (it never
    sees the returned batch), so the worker owning the rollout writes the
    monitoring artifacts. The layout matches the DataProto path exactly::

        rollout_trajectories/step_XXXXXX/sample_<index>.<nn>.{json,txt}   (train)
        rollout_trajectories/step_XXXXXX/sample_<index>.{json,txt}        (val)
        hermes_actions/step_XXXXXX.{jsonl,txt}                            (appended)

    Args:
        tokenizer: Tokenizer used to decode responses.
        step: Global step (``None`` → ``step_unknown``).
        relpath: Live trajectory relpath bound by ``_run_agent_loop``.
        sample_index: Dataset sample index.
        raw_prompt: Raw chat prompt, for the user request.
        outputs: ``AgentLoopOutput`` (or list) from the TransferQueue worker.

    Returns:
        None. Monitoring must never fail or alter rollout generation.
    """
    try:
        rows = list(outputs) if isinstance(outputs, list | tuple) else [outputs]
        rows = [row for row in rows if row is not None]
        if not rows:
            return
        final = rows[-1]
        response_ids = [token for row in rows for token in (row.response_ids or [])]
        response_mask = [mask for row in rows for mask in (row.response_mask or [])]
        if not response_ids:
            return
        prompt_ids = list(rows[0].prompt_ids or [])
        try:
            step_i = int(step) if step is not None else -1
        except (TypeError, ValueError):
            step_i = -1
        step_tag = f"step_{step_i:06d}" if step_i >= 0 else "step_unknown"
        run_dir = resolve_run_dir()
        trajectory_dir = run_dir / "rollout_trajectories" / step_tag
        monitor_dir = run_dir / "hermes_actions"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        monitor_dir.mkdir(parents=True, exist_ok=True)

        name = Path(relpath).name
        match = re.fullmatch(r"sample_(.+)\.(\d+)$", name)
        rollout_n = int(match.group(2)) if match else 0

        user_prompt = last_user_prompt(raw_prompt) if raw_prompt is not None else ""
        rollout_turns = split_rollout_turns(response_ids, response_mask, tokenizer, prompt_ids=prompt_ids)
        decoded_response = tokenizer.decode(list(response_ids), skip_special_tokens=False)
        image_paths = materialize_rollout_images(
            decoded_response=decoded_response,
            run_dir=run_dir,
            relpath=relpath,
            user_prompt=user_prompt,
        )
        image_dir = str(run_dir / "rollout_images" / relpath) if image_paths else ""
        task_type = str((getattr(final, "extra_fields", None) or {}).get("agentic_task_type") or "").strip().lower()
        ordered_turns = _annotate_ordered_turns(rollout_turns, user_prompt, task_type=task_type)
        raw_prompt_text = "" if not prompt_ids else tokenizer.decode(list(prompt_ids), skip_special_tokens=False)
        payload = _build_trajectory_payload(
            relpath=relpath,
            image_dir=image_dir,
            step_i=step_i,
            sample_index=sample_index,
            rollout_n=rollout_n,
            user_prompt=user_prompt,
            ordered_turns=ordered_turns,
            image_paths=image_paths,
            task_type=task_type,
            raw_prompt=raw_prompt_text,
            raw_response=decoded_response,
        )
        trajectory_text = _dump_header_lines(
            relpath=relpath,
            sample_index=sample_index,
            rollout_n=rollout_n,
            user_prompt=user_prompt,
            task_type=task_type,
        )
        trajectory_text += _raw_sections(raw_prompt_text, decoded_response)
        for turn in ordered_turns:
            trajectory_text.extend(_format_turn_text_block(turn))
        trajectory_text.append("")
        (trajectory_dir / f"{name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        (trajectory_dir / f"{name}.txt").write_text("\n".join(trajectory_text) + "\n")

        step_text = _dump_header_lines(
            relpath=name,
            sample_index=sample_index,
            rollout_n=rollout_n,
            user_prompt=user_prompt,
            task_type=task_type,
            prefixed=True,
        )
        step_text += _raw_sections(raw_prompt_text, decoded_response)
        for turn in ordered_turns:
            step_text.extend(_format_turn_text_block(turn))
        step_text.append("")
        monitor_payload = {
            **payload,
            "rollout_turns": [_compact_turn_record(turn) for turn in ordered_turns],
            "reward_metrics": _rollout_reward_metrics(
                getattr(final, "extra_fields", None),
                getattr(final, "reward_score", None),
            ),
        }
        _append_step_monitor(
            monitor_dir,
            step_tag,
            jsonl_line=json.dumps(monitor_payload, ensure_ascii=False),
            text_block="\n".join(step_text),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to dump rollout artifacts for %s: %s", relpath, exc)
