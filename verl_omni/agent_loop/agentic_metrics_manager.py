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

"""Stock AgentLoopManager with reward metrics and raw-rollout monitoring.

The agent implementation remains verl's registered ``tool_agent``. Monitoring
is done here, after generation, so it cannot alter, force, or replace tokens.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import ray
from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.utils import hf_tokenizer

from verl_omni.agent_loop.agentic_trajectory_context import (
    bind_run_artifact_env,
    build_trajectory_relpath,
    clear_good_enough_yes_reached,
    reset_active_trajectory_relpath,
    reset_active_user_prompt,
    resolve_run_dir,
    set_active_trajectory_relpath,
    set_active_user_prompt,
)

logger = logging.getLogger(__name__)

# WandB ``agentic_reward/*`` — only the scalar mix terms used in compute_score.
REWARD_COMPONENTS = (
    "reward_tool_call",
    "reward_correctness",
    "reward_aesthetics",
    "reward_done",
)
REWARD_ARTIFACT_FIELDS = (
    *REWARD_COMPONENTS,
    "num_hermes_tool_calls",
    "num_generate_image_prompts",
    "num_judge_image_calls",
    "judge_parse_ok",
    "judge_parse_fail",
    "judge_parse_ok_rate",
    "protocol_ok",
    "rewrite_after_yes",
    "reward_delta_c",
    "reward_rewrite_yes",
    "first_correctness",
    "first_judge_no",
    "rollout_valid",
)
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(?:\{.*?\"name\"\s*:\s*\"[^\"]+\".*?\}|"
    r"<function=[^>\s]+\s*>.*?</function>)\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_EXECUTED_TOOL_RESPONSE_RE = re.compile(r"\bagentic_(?:tool|reflect|judge)\s+ok=[01]\b", re.IGNORECASE)
_HERMES_PROMPT_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_QWEN_XML_PROMPT_RE = re.compile(
    r"<tool_call>\s*<function=generate_image\s*>.*?"
    r"<parameter=prompt\s*>\s*(.*?)\s*</parameter>.*?"
    r"</function>\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_JUDGE_CALL_RE = re.compile(r"<function=judge_image\b|\"name\"\s*:\s*\"judge_image\"", re.IGNORECASE)
_GEN_CALL_RE = re.compile(r"<function=generate_image\b|\"name\"\s*:\s*\"generate_image\"", re.IGNORECASE)
_AGENT_REFLECTION_RE = re.compile(r"\bReflection\s*:", re.IGNORECASE)
_VL_JUDGE_OBS_RE = re.compile(r"\b(?:VL judge|agentic_judge)\b", re.IGNORECASE)
_FORCED_REFLECTION_RE = re.compile(r"\bagentic_forced_reflection=1\b", re.IGNORECASE)
_STOP_DECISION_RE = re.compile(r"\bagentic_stop_decision_required=1\b", re.IGNORECASE)
_TERMINAL_DONE_RE = re.compile(r"(?is)^\s*(?:Reflection\s*:.*?)?Done\.\s*(?:<\|im_end\|>)?\s*$")


def _turn_kind(decode: str, turn_prompt: str, response: str = "") -> str:
    """Label turns so trajectory dumps make protocol stages grep-able."""
    resp = response or ""
    forced_context = f"{turn_prompt or ''}\n{resp}"
    if _JUDGE_CALL_RE.search(decode or ""):
        return "call_judge_image"
    if _GEN_CALL_RE.search(decode or ""):
        if _FORCED_REFLECTION_RE.search(resp) or _FORCED_REFLECTION_RE.search(turn_prompt or ""):
            return "agent_rewrite_after_forced_reflection"
        return "call_generate_image"
    if _TERMINAL_DONE_RE.search(decode or "") and _STOP_DECISION_RE.search(forced_context):
        if re.search(r"\bagentic_force_stop_max_passes=1\b", forced_context):
            return "agent_done_after_max_passes"
        return "agent_done_after_forced_reflection"
    if re.search(r"\bagentic_force_stop_max_passes=1\b", resp) or (
        not (decode or "").strip() and re.search(r"\bagentic_force_stop_max_passes=1\b", turn_prompt or "")
    ):
        return "forced_reflection_max_passes_stop_cue"
    if _FORCED_REFLECTION_RE.search(resp):
        if _STOP_DECISION_RE.search(resp):
            return "forced_reflection_stop_cue"
        return "forced_reflection_continue"
    if not (decode or "").strip() and _FORCED_REFLECTION_RE.search(turn_prompt or ""):
        if _STOP_DECISION_RE.search(turn_prompt or ""):
            return "forced_reflection_stop_cue"
        return "forced_reflection_continue"
    if _AGENT_REFLECTION_RE.search(decode or ""):
        if _GEN_CALL_RE.search(decode or ""):
            return "agent_reflection_rewrite"
        return "agent_reflection_done"
    if _VL_JUDGE_OBS_RE.search(turn_prompt or ""):
        return "after_judge_feedback"
    if "path=" in (turn_prompt or "") and "agentic_tool" in (turn_prompt or ""):
        return "after_generate_image"
    return "other"


def _extract_generate_image_prompts(decoded_response: str) -> list[str]:
    """Ordered prompts from Hermes JSON or Qwen3.5 XML tool calls."""
    found: list[tuple[int, str]] = []
    for match in _HERMES_PROMPT_RE.finditer(decoded_response or ""):
        raw = match.group(1)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if str(payload.get("name", "")).strip() != "generate_image":
            continue
        args = payload.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            continue
        prompt = args.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            found.append((match.start(), prompt.strip()))
    for match in _QWEN_XML_PROMPT_RE.finditer(decoded_response or ""):
        prompt = match.group(1).strip()
        if prompt:
            found.append((match.start(), prompt))
    return [prompt for _, prompt in sorted(found)]


def aggregate_agentic_reward_metrics(non_tensor_batch: dict[str, Any]) -> dict[str, float]:
    """Aggregate numeric reward extras already returned by verl's reward manager."""
    metrics: dict[str, float] = {}
    for key in REWARD_COMPONENTS:
        if key not in non_tensor_batch:
            continue
        values = np.asarray(non_tensor_batch[key], dtype=np.float64)
        if values.size == 0:
            continue
        prefix = f"agentic_reward/{key.removeprefix('reward_')}"
        metrics[f"{prefix}/mean"] = float(np.mean(values))
        metrics[f"{prefix}/min"] = float(np.min(values))
        metrics[f"{prefix}/max"] = float(np.max(values))
    return metrics


def _artifact_reward_metrics(output: Any, index: int) -> dict[str, float | int]:
    """Per-rollout scorer outputs for compact ``hermes_actions`` JSONL rows."""
    metrics: dict[str, float | int] = {}
    rm_scores = output.batch.get("rm_scores")
    if rm_scores is not None:
        # AgentLoopManager writes the scalar reward on the final valid response
        # token; the first token is normally zero. Sum the token-level tensor.
        value = np.asarray(rm_scores[index].detach().cpu()).sum()
        metrics["score"] = float(value)

    integer_fields = {
        "num_hermes_tool_calls",
        "num_generate_image_prompts",
        "num_judge_image_calls",
        "protocol_ok",
        "rollout_valid",
    }
    for key in REWARD_ARTIFACT_FIELDS:
        values = output.non_tensor_batch.get(key)
        if values is None:
            continue
        value = np.asarray(values[index]).reshape(-1)[0]
        metrics[key] = int(value) if key in integer_fields else float(value)
    return metrics


def _split_env_blob(blob: str) -> tuple[str, str]:
    """Split mask=0 env text into ``(turn_prompt, response)``.

    ``turn_prompt`` = tool / user obs the policy reads.
    ``response`` = injected assistant text (forced ``Reflection:…``), if any.
    These can diverge from the previous turn's ``decode`` because the agent loop
    may inject Reflection after ``judge_image`` without sampling it from the policy.
    """
    text = (blob or "").strip()
    if not text:
        return "", ""
    force_idx = -1
    for match in re.finditer(r"(?:^|\n)\s*Reflection\s*:", text, re.IGNORECASE):
        # Prefer the forced marker when present.
        window = text[match.start() : match.start() + 400]
        if "agentic_forced_reflection=1" in window or force_idx < 0:
            force_idx = match.start()
            if "agentic_forced_reflection=1" in window:
                break
    if force_idx < 0 or not _FORCED_REFLECTION_RE.search(text):
        # No injected assistant response — entire blob is the next-turn prompt.
        return text, ""
    # Include any chat-template role tags immediately before Reflection.
    cut = force_idx
    preamble = text[:force_idx]
    # If the decode left a trailing bare ``assistant`` / think block before Reflection,
    # keep tool_response in turn_prompt and put Reflection(+trailing) in response.
    tool_end = preamble.rfind("</tool_response>")
    if tool_end >= 0:
        turn_prompt = text[: tool_end + len("</tool_response>")].strip()
        response = text[tool_end + len("</tool_response>") :].strip()
        # Drop leading role/think scaffolding noise from response but keep Reflection.
        refl = re.search(r"Reflection\s*:", response, re.IGNORECASE)
        if refl:
            response = response[refl.start() :].strip()
        return turn_prompt, response
    return text[:cut].strip(), text[cut:].strip()


def _unpad_left_ids(token_ids, pad_token_id: int | None) -> list[int]:
    """Strip left padding from prompt ids (verl left-pads prompts)."""
    ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    pads = {0}
    if pad_token_id is not None:
        pads.add(int(pad_token_id))
    start = 0
    while start < len(ids) and int(ids[start]) in pads:
        start += 1
    return [int(x) for x in ids[start:]]


def _turn_record(
    *,
    turn: int,
    turn_prompt: str,
    response: str,
    decode: str,
    turn_input: str = "",
) -> dict[str, Any]:
    return {
        "turn": turn,
        "turn_prompt": turn_prompt or "",
        "turn_input": turn_input or "",
        "decode": decode or "",
        "response": response or "",
        "decode_has_tool_call": "<tool_call>" in (decode or "").lower(),
    }


def split_assistant_rollouts(token_ids, response_mask, tokenizer) -> list[str]:
    """Decode contiguous model-token spans; tool observations have mask 0."""
    return [turn["decode"] for turn in split_rollout_turns(token_ids, response_mask, tokenizer)]


def split_rollout_turns(
    token_ids,
    response_mask,
    tokenizer,
    prompt_ids=None,
) -> list[dict[str, Any]]:
    """Split response into turns with explicit prompt / response / decode fields.

    ``response_mask==1`` → model tokens (``decode``).
    ``response_mask==0`` → env tokens, split into:
      - ``turn_prompt``: tool obs the policy conditions on (short env delta)
      - ``response``: forced ``Reflection:…`` (injected, not policy-sampled)

    When ``prompt_ids`` is provided (unpadded chat-templated prompt tokens), each
    turn also gets ``turn_input``: the exact decoded prefix the model saw before
    generating that turn (system + Tools schema + history), including special tokens.

    Keys per turn: ``turn``, ``turn_prompt``, ``turn_input``, ``decode``,
    ``response``, ``decode_has_tool_call``.
    """
    ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    mask = response_mask.tolist() if hasattr(response_mask, "tolist") else list(response_mask)
    prompt_prefix = (
        _unpad_left_ids(prompt_ids, getattr(tokenizer, "pad_token_id", None)) if prompt_ids is not None else None
    )
    turns: list[dict[str, Any]] = []
    current_model: list[int] = []
    current_tool: list[int] = []
    pending_prompt = ""
    pending_response = ""
    model_start = 0

    def _decode_input(response_prefix_len: int) -> str:
        if prompt_prefix is None:
            return ""
        prefix = prompt_prefix + [int(x) for x in ids[:response_prefix_len]]
        return tokenizer.decode(prefix, skip_special_tokens=False)

    def _flush_tool() -> None:
        nonlocal pending_prompt, pending_response, current_tool
        if not current_tool:
            return
        blob = tokenizer.decode(current_tool, skip_special_tokens=True).strip()
        current_tool = []
        prompt, response = _split_env_blob(blob)
        pending_prompt = prompt
        pending_response = response

    def _flush_model() -> None:
        nonlocal current_model, pending_prompt, pending_response
        if not current_model:
            return
        decode = tokenizer.decode(current_model, skip_special_tokens=False)
        turns.append(
            _turn_record(
                turn=len(turns) + 1,
                turn_prompt=pending_prompt,
                response=pending_response,
                decode=decode,
                turn_input=_decode_input(model_start),
            )
        )
        pending_prompt = ""
        pending_response = ""
        current_model = []

    for idx, (token_id, is_model_token) in enumerate(zip(ids, mask, strict=True)):
        if int(is_model_token) == 1:
            if current_tool:
                _flush_tool()
            if not current_model:
                model_start = idx
            current_model.append(int(token_id))
        else:
            if current_model:
                _flush_model()
            current_tool.append(int(token_id))
    if current_model:
        _flush_model()
    # Trailing env (e.g. final judge + forced Done with no further decode).
    if current_tool:
        _flush_tool()
        if pending_prompt or pending_response:
            turns.append(
                _turn_record(
                    turn=len(turns) + 1,
                    turn_prompt=pending_prompt,
                    response=pending_response,
                    decode="",
                    turn_input=_decode_input(len(ids)),
                )
            )
        pending_prompt = ""
        pending_response = ""
    return turns


def _last_user_prompt(raw_prompt: Any) -> str:
    messages = list(raw_prompt) if raw_prompt is not None else []
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            return "\n".join(
                str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text"
            )
        return str(content)
    return ""


def _run_dir() -> Path:
    return resolve_run_dir()


def _bind_run_artifact_env(config) -> None:
    """Derive the per-run artifact path from the trainer experiment name."""
    bind_run_artifact_env(config)


def _materialize_rollout_images(
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

    prompts = _extract_generate_image_prompts(decoded_response)
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


class AgenticAgentLoopWorker(AgentLoopWorker):
    """Worker-side hooks: trajectory bind + step kwargs for force-first curriculum.

    ``AgentLoopManager.generate_sequences`` dispatches to Ray ``AgentLoopWorker``s.
    Overrides on the Manager class never run per-rollout — they must live here.

    Also hard-binds agentic multi-turn defaults (Hermes + ``verl_omni/tools``)
    so launch recipes need not pass ``function_tool_path`` / ``format`` Hydra overrides.
    """

    # Agentic Hermes wire format (teacher-force + parsers assume hermes).
    _AGENTIC_TOOL_FORMAT = "hermes"
    # Frozen tools live under ``verl_omni/tools`` (parents[1] == verl_omni).
    _AGENTIC_FUNCTION_TOOLS = Path(__file__).resolve().parents[1] / "tools" / "image_gen.py"

    def __init__(self, config, *args, **kwargs):
        from omegaconf import open_dict

        # Resolve path without importing — importing would register @function_tool
        # decorators, then load_function_tools_from_path would re-exec the file and
        # raise "already registered".
        _bind_run_artifact_env(config)
        tool_path = self._AGENTIC_FUNCTION_TOOLS
        if not tool_path.is_file():
            raise FileNotFoundError(
                f"agentic function tools not found at {tool_path}. Expected verl_omni/tools/image_gen.py"
            )
        with open_dict(config.actor_rollout_ref.rollout.multi_turn):
            config.actor_rollout_ref.rollout.multi_turn.function_tool_path = str(tool_path)
            config.actor_rollout_ref.rollout.multi_turn.format = self._AGENTIC_TOOL_FORMAT
        super().__init__(config, *args, **kwargs)

    async def _run_agent_loop(
        self,
        sampling_params,
        trajectory,
        *,
        agent_name,
        trace=True,
        **kwargs,
    ):
        relpath = build_trajectory_relpath(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
        )
        raw_prompt = kwargs.get("raw_prompt")
        user_prompt = _last_user_prompt(raw_prompt) if raw_prompt is not None else ""
        path_token = set_active_trajectory_relpath(relpath)
        prompt_token = set_active_user_prompt(user_prompt)
        # Fresh trajectory: allow generate_image until the first good_enough=YES.
        clear_good_enough_yes_reached()
        # Force-first curriculum reads these inside AgenticToolAgentLoop.run().
        kwargs["_agentic_step"] = trajectory["step"]
        kwargs["_agentic_validate"] = trajectory["validate"]
        # Same id used by agentic_tool saves and post-hoc trajectory dumps.
        kwargs["_agentic_trajectory_relpath"] = relpath
        try:
            return await super()._run_agent_loop(
                sampling_params,
                trajectory,
                agent_name=agent_name,
                trace=trace,
                **kwargs,
            )
        finally:
            reset_active_user_prompt(prompt_token)
            reset_active_trajectory_relpath(path_token)


class AgenticMetricsAgentLoopManager(AgentLoopManager):
    """Use stock rollout management; observe outputs without changing them."""

    def __init__(self, *args, **kwargs):
        # Must set before AgentLoopManager.__init__ creates Ray workers.
        self.agent_loop_workers_class = ray.remote(AgenticAgentLoopWorker)
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        if config is not None:
            _bind_run_artifact_env(config)
        super().__init__(*args, **kwargs)
        model_path = self.model_config.get("tokenizer_path") or self.model_config.get("path")
        trust_remote_code = bool(self.model_config.get("trust_remote_code", False))
        self._monitor_tokenizer = hf_tokenizer(model_path, trust_remote_code=trust_remote_code)

    def _dump_raw_rollouts(self, prompts, output, step) -> None:
        """Write user prompt + raw assistant turns only."""
        try:
            responses = output.batch["responses"]
            response_masks = output.batch["response_mask"]
            raw_prompts = output.non_tensor_batch.get("raw_prompt")
            indices = output.non_tensor_batch.get("index", np.arange(len(responses)))
            step_i = int(step) if step is not None else -1
            step_tag = f"step_{step_i:06d}" if step_i >= 0 else "step_unknown"
            run_dir = _run_dir()
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
                # Prefer the live artifact id stamped by AgenticToolAgentLoop so
                # trajectory JSON / image folders match path= markers in tool obs.
                # Fallback: renumber by batch order (legacy; can diverge if workers
                # reorder outputs relative to dispatch).
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
                rollout_counts[sample_key] = max(rollout_counts.get(sample_key, 0), int(rollout_n) + 1)
                user_prompt = _last_user_prompt(raw_prompts[i]) if raw_prompts is not None else ""
                prompt_ids = None
                if "prompts" in output.batch:
                    prompt_ids = _unpad_left_ids(
                        output.batch["prompts"][i],
                        getattr(self._monitor_tokenizer, "pad_token_id", None),
                    )
                rollout_turns = split_rollout_turns(
                    responses[i],
                    response_masks[i],
                    self._monitor_tokenizer,
                    prompt_ids=prompt_ids,
                )
                if rollout_turns and not rollout_turns[0].get("turn_prompt"):
                    rollout_turns[0]["turn_prompt"] = user_prompt
                decoded_response = self._monitor_tokenizer.decode(
                    responses[i].tolist(),
                    skip_special_tokens=False,
                )
                image_paths = _materialize_rollout_images(
                    decoded_response=decoded_response,
                    run_dir=run_dir,
                    relpath=relpath,
                    user_prompt=user_prompt,
                )
                image_dir = str(run_dir / "rollout_images" / relpath) if image_paths else ""
                # Prefer the full chat-templated model input in ``turn_prompt`` (system +
                # Tools schema + history). Keep the short env delta in ``turn_obs``.
                for turn in rollout_turns:
                    turn_obs = turn.get("turn_prompt") or ""
                    turn_input = turn.get("turn_input") or ""
                    turn["turn_obs"] = turn_obs
                    if turn_input:
                        turn["turn_prompt"] = turn_input
                    turn["turn_kind"] = _turn_kind(
                        turn.get("decode") or "",
                        turn_obs,
                        turn.get("response") or "",
                    )
                ordered_turns = [
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
                payload = {
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
                                r"path=((?:/|[A-Za-z]:\\)[^\s\"']+\.(?:png|jpg|jpeg|webp))",
                                turn.get("turn_obs") or "",
                                flags=re.IGNORECASE,
                            )
                        }
                    ),
                    "num_tool_calls_executed": sum(
                        len(_EXECUTED_TOOL_RESPONSE_RE.findall(turn.get("turn_obs") or "")) for turn in ordered_turns
                    ),
                    "num_forced_tool_calls": 0,
                    "num_voluntary_tool_calls": sum(
                        len(_TOOL_CALL_RE.findall(turn.get("decode") or "")) for turn in ordered_turns
                    ),
                    # Legacy field name retained for downstream dashboards.
                    "num_voluntary_hermes": sum(
                        len(_TOOL_CALL_RE.findall(turn.get("decode") or "")) for turn in ordered_turns
                    ),
                }
                reward_metrics = _artifact_reward_metrics(output, i)

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
                    t = int(turn["turn"])
                    turn_prompt = turn.get("turn_prompt") or ""
                    response = turn.get("response") or ""
                    decode = turn.get("decode") or ""
                    kind = turn.get("turn_kind") or "other"
                    header = f"  turn={t} kind={kind} decode_has_tool_call={turn['decode_has_tool_call']}"
                    block = [
                        header,
                        f"    turn_{t}_prompt:",
                        *[f"      {line}" for line in (turn_prompt.splitlines() or [""])],
                        f"    turn_{t}_response:",
                        *[f"      {line}" for line in (response.splitlines() or [""])],
                        "    decode:",
                        *[f"      {line}" for line in (decode.splitlines() or [""])],
                    ]
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

    def generate_sequences(self, prompts):
        step = prompts.meta_info.get("global_steps")
        output = super().generate_sequences(prompts)
        # Dump before discard so hermes_actions shows real policy decodes
        # (discard zeros response_mask, which would hide tool-less prose as env text).
        self._dump_raw_rollouts(prompts, output, step)
        self._discard_invalid_rollouts(output)
        metrics = aggregate_agentic_reward_metrics(output.non_tensor_batch)
        if metrics:
            try:
                import wandb

                if wandb.run is not None:
                    # The trainer's Tracking.log call commits this same global step.
                    wandb.log(metrics, step=int(step) if step is not None else None, commit=False)
            except Exception as exc:  # noqa: BLE001
                # Logging must never fail or alter rollout generation.
                logger.warning("Failed to log agentic reward metrics to W&B: %s", exc)
        return output

    @staticmethod
    def _discard_invalid_rollouts(output: Any) -> None:
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
