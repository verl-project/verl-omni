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

"""Rollout turn parsing helpers for agentic image-gen monitoring."""

from __future__ import annotations

import json
import re
from typing import Any


def turn_kind(decode: str, turn_prompt: str, response: str = "") -> str:
    """Label turns so trajectory dumps make protocol stages grep-able."""
    resp = response or ""
    forced_context = f"{turn_prompt or ''}\n{resp}"
    if re.search(r"<function=judge_image\b|\"name\"\s*:\s*\"judge_image\"", decode or "", re.IGNORECASE):
        return "call_judge_image"
    if re.search(r"<function=generate_image\b|\"name\"\s*:\s*\"generate_image\"", decode or "", re.IGNORECASE):
        if re.search(r"\bagentic_forced_reflection=1\b", resp, re.IGNORECASE) or re.search(
            r"\bagentic_forced_reflection=1\b", turn_prompt or "", re.IGNORECASE
        ):
            return "agent_rewrite_after_forced_reflection"
        return "call_generate_image"
    if re.search(
        r"(?is)^\s*(?:Reflection\s*:.*?)?Done\.\s*(?:<\|im_end\|>)?\s*$",
        decode or "",
    ) and re.search(r"\bagentic_stop_decision_required=1\b", forced_context, re.IGNORECASE):
        if re.search(r"\bagentic_force_stop_max_passes=1\b", forced_context):
            return "agent_done_after_max_passes"
        return "agent_done_after_forced_reflection"
    if re.search(r"\bagentic_force_stop_max_passes=1\b", resp) or (
        not (decode or "").strip() and re.search(r"\bagentic_force_stop_max_passes=1\b", turn_prompt or "")
    ):
        return "forced_reflection_max_passes_stop_cue"
    if re.search(r"\bagentic_forced_reflection=1\b", resp, re.IGNORECASE):
        if re.search(r"\bagentic_stop_decision_required=1\b", resp, re.IGNORECASE):
            return "forced_reflection_stop_cue"
        return "forced_reflection_continue"
    if not (decode or "").strip() and re.search(r"\bagentic_forced_reflection=1\b", turn_prompt or "", re.IGNORECASE):
        if re.search(r"\bagentic_stop_decision_required=1\b", turn_prompt or "", re.IGNORECASE):
            return "forced_reflection_stop_cue"
        return "forced_reflection_continue"
    if re.search(r"\bReflection\s*:", decode or "", re.IGNORECASE):
        if re.search(
            r"<function=generate_image\b|\"name\"\s*:\s*\"generate_image\"",
            decode or "",
            re.IGNORECASE,
        ):
            return "agent_reflection_rewrite"
        return "agent_reflection_done"
    if re.search(r"\b(?:VL judge|agentic_judge)\b", turn_prompt or "", re.IGNORECASE):
        return "after_judge_feedback"
    if "path=" in (turn_prompt or "") and "agentic_tool" in (turn_prompt or ""):
        return "after_generate_image"
    return "other"


def extract_generate_image_prompts(decoded_response: str) -> list[str]:
    """Ordered prompts from Hermes JSON or Qwen3.5 XML tool calls."""
    found: list[tuple[int, str]] = []
    hermes_pat = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    for match in re.finditer(hermes_pat, decoded_response or "", re.IGNORECASE | re.DOTALL):
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
    qwen_pat = (
        r"<tool_call>\s*<function=generate_image\s*>.*?"
        r"<parameter=prompt\s*>\s*(.*?)\s*</parameter>.*?"
        r"</function>\s*</tool_call>"
    )
    for match in re.finditer(qwen_pat, decoded_response or "", re.IGNORECASE | re.DOTALL):
        prompt = match.group(1).strip()
        if prompt:
            found.append((match.start(), prompt))
    return [prompt for _, prompt in sorted(found)]


def split_env_blob(blob: str) -> tuple[str, str]:
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
    if force_idx < 0 or not re.search(r"\bagentic_forced_reflection=1\b", text, re.IGNORECASE):
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


def unpad_left_ids(token_ids, pad_token_id: int | None) -> list[int]:
    """Strip left padding from prompt ids (verl left-pads prompts)."""
    ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    pads = {0}
    if pad_token_id is not None:
        pads.add(int(pad_token_id))
    start = 0
    while start < len(ids) and int(ids[start]) in pads:
        start += 1
    return [int(x) for x in ids[start:]]


def turn_record(
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


class _RolloutTurnSplitter:
    """Stateful splitter for ``response_mask`` model / env spans."""

    def __init__(self, token_ids, response_mask, tokenizer, prompt_ids=None):
        self.ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
        self.mask = response_mask.tolist() if hasattr(response_mask, "tolist") else list(response_mask)
        self.tokenizer = tokenizer
        self.prompt_prefix = (
            unpad_left_ids(prompt_ids, getattr(tokenizer, "pad_token_id", None)) if prompt_ids is not None else None
        )
        self.turns: list[dict[str, Any]] = []
        self.current_model: list[int] = []
        self.current_tool: list[int] = []
        self.pending_prompt = ""
        self.pending_response = ""
        self.model_start = 0

    def _decode_input(self, response_prefix_len: int) -> str:
        if self.prompt_prefix is None:
            return ""
        prefix = self.prompt_prefix + [int(x) for x in self.ids[:response_prefix_len]]
        return self.tokenizer.decode(prefix, skip_special_tokens=False)

    def _flush_tool(self) -> None:
        if not self.current_tool:
            return
        blob = self.tokenizer.decode(self.current_tool, skip_special_tokens=True).strip()
        self.current_tool = []
        prompt, response = split_env_blob(blob)
        self.pending_prompt = prompt
        self.pending_response = response

    def _flush_model(self) -> None:
        if not self.current_model:
            return
        decode = self.tokenizer.decode(self.current_model, skip_special_tokens=False)
        self.turns.append(
            turn_record(
                turn=len(self.turns) + 1,
                turn_prompt=self.pending_prompt,
                response=self.pending_response,
                decode=decode,
                turn_input=self._decode_input(self.model_start),
            )
        )
        self.pending_prompt = ""
        self.pending_response = ""
        self.current_model = []

    def run(self) -> list[dict[str, Any]]:
        for idx, (token_id, is_model_token) in enumerate(zip(self.ids, self.mask, strict=True)):
            if int(is_model_token) == 1:
                if self.current_tool:
                    self._flush_tool()
                if not self.current_model:
                    self.model_start = idx
                self.current_model.append(int(token_id))
            else:
                if self.current_model:
                    self._flush_model()
                self.current_tool.append(int(token_id))
        if self.current_model:
            self._flush_model()
        # Trailing env (e.g. final judge + forced Done with no further decode).
        if self.current_tool:
            self._flush_tool()
            if self.pending_prompt or self.pending_response:
                self.turns.append(
                    turn_record(
                        turn=len(self.turns) + 1,
                        turn_prompt=self.pending_prompt,
                        response=self.pending_response,
                        decode="",
                        turn_input=self._decode_input(len(self.ids)),
                    )
                )
            self.pending_prompt = ""
            self.pending_response = ""
        return self.turns


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
    return _RolloutTurnSplitter(token_ids, response_mask, tokenizer, prompt_ids=prompt_ids).run()


def last_user_prompt(raw_prompt: Any) -> str:
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
