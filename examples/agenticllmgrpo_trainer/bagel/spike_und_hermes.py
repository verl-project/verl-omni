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
"""UND Hermes serving spike for published Bagel (fail-closed; no Qwen fallback).

Exit 0 only when a Bagel replica emits ``<tool_call>`` / ``generate_image``.
``bagel_single_stage`` GEN deploy yaml is not UND proof.

Environment:
  BAGEL_MODEL_PATH   local Bagel checkpoint (tokenizer + specials)
  BAGEL_UND_URL      optional OpenAI-compatible chat URL for a live replica
  BAGEL_SPIKE_OFFLINE_SCHEMA=1  tokenizer/schema only (CI); does not prove serving

Serve AR (bagel_think) first:
  bash examples/agenticllmgrpo_trainer/bagel/run_bagel_und_ar_serve.sh
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

# Keep schema/token constants local so offline mode does not import verl_omni
# pipelines (those pull vllm-omni diffusion → CUDA init).
GENERATE_IMAGE_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": "Generate an image from a text prompt using the Bagel GEN pathway.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Text prompt for image generation."},
            },
            "required": ["prompt"],
        },
    },
}
HERMES_SPECIAL_TOKENS: tuple[str, ...] = (
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
)
# Fallback protocol text, used only when the training parquet cannot be read. Kept verbatim
# from the dataset's system turn so the schema/no-schema comparison is about the ``<tools>``
# block and nothing else.
UND_SYSTEM_PROMPT = (
    "You are a visual creation agent with two tools:\n"
    "1) generate_image — create an image from a complete diffusion prompt\n"
    "2) judge_image — inspect the last generated image and return structured feedback\n"
    "\n"
    "Protocol:\n"
    "1. Write a short numbered plan of at most three complete subtask image prompts.\n"
    "2. Call generate_image once per planned subtask, in order.\n"
    "3. After the final image, call judge_image on that image.\n"
    "4. Reflect briefly on the feedback and finish with Done.\n"
    "\n"
    "Do not judge between subtasks or generate more images than the plan lists."
)
# Where the trainer's UND prompts come from; the spike reads a row so its prompt cannot drift
# from what the rollouts actually saw.
DEFAULT_UND_PARQUET = "outputs/data/agentic_unicot/train.parquet"

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_DONE_RE = re.compile(r"\bDone\.\s*$", re.IGNORECASE)
# The published checkpoint often samples the Hermes *payload* without the surrounding
# ``<tool_call>`` tags, either bare or behind a short fenced label. This mirrors
# ``bagel_corl_lib.parse_und_tool_call`` so the spike scores a rollout the same way the
# harness does; a tagged-only classifier here would report ``continue`` for a turn the
# trainer actually executed, and the sweep would under-count GEN hits.
_ROLE_ECHO_RE = re.compile(r"^\s*(?:assistant|system|user)\s*[:\-]?\s*", re.IGNORECASE)
_LEADING_MARKER_RE = re.compile(r"^\s*(?:<[^>{}\n]{0,48}>\s*)+")
_FENCED_PREAMBLE_LIMIT = 120
# Mirrors ``bagel_corl_lib._FENCED_BLOCK_RE``. Iterating blocks matters: the checkpoint numbers
# its steps and fences one JSON object per step, so the plan is the *first* fence and the
# ``generate_image`` call is a later one.
_FENCED_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+.-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)
_INERT_BARE_TOOLS = frozenset({"judge_image"})
# Mirrors ``bagel_corl_lib._STOP_TOOL_NAMES``: a stop payload from the checkpoint's own dialect
# (e.g. ``{"name": "Done"}``) is the terminal signal, never an unsupported Qwen tool.
_STOP_TOOL_NAMES = frozenset(
    {"done", "done.", "stop", "finish", "finished", "finalize", "end", "complete", "terminate"}
)


def _normalize_call(payload: dict | None) -> dict | None:
    if not isinstance(payload, dict):
        return None
    if "name" not in payload and isinstance(payload.get("function"), dict):
        return payload["function"]
    return payload


def _leading_json_object(text: str) -> dict | None:
    """Decode the JSON object that starts ``text`` (``raw_decode`` ignores any tail)."""
    if not text.startswith("{"):
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(text)
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _parse_hermes_tool_call(text: str) -> dict | None:
    match = _TOOL_CALL_RE.search(text)
    if match is None:
        return None
    return _normalize_call(json.loads(match.group(1)))


def _named_payload(payload: dict | None) -> dict | None:
    """Accept ``{"function": {...}}`` (OpenAI shape) and require an action ``name``.

    Mirrors ``bagel_corl_lib._named_payload``. A payload with no ``name`` is *not* a call --
    that is the guard that keeps a JSON example (or the numbered plan's ``{"subtasks": [...]}``)
    from being scored as an executed tool turn.
    """
    if payload is None:
        return None
    if "name" not in payload and isinstance(payload.get("function"), dict):
        payload = payload["function"]
    return payload if "name" in payload else None


def _labelled_fenced_call(text: str) -> dict | None:
    """Parse the first *labelled* fenced block in ``text`` that names a tool.

    Mirrors ``bagel_corl_lib._labelled_fenced_call``: the label bound is measured **per block**,
    so a turn that numbers its steps and fences one object per step still counts, while a lone
    quoted example (whose label is the whole prose plan) does not.
    """
    cursor = 0
    for match in _FENCED_BLOCK_RE.finditer(text):
        label = text[cursor : match.start()]
        cursor = match.end()
        if len(label) > _FENCED_PREAMBLE_LIMIT or "{" in label or "}" in label:
            continue
        payload = _named_payload(_leading_json_object(match.group(1)))
        if payload is not None:
            return payload
    return None


def _parse_und_tool_call(text: str) -> dict | None:
    """Accept every dialect the harness accepts: tagged, bare, or fenced behind a label."""
    tagged = _parse_hermes_tool_call(text)
    if tagged is not None:
        return tagged
    head = _LEADING_MARKER_RE.sub("", _ROLE_ECHO_RE.sub("", text, count=1).lstrip())
    if head.startswith("{"):
        return _named_payload(_leading_json_object(head))
    if "```" not in text:
        return None
    return _labelled_fenced_call(text)


def _und_turn_kind(text: str) -> str:
    tagged = _parse_hermes_tool_call(text)
    call = tagged if tagged is not None else _parse_und_tool_call(text)
    if call is not None:
        name = str(call.get("name", ""))
        normalized = name.strip().strip(".\"' ").lower()
        if normalized == "generate_image":
            return "generate_image"
        if normalized in _STOP_TOOL_NAMES:
            # Mirrors ``bagel_corl_lib``: the checkpoint's own stop payload (``{"name": "Done"}``)
            # is the terminal signal, not an unsupported Qwen tool.
            return "done"
        if tagged is not None:
            raise ValueError(f"Bagel CoRL UND emitted unsupported tool {name!r}; Qwen/other tools are fail-closed")
        # The recipe's prompt asks the lane to judge after the last image and the RM does that
        # judging, so a non-action name is inert, not a hard failure.
        return "continue"
    if _DONE_RE.search(text.strip()):
        return "done"
    return "continue"


def inspect_tokenizer(model_path: str) -> dict[str, bool]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    vocab_specials = set(tokenizer.all_special_tokens)
    added = set(getattr(tokenizer, "additional_special_tokens", None) or [])
    present = {}
    for token in HERMES_SPECIAL_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(token)
        unk = getattr(tokenizer, "unk_token_id", None)
        present[token] = token in vocab_specials or token in added or (tid is not None and tid != unk)
    return present


def resolve_model_id(url: str, fallback: str | None = None) -> str:
    """Return the model id the replica actually serves.

    ``vllm-omni serve`` registers the checkpoint *path* as the model id, not a nickname, and
    rejects an unknown id with ``404 NotFoundError: The model `bagel` does not exist.`` -- a
    routing error that looks nothing like a sampling failure but fails the sweep the same way.
    Ask the server rather than hardcoding a name that only matches some deploys.
    """
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=30.0) as resp:
            data = json.loads(resp.read().decode("utf-8")).get("data") or []
        if data and data[0].get("id"):
            return str(data[0]["id"])
    except (urllib.error.URLError, ValueError, KeyError, IndexError):
        pass
    return fallback or "bagel"


def _message_text(message: dict) -> str:
    """Flatten a chat message to the Hermes text the harness's parser expects.

    BAGEL's AR lane emits the ``<tool_call>`` payload as *content* (its chat template has no
    tool-call parser), but a deploy that does configure ``--tool-call-parser`` returns the same
    call structured under ``tool_calls`` with empty content. Re-render that shape as Hermes text
    so one parser scores both, instead of reporting ``continue`` for an executed call.
    """
    content = message.get("content") or ""
    if content:
        return content
    rendered = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        payload = {"name": fn.get("name"), "arguments": fn.get("arguments")}
        rendered.append(f"<tool_call>\n{json.dumps(payload)}\n</tool_call>")
    return "\n".join(rendered)


def _und_messages(messages_json: str | None, parquet: str | None, row: int, prompt: str) -> list[dict]:
    """Return the ``[{system}, {user}]`` pair the trainer seeds the UND lane with.

    Preferred source is the real training row, so the spike cannot drift from the dataset's
    protocol text. ``messages_json`` (a file path or ``-``) and ``parquet``/``row`` are both
    optional; ``prompt`` alone falls back to the built-in protocol constant.
    """
    if messages_json:
        raw = sys.stdin.read() if messages_json == "-" else open(messages_json, encoding="utf-8").read()
        loaded = json.loads(raw)
        if isinstance(loaded, list):
            return list(loaded)
    if parquet:
        try:
            import pandas as pd  # noqa: PLC0415 -- keep the offline schema check import-free

            messages = list(pd.read_parquet(parquet).iloc[row]["prompt"])
            system = next((m for m in messages if m.get("role") == "system"), None)
            user = next((m for m in messages if m.get("role") == "user"), None)
            if system is not None and user is not None:
                return [{"role": "system", "content": system["content"]}, {"role": "user", "content": prompt}]
        except Exception as exc:  # noqa: BLE001 -- a missing parquet must not fail the gate
            print(f"note: could not read {parquet} row {row} ({exc}); using the built-in protocol", file=sys.stderr)
    return [
        {"role": "system", "content": UND_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def _load_und_tokenizer(model_path: str):
    from transformers import AutoTokenizer  # noqa: PLC0415 -- offline mode must not need it

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def render_und_prompt(
    model_path: str, messages: list[dict], *, with_tools: bool, tokenizer: object | None = None
) -> str:
    """Render ``messages`` to the exact text the trainer feeds the UND lane.

    ``with_tools`` mirrors ``bagel_corl.py``'s ``apply_chat_template(raw_prompt,
    tools=[GENERATE_IMAGE_TOOL_SCHEMA])``, which is the difference the sweep is about: BAGEL's
    template emits the ``<tools>`` block that elicits ``<tool_call>`` *only* when the schema is
    passed.

    ``tokenizer`` is optional so the sweep can load the checkpoint once instead of per arm.
    """
    tokenizer = tokenizer or _load_und_tokenizer(model_path)
    kwargs = {"tools": [GENERATE_IMAGE_TOOL_SCHEMA]} if with_tools else {}
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **kwargs)
    # ``apply_chat_template(tokenize=True)`` returns a ``BatchEncoding``, which is a UserDict --
    # not a ``dict`` subclass -- so attribute access is the portable path here.
    if hasattr(encoded, "input_ids"):
        ids = list(encoded.input_ids)
    elif isinstance(encoded, dict):
        ids = list(encoded["input_ids"])
    else:
        ids = list(encoded)
    text = tokenizer.decode(ids, skip_special_tokens=False)
    # Round-trip guard: everything downstream replays this text, so a lossy render would quietly
    # test a different prompt than the one whose length is printed here.
    assert tokenizer.encode(text, add_special_tokens=False) == ids, "rendered prompt is not id-stable"
    return text


def query_und_replica(
    url: str,
    prompt_text: str,
    timeout_s: float = 120.0,
    *,
    model: str = "bagel",
    temperature: float = 0.7,
    top_p: float = 1.0,
    top_k: int = -1,
    repetition_penalty: float = 1.0,
    max_tokens: int = 256,
) -> str:
    """Decode one UND turn from ``prompt_text``.

    ``prompt_text`` is the **already-rendered** chat prompt (see
    :func:`render_und_prompt`), not a user request, and it is sent as a single ``user`` message.

    That looks wrong and is the only faithful option here: this deploy exposes no chat template
    for the OpenAI route, so passing ``messages`` + ``tools=`` leaves the server unable to render
    either the framing *or* the checkpoint's ``<tools>`` block. The trainer never uses this route
    anyway: it renders client-side and posts raw ``token_ids`` to ``/inference/v1/generate``.
    Pre-rendering and replaying the text is what reproduces those ids here.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return _message_text(body["choices"][0]["message"])


# Configs for ``--sweep``. The agent loop sends temperature / top_p / top_k /
# repetition_penalty **explicitly** on every UND decode, so whatever the deploy config puts
# in ``bagel_corl_deploy_ar.yaml:default_sampling_params`` is overridden for those four
# keys -- the serve defaults never took effect in training. The two configs below are that
# override, before and after the fix:
#
#   as-trained  the rollout defaults the loop used to send (temperature 1.0, top_p 1,
#               top_k -1, repetition_penalty 1.0) -- untruncated, no anti-repetition pressure.
#   mitigated   what the recipe now sends (see ``run_agentic_bagel_rpco_lora.sh``), which
#               matches the deploy config's own intended values.
#
# ``--sweep`` crosses these with whether the ``<tools>`` schema block is in the prompt. Both are
# necessary: sampling alone leaves the model writing prose, schema alone leaves a malformed
# payload. Only ``schema+mitigated`` reaches the GEN lane, which is why only that arm is graded.
#
# ``max_tokens`` is left at the episode response budget for both arms: the loop does not
# send it, and a repetition loop is only observable if it is allowed to run to the cap.
SWEEP_CONFIGS: dict[str, dict] = {
    "as-trained": {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "max_tokens": 1024,
    },
    "mitigated": {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.05,
        "max_tokens": 1024,
    },
}


def degenerate_ratio(text: str) -> float:
    """Fraction of non-empty lines that are the single most repeated line.

    The observed stall is one short string emitted over and over until the budget runs out
    (``'assistant\\nassistant\\nassistant\\n...'``), so "what share of the output is the
    modal line" separates a repetition loop from a real multi-line answer. Returns 0.0 for
    outputs too short to judge.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return 0.0
    modal = max(set(lines), key=lines.count)
    return lines.count(modal) / len(lines)


def _sweep(
    url: str,
    user_prompt: str,
    repeats: int,
    timeout_s: float,
    model: str,
    model_path: str,
    messages: list[dict],
) -> int:
    """A/B the prompt shape against the sampling config; report the outcome mix.

    Two axes:

      schema  whether the checkpoint's ``<tools>`` block is in the prompt.
      sampling  ``as-trained`` (temperature 1.0 / top_p 1 / top_k -1 / repetition_penalty 1.0,
              untruncated) vs ``mitigated`` (what the recipe now sends).

    The 2x2 separates them: if only the ``no-schema`` arms degenerate, the missing ``<tools>``
    block was the cause and the sampling change is prophylaxis, not the fix.
    """
    all_ok = True
    tokenizer = _load_und_tokenizer(model_path)
    prompts = {
        f"schema+{name}": (render_und_prompt(model_path, messages, with_tools=True, tokenizer=tokenizer), cfg)
        for name, cfg in SWEEP_CONFIGS.items()
    }
    prompts.update(
        {
            f"no-schema+{name}": (render_und_prompt(model_path, messages, with_tools=False, tokenizer=tokenizer), cfg)
            for name, cfg in SWEEP_CONFIGS.items()
        }
    )
    print(f"sweep: repeats={repeats} model={model!r} user_prompt={user_prompt!r}")
    for arm, (prompt_text, cfg) in prompts.items():
        prompt_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        kinds: list[str] = []
        ratios: list[float] = []
        for i in range(repeats):
            try:
                text = query_und_replica(url, prompt_text, timeout_s, model=model, **cfg)
            except urllib.error.URLError as exc:
                print(f"  {arm}[{i}] FAIL-CLOSED: request failed: {exc}", file=sys.stderr)
                return 2
            kind = _und_turn_kind(text)
            kinds.append(kind)
            ratios.append(degenerate_ratio(text))
            print(f"  {arm}[{i}] kind={kind} degenerate_ratio={ratios[-1]:.2f} text={text[:160]!r}")
        hits = kinds.count("generate_image")
        loops = sum(1 for r in ratios if r >= 0.5)
        print(
            f"  {arm}: prompt_tokens={prompt_tokens} generate_image={hits}/{repeats} "
            f"repetition_loops={loops}/{repeats} mean_degenerate_ratio={sum(ratios) / max(repeats, 1):.2f}"
        )
        # A run only counts as healthy when the tool call is the common outcome, not a
        # lucky one, and repetition loops are not the dominant failure. Only the arms that
        # match the shipped recipe (schema + mitigated) are graded -- see the 2x2 docstring.
        if arm == "schema+mitigated" and (hits * 2 <= repeats or loops * 2 > repeats):
            all_ok = False
    print("sweep_ok" if all_ok else "sweep_verdict: schema+mitigated did not reach generate_image reliably")
    return 0 if all_ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bagel UND Hermes generate_image spike (no Qwen fallback)")
    parser.add_argument("--model-path", default=os.environ.get("BAGEL_MODEL_PATH"))
    parser.add_argument("--und-url", default=os.environ.get("BAGEL_UND_URL"))
    parser.add_argument(
        "--model",
        default=os.environ.get("BAGEL_UND_MODEL"),
        help="Model id to send to the replica. Defaults to whatever /v1/models advertises.",
    )
    parser.add_argument(
        "--prompt",
        default="Draw a red circle on a white background. Use generate_image if you can render it.",
        help="User request. The system turn comes from --parquet/--messages-json.",
    )
    parser.add_argument(
        "--parquet",
        default=os.environ.get("BAGEL_UND_PARQUET"),
        help=f"Training parquet to take the UND system turn from (default {DEFAULT_UND_PARQUET} when it exists).",
    )
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument(
        "--messages-json",
        default=None,
        help="Explicit [system, user] messages as JSON (a path, or '-' for stdin); wins over --parquet.",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Render without the <tools> schema block (the pre-fix prompt shape).",
    )
    parser.add_argument(
        "--offline-schema",
        action="store_true",
        default=os.environ.get("BAGEL_SPIKE_OFFLINE_SCHEMA") == "1",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="sample the replica N times; a single lucky tool call proves nothing",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="2x2 across the <tools> schema and the as-trained/mitigated sampling in SWEEP_CONFIGS",
    )
    args = parser.parse_args(argv)

    print("tool_schema:", json.dumps(GENERATE_IMAGE_TOOL_SCHEMA))
    print("prompt_token_ids: Bagel UND should prefer skipping decode→re-tokenize when the replica returns ids")

    if args.model_path:
        specials = inspect_tokenizer(args.model_path)
        print("tokenizer_hermes_specials:", json.dumps(specials))
        if not specials.get("<tool_call>") or not specials.get("</tool_call>"):
            print("FAIL-CLOSED: Bagel tokenizer missing Hermes <tool_call> specials", file=sys.stderr)
            return 1

    if args.offline_schema and not args.und_url:
        print("offline schema check only; this is NOT UND serving proof")
        return 0

    if not args.und_url:
        print(
            "FAIL-CLOSED: set BAGEL_UND_URL to a Bagel AR replica. "
            "Do not substitute Qwen3-VL. bagel_single_stage is GEN-only. "
            "Start: bash examples/agenticllmgrpo_trainer/bagel/run_bagel_und_ar_serve.sh",
            file=sys.stderr,
        )
        return 2

    # ``vllm-omni serve`` names the model after the checkpoint path; ask the replica instead of
    # guessing, so an id mismatch cannot masquerade as a sampling verdict.
    model = args.model or resolve_model_id(args.und_url, args.model_path)
    print("model_id:", model)

    parquet = args.parquet
    if parquet is None and os.path.exists(DEFAULT_UND_PARQUET):
        parquet = DEFAULT_UND_PARQUET
    messages = _und_messages(args.messages_json, parquet, args.row, args.prompt)
    print("messages_roles:", [m.get("role") for m in messages])

    if args.sweep:
        return _sweep(args.und_url, args.prompt, max(args.repeat, 1), args.timeout_s, model, args.model_path, messages)

    prompt_text = render_und_prompt(args.model_path, messages, with_tools=not args.no_tools)
    print("render_with_tools:", not args.no_tools)

    try:
        text = query_und_replica(
            args.und_url,
            prompt_text,
            args.timeout_s,
            model=model,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            max_tokens=args.max_tokens,
        )
    except urllib.error.URLError as exc:
        print(f"FAIL-CLOSED: UND replica request failed: {exc}", file=sys.stderr)
        return 2

    print("und_text:", text)
    print(f"degenerate_ratio: {degenerate_ratio(text):.2f}")
    kind = _und_turn_kind(text)
    call = _parse_hermes_tool_call(text)
    if kind != "generate_image" or call is None or call.get("name") != "generate_image":
        print("FAIL-CLOSED: Bagel UND did not emit Hermes <tool_call> generate_image", file=sys.stderr)
        return 1
    print("spike_ok: generate_image from Bagel UND (not Qwen)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
