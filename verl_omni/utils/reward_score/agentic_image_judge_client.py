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
"""HTTP client for the frozen image-judge sidecar (reward C/A fallback).

Primary C/A comes from trajectory ``agentic_judge`` observations. Knobs come
from ``extra_info``; HTTP POST lives in ``verl_omni.utils.agentic.vllm_chat``.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from verl_omni.utils.agentic.vllm_chat import post_vllm_chat
from verl_omni.utils.agentic_image_judge_parse import build_judge_prompt, parse_judge_json

logger = logging.getLogger(__name__)


def _normalize_scored(data: dict, *, backend: str) -> dict | None:
    try:
        correctness = float(data.get("correctness", 0.0))
        aesthetics = float(data.get("aesthetics", 0.0))
    except (TypeError, ValueError):
        return None
    match = float(data.get("match", 0.55 * correctness + 0.45 * aesthetics))
    correctness_scores = data.get("correctness_scores") or {}
    aesthetics_scores = data.get("aesthetics_scores") or {}
    if not isinstance(correctness_scores, dict):
        correctness_scores = {}
    if not isinstance(aesthetics_scores, dict):
        aesthetics_scores = {}
    return {
        "ok": True,
        "correctness": max(0.0, min(1.0, correctness)),
        "aesthetics": max(0.0, min(1.0, aesthetics)),
        "correctness_scores": {
            str(key): max(0.0, min(1.0, float(value)))
            for key, value in correctness_scores.items()
            if isinstance(value, int | float)
        },
        "aesthetics_scores": {
            str(key): max(0.0, min(1.0, float(value)))
            for key, value in aesthetics_scores.items()
            if isinstance(value, int | float)
        },
        "match": max(0.0, min(1.0, match)),
        "good_enough": bool(data.get("good_enough", False)),
        "findings": str(data.get("findings") or ""),
        "suggested_fixes": str(data.get("suggested_fixes") or "none"),
        "backend": str(data.get("backend") or backend),
        "missing_attrs": [],
        "fixes": [],
        "parse_ok": 1,
    }


def _call_vllm_openai(
    *,
    user_request: str,
    image_prompt: str,
    notes: str,
    image_path: str,
    vllm_url: str,
    vllm_model: str = "",
    reflect_max_new_tokens: int = 1024,
    judge_parse_retries: int = 1,
    reflect_vlm_timeout: float = 120.0,
    judge_enable_thinking: bool = False,
    good_enough_threshold: object | None = None,
) -> dict | None:
    try:
        image_b64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    except OSError as exc:
        logger.warning("reflect VLM cannot read image %s: %s", image_path, exc)
        return None

    base_tokens = int(reflect_max_new_tokens)
    max_retries = max(0, int(judge_parse_retries))

    for attempt in range(max_retries + 1):
        strict = attempt > 0
        tokens = base_tokens if attempt == 0 else max(base_tokens, 1536)
        raw_text, err = post_vllm_chat(
            vllm_url=vllm_url,
            image_b64=image_b64,
            prompt_text=build_judge_prompt(user_request, image_prompt, notes, strict_json=strict),
            max_tokens=tokens,
            model=vllm_model,
            timeout=float(reflect_vlm_timeout),
            enable_thinking=bool(judge_enable_thinking),
        )
        if err is not None:
            logger.warning("reflect VLM OpenAI call failed (%s); C/A will be zeroed", err)
            return None
        assert raw_text is not None
        parsed = parse_judge_json(raw_text, good_enough_threshold_value=good_enough_threshold)
        if parsed is not None:
            return _normalize_scored(parsed, backend="vllm")
        logger.warning("reflect VLM OpenAI unparseable (attempt=%d)", attempt)
    return None


def _scorer_knob(extra_info: dict, key: str, default):
    if key not in extra_info or extra_info[key] is None:
        return default
    return extra_info[key]


def _require_scorer_knob(extra_info: dict, key: str):
    if key not in extra_info or extra_info[key] is None:
        raise KeyError(
            f"agentic scorer knob {key!r} missing from extra_info; "
            "driver must call merge_agentic_scorer_knobs (or pass the key explicitly)"
        )
    return extra_info[key]


def call_reflect_vlm(
    *,
    user_request: str,
    image_prompt: str,
    notes: str = "",
    image_path: str | None = None,
    extra_info: dict | None = None,
) -> dict | None:
    """Score a PNG via the frozen VL sidecar (reward fallback).

    Args:
        user_request: Original user task.
        image_prompt: Diffusion prompt for the image.
        notes: Extra judge notes.
        image_path: Path to an existing PNG; missing file returns ``None``.
        extra_info: Scorer knobs. Must include ``good_enough_threshold``.

    Returns:
        Parsed judge dict, or ``None`` on missing image, empty URL, or HTTP/parse failure.

    Raises:
        KeyError: If ``good_enough_threshold`` is missing from ``extra_info``.
    """
    info = dict(extra_info or {})
    # Threshold must be explicit on extra_info (driver merge / compute_score).
    good_enough_threshold = _require_scorer_knob(info, "good_enough_threshold")
    vllm_url = str(_scorer_knob(info, "vllm_url", "") or "").strip()
    if not image_path or not Path(image_path).is_file():
        return None
    if not vllm_url:
        logger.warning(
            "agentic VL fallback skipped: extra_info['vllm_url'] empty "
            "(set agentic_image_gen.vllm_url or rely on trajectory judge markers)"
        )
        return None
    return _call_vllm_openai(
        user_request=user_request,
        image_prompt=image_prompt,
        notes=notes,
        image_path=image_path,
        vllm_url=vllm_url,
        vllm_model=str(_scorer_knob(info, "vllm_model", "") or "").strip(),
        reflect_max_new_tokens=int(_scorer_knob(info, "reflect_max_new_tokens", 1024)),
        judge_parse_retries=int(_scorer_knob(info, "judge_parse_retries", 1)),
        reflect_vlm_timeout=float(_scorer_knob(info, "reflect_vlm_timeout", 120.0)),
        judge_enable_thinking=bool(_scorer_knob(info, "judge_enable_thinking", False)),
        good_enough_threshold=good_enough_threshold,
    )
