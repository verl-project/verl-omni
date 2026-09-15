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
"""Shared agentic image-judge JSON parse + prompt helpers (tool + reward)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Discrete facet grid. Continuous VLM scores are snapped to nearest level
# (ties → lower value) so good_enough flips only across 0.2 boundaries.
_SCORE_GRID: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

_CORRECTNESS_KEYS = (
    "subject_entities",
    "attributes",
    "relations_layout",
    "scene_context",
    "completeness",
)
_AESTHETICS_KEYS = (
    "composition",
    "lighting",
    "color",
    "fidelity",
    "appeal",
)

_AGENTIC_DATA_DIR = Path(__file__).resolve().parent / "agentic"


def _load_judge_questions() -> tuple[dict[str, str], dict[str, str]]:
    payload = json.loads((_AGENTIC_DATA_DIR / "image_judge_questions.json").read_text(encoding="utf-8"))
    correctness = dict(payload["correctness"])
    aesthetics = dict(payload["aesthetics"])
    if tuple(correctness) != _CORRECTNESS_KEYS:
        raise ValueError(f"image_judge_questions.json correctness keys mismatch: {tuple(correctness)}")
    if tuple(aesthetics) != _AESTHETICS_KEYS:
        raise ValueError(f"image_judge_questions.json aesthetics keys mismatch: {tuple(aesthetics)}")
    return correctness, aesthetics


def _load_judge_calibration() -> str:
    payload = json.loads((_AGENTIC_DATA_DIR / "image_judge_calibration.json").read_text(encoding="utf-8"))
    body = str(payload.get("text") or "").strip()
    if not body:
        raise ValueError("image_judge_calibration.json missing non-empty 'text'")
    return body + "\n"


def good_enough_threshold(raw: object | None = None) -> float:
    """Minimum C and A for ``good_enough=YES``.

    Args:
        raw: Explicit threshold. If omitted, reads the bound Hydra knob.

    Returns:
        Float in ``[0, 1]``.

    Raises:
        ValueError: If the value is not a float in ``[0, 1]``.
    """
    if raw is None:
        from verl_omni.tools.trajectory.hydra_env import agentic_get

        raw = agentic_get("good_enough_threshold")
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"agentic_image_gen.good_enough_threshold must be a float in [0, 1], got {raw!r}") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"agentic_image_gen.good_enough_threshold must be in [0, 1], got {value}")
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def snap_score(value: Any, default: float = 0.0) -> float:
    """Snap a continuous score onto the discrete judge grid.

    Args:
        value: Raw score.
        default: Fallback when ``value`` is not numeric.

    Returns:
        Grid value in ``{0.0, 0.2, ..., 1.0}``.
    """
    v = _safe_float(value, default)
    if v >= 1.0:
        return 1.0
    if v >= 0.9:
        return 0.8
    return min(_SCORE_GRID, key=lambda g: (abs(g - v), g))


def _mean_scores(scores: dict[str, float]) -> float:
    if not scores:
        return 0.0
    return sum(scores.values()) / max(1, len(scores))


# Soft symmetric ceiling after a rubber-stamp detection. Keeps C/A usable for
# reward learning while still below a "perfect" 1.0 band.
_RUBBER_STAMP_SCORE_CEILING = 0.8
_RUBBER_STAMP_FINDINGS_NOTE = (
    "[client] rubber-stamp flat high facets: good_enough forced NO; "
    f"C/A facets capped at {_RUBBER_STAMP_SCORE_CEILING:.1f}"
)


def _is_flat_high_facets(scores: dict[str, float], *, min_value: float = 0.9) -> bool:
    """True when ≥2 facets are identical and each is ≥ ``min_value`` (rubber-stamp).

    Default ``min_value=0.9`` so a legitimate discrete-grid ``0.8`` across facets
    can still earn ``good_enough=YES``. Only near-max flat copies are stamped.
    """
    if len(scores) < 2:
        return False
    values = list(scores.values())
    first = values[0]
    if first < min_value:
        return False
    return all(abs(v - first) <= 1e-9 for v in values[1:])


def _cap_facets(scores: dict[str, float], *, ceiling: float) -> dict[str, float]:
    """Clamp each facet to ``ceiling`` (symmetric soft demotion)."""
    return {k: min(v, ceiling) for k, v in scores.items()}


def _annotate_rubber_stamp_findings(findings: str) -> str:
    note = _RUBBER_STAMP_FINDINGS_NOTE
    text = (findings or "").strip()
    if not text:
        return note
    if note in text:
        return text
    return f"{text} {note}"


def normalize_judge_payload(
    data: dict[str, Any],
    *,
    good_enough_threshold_value: object | None = None,
) -> dict[str, Any] | None:
    """Normalize a parsed judge dict into the canonical scored shape.

    Args:
        data: Parsed judge JSON/object.
        backend: Backend label stored on the result.

    Returns:
        Canonical scored dict, or ``None`` if required fields are missing.
    """
    if not isinstance(data, dict):
        return None

    c_scores_raw = data.get("correctness_scores")
    a_scores_raw = data.get("aesthetics_scores")
    c_raw: dict[str, float] = {}
    a_raw: dict[str, float] = {}
    c_scores: dict[str, float] = {}
    a_scores: dict[str, float] = {}
    if isinstance(c_scores_raw, dict) and c_scores_raw:
        for key, value in c_scores_raw.items():
            if isinstance(value, int | float):
                c_raw[str(key)] = _safe_float(value)
                c_scores[str(key)] = snap_score(value)
    if isinstance(a_scores_raw, dict) and a_scores_raw:
        for key, value in a_scores_raw.items():
            if isinstance(value, int | float):
                a_raw[str(key)] = _safe_float(value)
                a_scores[str(key)] = snap_score(value)

    rubber_stamp = False
    if c_scores and a_scores:
        # Detect on raw continuous facets (before snap) to avoid double penalty.
        rubber_stamp = _is_flat_high_facets(c_raw) or _is_flat_high_facets(a_raw)
        if rubber_stamp:
            c_scores = _cap_facets(c_scores, ceiling=_RUBBER_STAMP_SCORE_CEILING)
            a_scores = _cap_facets(a_scores, ceiling=_RUBBER_STAMP_SCORE_CEILING)
        correctness = _mean_scores(c_scores)
        aesthetics = _mean_scores(a_scores)
    elif "correctness" in data or "aesthetics" in data:
        correctness = snap_score(data.get("correctness", 0.0))
        aesthetics = snap_score(data.get("aesthetics", 0.0))
        # Scalar-only: treat both axes ≥ 0.9 as an undifferentiated stamp.
        rubber_stamp = (
            _safe_float(data.get("correctness", 0.0)) >= 0.9 and _safe_float(data.get("aesthetics", 0.0)) >= 0.9
        )
        if rubber_stamp:
            correctness = min(correctness, _RUBBER_STAMP_SCORE_CEILING)
            aesthetics = min(aesthetics, _RUBBER_STAMP_SCORE_CEILING)
    else:
        return None

    thr = good_enough_threshold(good_enough_threshold_value)
    # Always derive YES/NO from scores × Hydra threshold. Ignore any model-emitted
    # ``good_enough`` flag so ``agentic_image_gen.good_enough_threshold`` controls
    # rewrite pressure. Rubber-stamps never count as YES.
    good_enough = (not rubber_stamp) and correctness >= thr and aesthetics >= thr

    findings = str(data.get("findings") or "")
    if rubber_stamp:
        findings = _annotate_rubber_stamp_findings(findings)

    return {
        "correctness": correctness,
        "aesthetics": aesthetics,
        "correctness_scores": c_scores,
        "aesthetics_scores": a_scores,
        "findings": findings,
        "suggested_fixes": str(data.get("suggested_fixes") or ""),
        "good_enough": good_enough,
        "rubber_stamp": rubber_stamp,
    }


def parse_judge_json(text: str, *, good_enough_threshold_value: object | None = None) -> dict[str, Any] | None:
    """Extract C/A judge scores from VLM text.

    Args:
        text: Raw VLM response (may include fences / think blocks).
        backend: Backend label for the normalized payload.

    Returns:
        Canonical scored dict, or ``None`` on parse failure.
    """
    blob = (text or "").strip()
    blob = re.sub(r"<think>[\s\S]*?</think>", " ", blob, flags=re.IGNORECASE)
    blob = re.sub(r"```(?:json)?\s*", "", blob, flags=re.IGNORECASE).replace("```", "")
    blob = blob.strip()

    decoder = json.JSONDecoder()
    best: tuple[int, int, dict[str, Any]] | None = None
    for index, char in enumerate(blob):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(blob[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        normalized = normalize_judge_payload(data, good_enough_threshold_value=good_enough_threshold_value)
        if normalized is None:
            continue
        # Prefer full facet dicts over scalar-only parses.
        score = 2 if normalized["correctness_scores"] and normalized["aesthetics_scores"] else 1
        cand = (score, -index, normalized)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is not None:
        return best[2]

    # Last resort: aggregate scalars if JSON was truncated mid-object.
    c_m = re.search(r'"?correctness"?\s*[:=]\s*([0-9]*\.?[0-9]+)', blob, re.IGNORECASE)
    a_m = re.search(r'"?aesthetics"?\s*[:=]\s*([0-9]*\.?[0-9]+)', blob, re.IGNORECASE)
    if c_m and a_m:
        return normalize_judge_payload(
            {
                "correctness": float(c_m.group(1)),
                "aesthetics": float(a_m.group(1)),
                "findings": "parsed from truncated VLM text",
                "suggested_fixes": "none",
            },
            good_enough_threshold_value=good_enough_threshold_value,
        )
    return None


def build_judge_prompt(user_request: str, image_prompt: str, notes: str = "", *, strict_json: bool = False) -> str:
    """Build the VL judge prompt.

    Args:
        user_request: Original user task.
        image_prompt: Diffusion prompt for the image.
        notes: Optional extra notes.
        strict_json: If True, use the parse-failure retry prompt.

    Returns:
        Prompt string.
    """
    c_schema = ",\n".join(f'    "{key}": 0.0' for key in _CORRECTNESS_KEYS)
    a_schema = ",\n".join(f'    "{key}": 0.0' for key in _AESTHETICS_KEYS)
    rubric = "\n".join(
        [
            "CORRECTNESS QUESTIONS:",
            *[f"- {key}: {q}" for key, q in _load_judge_questions()[0].items()],
            "AESTHETICS QUESTIONS:",
            *[f"- {key}: {q}" for key, q in _load_judge_questions()[1].items()],
        ]
    )
    header = (
        "You are a strict, calibrated visual reward judge. Inspect the pixels, not merely the "
        "diffusion prompt. Independently answer all ten rubric questions.\n"
        f"User request: {user_request}\n"
        f"Diffusion prompt used (context only; never treat it as visual evidence): {image_prompt or '(none)'}\n"
        f"Notes: {notes or '(none)'}\n\n"
        f"{rubric}\n\n"
    )
    calibration = _load_judge_calibration()
    if strict_json:
        return (
            header + "CRITICAL RETRY: Your previous reply was not valid JSON.\n"
            "Reply with ONE JSON object only. No markdown, no <think>, no prose before/after.\n"
            "Each facet score MUST be exactly one of {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}.\n"
            "Use this exact shape:\n"
            "{\n"
            '  "correctness_scores": {\n'
            f"{c_schema}\n"
            "  },\n"
            '  "aesthetics_scores": {\n'
            f"{a_schema}\n"
            "  },\n"
            '  "findings": "short pixel evidence",\n'
            '  "suggested_fixes": "short rewrite hints"\n'
            "}\n"
        )
    return (
        header + calibration + "Return ONLY this JSON shape (replace every 0.0 with a grid score):\n"
        "{\n"
        '  "correctness_scores": {\n'
        f"{c_schema}\n"
        "  },\n"
        '  "aesthetics_scores": {\n'
        f"{a_schema}\n"
        "  },\n"
        '  "findings": "specific visual evidence for the lowest scores (and for any ≥0.8)",\n'
        '  "suggested_fixes": "specific prompt rewrite hints"\n'
        "}\n"
    )


def format_judge_observation(
    *,
    image_path: str,
    parsed: dict[str, Any],
    backend: str,
    parse_retries: int = 0,
) -> tuple[str, dict[str, Any]]:
    """Format a successful judge observation for the trajectory.

    Args:
        scored: Canonical scored judge dict.
        image_path: Path recorded in the observation.

    Returns:
        Tool observation text with ``agentic_judge ok=1``.
    """
    correctness = float(parsed["correctness"])
    aesthetics = float(parsed["aesthetics"])
    good = bool(parsed.get("good_enough", False))
    findings_short = re.sub(r"\s+", " ", str(parsed.get("findings") or "no specific findings")).strip()[:220]
    fixes_short = re.sub(r"\s+", " ", str(parsed.get("suggested_fixes") or "none")).strip()[:160]
    text = (
        f"VL judge on the last generated image:\n"
        f"  path={image_path}\n"
        f"  correctness={correctness:.2f}\n"
        f"  aesthetics ={aesthetics:.2f}\n"
        f"  good_enough ={'YES' if good else 'NO'}\n"
        f"  findings: {findings_short}\n"
        f"  suggested_fixes: {fixes_short}\n"
        f"  agentic_judge ok=1 parse_ok=1 stub=0 backend={backend} parse_retries={parse_retries}"
    )
    meta = {
        "correctness": correctness,
        "aesthetics": aesthetics,
        "good_enough": good,
        "findings": str(parsed.get("findings") or ""),
        "suggested_fixes": str(parsed.get("suggested_fixes") or "none"),
        "image_path": image_path,
        "backend": backend,
        "parse_ok": 1,
        "parse_retries": int(parse_retries),
    }
    if "rubber_stamp" in parsed:
        meta["rubber_stamp"] = bool(parsed.get("rubber_stamp"))
    for key, value in (parsed.get("correctness_scores") or {}).items():
        if isinstance(value, int | float):
            meta[f"correctness_{key}"] = float(value)
    for key, value in (parsed.get("aesthetics_scores") or {}).items():
        if isinstance(value, int | float):
            meta[f"aesthetics_{key}"] = float(value)
    return text, meta


def format_judge_parse_error(
    *,
    image_path: str,
    raw_text: str = "",
    backend: str = "vllm",
    parse_retries: int = 0,
) -> tuple[str, dict[str, Any]]:
    """Format a failed judge observation (no fake C/A).

    Args:
        image_path: Path recorded in the observation.
        raw_text: Raw VLM text that failed to parse.
        backend: Backend label.
        parse_retries: Number of parse retries already attempted.

    Returns:
        Tool observation text with ``agentic_judge ok=0 parse_ok=0``.
    """
    text = (
        "[judge error] VLM returned unparseable response — do not invent scores. "
        "Retry judge_image or rewrite the diffusion prompt and generate again.\n"
        f"  path={image_path}\n"
        f"  agentic_judge ok=0 parse_ok=0 stub=0 backend={backend} parse_retries={parse_retries}"
    )
    meta = {
        "error": "unparseable",
        "raw": (raw_text or "")[:300],
        "image_path": image_path,
        "backend": backend,
        "parse_ok": 0,
        "parse_retries": int(parse_retries),
    }
    return text, meta
