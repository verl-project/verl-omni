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
import os
import re
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

_CORRECTNESS_QUESTIONS = {
    "subject_entities": "Are the requested primary subjects/entities visibly present and recognizable? If the "
    "subject is a person, is their gender, age, and ethnicity correct? Is he/she facially recognizable?",
    "attributes": "Are non-text attributes (color, count, material, identity) correct? When the request "
    "specifies any text/typography (headlines, slogans, labels, numbers, logo lettering), OCR the "
    "pixels: are the exact requested strings fully legible and spelled correctly — no gibberish, "
    "substitutions, missing/extra words, or wrong script? Scene beauty alone does not pass this facet.",
    "relations_layout": "Are requested actions, spatial relations, and layout/composition constraints correct "
    "(including poster hierarchy when requested: e.g. headline at top, main subject center, "
    "footer/tagline at bottom)?",
    "scene_context": "Does the environment, setting, style, and overall scene match the request?",
    "completeness": "Is the request fully satisfied without missing requested details or contradictory extras? "
    "If text was requested, treat missing, truncated, illegible, or wrong strings as incompleteness "
    "even when the rest of the scene looks right.",
}
_AESTHETICS_QUESTIONS = {
    "composition": "Is the composition balanced with a clear focal hierarchy and intentional framing?",
    "lighting": "Are lighting, exposure, contrast, and depth visually effective?",
    "color": "Are color harmony, saturation, and tonal relationships pleasing and coherent?",
    "fidelity": "Is the image sharp and spatially coherent, without obvious generation artifacts or distortions?",
    "appeal": "Does the image have strong overall visual appeal and professional finish?",
}


def good_enough_threshold() -> float:
    """Min C and A for good_enough=YES (client-side vLLM judge path).

    Default 0.80: both axes must reach the 0.8 grid band (aligned with discrete
    facet scores). Override with ``AGENTIC_JUDGE_GOOD_ENOUGH_THRESHOLD`` or
    ``AGENTIC_REFLECT_GOOD_ENOUGH``.
    """
    raw = os.getenv("AGENTIC_JUDGE_GOOD_ENOUGH_THRESHOLD") or os.getenv("AGENTIC_REFLECT_GOOD_ENOUGH") or "0.80"
    try:
        return float(raw)
    except ValueError:
        return 0.80


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def snap_score(value: Any, default: float = 0.0) -> float:
    """Snap a score to ``_SCORE_GRID``.

    Values in ``[0.9, 1.0)`` map to ``0.8`` so mid-high continuous scores cannot
    become max. Only an exact ``1.0`` stays ``1.0``. Other values use nearest
    grid point (ties → lower).
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


def normalize_judge_payload(data: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a parsed judge dict into the canonical scored shape.

    Facet scores are snapped onto ``_SCORE_GRID``. Rubber-stamps (flat identical
    near-max facets on the *raw* pre-snap values, default ≥0.9) are handled
    model-agnostically:

    - ``good_enough`` is forced ``False`` (blocks Done on undifferentiated maxing),
    - C and A facets share a soft symmetric cap at 0.8 (keeps reward signal),
    - ``findings`` gains an explicit ``[client]`` note so obs match the scores.

    A uniform discrete ``0.8`` grid is *not* a stamp — that path must remain able
    to return YES so the agent can learn NO→rewrite→YES without max-pass stops.
    Detection uses raw facets so ``snap_score``'s ``[0.9, 1.0)→0.8`` path does
    not itself create a flat group that then gets double-penalized. Model-emitted
    ``good_enough`` flags are ignored; YES requires snapped C/A ≥ env threshold
    and no rubber-stamp.
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

    thr = good_enough_threshold()
    # Always derive YES/NO from scores × env threshold. Ignore any model-emitted
    # ``good_enough`` flag so AGENTIC_JUDGE_GOOD_ENOUGH_THRESHOLD actually controls
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


def parse_judge_json(text: str) -> dict[str, Any] | None:
    """Extract C/A judge scores from VLM text (think blocks / fences / truncation)."""
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
        normalized = normalize_judge_payload(data)
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
            }
        )
    return None


def build_judge_prompt(user_request: str, image_prompt: str, notes: str = "", *, strict_json: bool = False) -> str:
    """Build the VL judge prompt. ``strict_json`` is used on parse-failure retry."""
    c_schema = ",\n".join(f'    "{key}": 0.0' for key in _CORRECTNESS_KEYS)
    a_schema = ",\n".join(f'    "{key}": 0.0' for key in _AESTHETICS_KEYS)
    rubric = "\n".join(
        [
            "CORRECTNESS QUESTIONS:",
            *[f"- {key}: {q}" for key, q in _CORRECTNESS_QUESTIONS.items()],
            "AESTHETICS QUESTIONS:",
            *[f"- {key}: {q}" for key, q in _AESTHETICS_QUESTIONS.items()],
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
    calibration = (
        "SCORE GRID (REQUIRED — each facet MUST be exactly one of these):\n"
        "  {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}\n"
        "Do NOT emit continuous mid values (e.g. 0.55, 0.84, 0.95). Pick the closest grid level.\n\n"
        "CORRECTNESS LEVEL ANCHORS (apply per facet independently):\n"
        "  0.0 = requested content absent, unrecognizable, or completely wrong\n"
        "  0.2 = severely deficient; majority missing or wrong\n"
        "  0.4 = partial; several elements present but many wrong/blurry/misplaced\n"
        "  0.6 = subjects/entities mostly present BUT attributes (incl. OCR text), relations, or "
        "details weak (presence alone → max 0.6 for subject_entities / completeness)\n"
        "  0.8 = key attributes and relations clearly correct; only minor missing detail\n"
        "  1.0 = no missing requested detail and no contradictory extras (RARE)\n"
        "TEXT/OCR RULE (when the user request quotes or requires specific wording):\n"
        "  - Illegible, gibberish, or wrong strings → attributes ≤ 0.4 and completeness ≤ 0.6 "
        "(do NOT inflate from cozy lighting / nice cup / matching style).\n"
        "  - findings MUST quote the glyphs actually visible vs the requested strings.\n"
        "  - suggested_fixes MUST give concrete typography/legibility rewrite hints "
        "(e.g. exact quoted text, 'highly legible poster typography', placement).\n\n"
        "AESTHETICS LEVEL ANCHORS (apply per facet independently):\n"
        "  0.0 = unusable (severe artifacts, collapse, or illegible)\n"
        "  0.2 = major artifacts / broken anatomy / harsh clutter\n"
        "  0.4 = typical rough sketch / flat fantasy render — acceptable but weak finish\n"
        "  0.6 = readable composition and color, still soft lighting or mild artifacts\n"
        "  0.8 = clear composition + effective lighting + low artifact (reserve for this)\n"
        "  1.0 = near-professional finish (VERY RARE for diffusion sketches)\n"
        "Default typical Qwen-Image / fantasy sketch renders to 0.4–0.6 on aesthetics facets.\n\n"
        "INDEPENDENCE RULES:\n"
        "- Score each facet from its own pixel evidence; do NOT copy the same number across "
        "all five correctness facets unless each facet independently earns it.\n"
        "- For any facet ≥ 0.8, findings MUST include one short sentence of concrete pixel evidence.\n"
        "- Prefer under-scoring over generosity; mid-high saturation (all ~0.9) is a failure mode.\n"
    )
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
    """Format a successful judge obs (``agentic_judge ok=1 parse_ok=1``)."""
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
    """Format a failed judge obs (``agentic_judge ok=0 parse_ok=0``) — no fake C/A."""
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
