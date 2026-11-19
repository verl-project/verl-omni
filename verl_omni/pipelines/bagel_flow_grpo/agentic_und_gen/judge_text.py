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

"""Reduce one GEN call's RM verdicts into the Mode-2a judge observation the UND lane reads."""

from __future__ import annotations

import numpy as np

from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.samples import GenSample
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.turn_protocol import (
    derive_good_enough_from_scores,
    format_rm_scores_as_judge_text,
)

_UNIFIED_AXES = ("alignment", "coherence", "style")
_AXIS_FINDINGS = {
    "alignment": "content vs the prompt",
    "coherence": "glitches / distorted objects",
    "style": "aesthetic appeal",
}
_AXIS_FIXES = {
    "alignment": "state every requested subject, attribute, count and relation explicitly in the diffusion prompt",
    "coherence": "simplify the composition and describe shapes, poses and spatial layout concretely",
    "style": "specify the medium, lighting, palette and camera framing",
}


def _episode_good_enough(explicit: list[bool], *, reduction: str, threshold: float | None) -> bool:
    """Episode-level stop bit over the per-seed explicit flags (``any`` / ``all`` / ``mean``)."""
    if reduction == "all":
        return all(explicit)
    if reduction == "mean":
        effective = float(threshold) if threshold is not None else 0.7
        return (sum(1.0 for flag in explicit if flag) / len(explicit)) >= effective
    return any(explicit)


def _mean_rm_axes(samples: list[GenSample]) -> dict[str, float] | None:
    """Average the UnifiedReward axes over the call's valid seeds, or ``None`` if absent."""
    rows = [s.rm_axes for s in samples if s.valid and s.rm_axes and set(_UNIFIED_AXES) <= set(s.rm_axes)]
    if not rows:
        return None
    return {axis: float(sum(float(r[axis]) for r in rows) / len(rows)) for axis in _UNIFIED_AXES}


def _judge_facets_from_axes(axes: dict[str, float]) -> tuple[float, float, str, str]:
    """Map 1-5 UnifiedReward axes onto the Mode-2a judge facets (both on [0, 1]) plus findings/fix text."""

    def unit(value: float) -> float:
        return (min(max(float(value), 1.0), 5.0) - 1.0) / 4.0

    correctness = unit(axes["alignment"])
    aesthetics = (unit(axes["coherence"]) + unit(axes["style"])) / 2.0
    findings = "; ".join(f"{axis}={axes[axis]:.1f}/5 ({_AXIS_FINDINGS[axis]})" for axis in _UNIFIED_AXES)
    weakest = min(_UNIFIED_AXES, key=lambda axis: axes[axis])
    fixes = f"{weakest} is weakest: {_AXIS_FIXES[weakest]}" if axes[weakest] < 5.0 else ""
    return correctness, aesthetics, f"UnifiedReward axes {findings}", fixes


def judge_text_from_gen_samples(
    samples: list[GenSample], *, reduction: str = "any", threshold: float | None = None
) -> str | None:
    """Reduce RM scores into Mode-2a judge text for ``build_forced_reflection``.

    ``reduction`` is the episode-level stop bit over per-seed explicit flags; ``threshold`` is
    ``agentic_image_gen.good_enough_threshold`` (``None`` keeps the turn-protocol default).
    Samples without an explicit flag receive the derived one.
    """
    scores = [float(s.rm_score) for s in samples if s.valid and s.rm_score is not None]
    if not scores:
        return None
    mean = float(sum(scores) / len(scores))
    explicit = [bool(s.good_enough) for s in samples if s.good_enough is not None]
    if explicit:
        good_enough = _episode_good_enough(explicit, reduction=reduction, threshold=threshold)
    else:
        good_enough = derive_good_enough_from_scores(
            correctness=mean,
            aesthetics=mean,
            similarity=None,
            **({"threshold": float(threshold)} if threshold is not None else {}),
        )
    for sample in samples:
        if sample.good_enough is None:
            sample.good_enough = good_enough
    correctness, aesthetics = mean, mean
    findings, fixes = "", ""
    rubber_stamp = False
    judges = [s.rm_judge for s in samples if s.valid and s.rm_judge]
    axes = _mean_rm_axes(samples)
    if judges:
        correctness = float(np.mean([float(j.get("correctness", 0.0)) for j in judges]))
        aesthetics = float(np.mean([float(j.get("aesthetics", 0.0)) for j in judges]))
        worst = min(judges, key=lambda j: float(j.get("correctness", 0.0)) + float(j.get("aesthetics", 0.0)))
        findings = str(worst.get("findings") or "")
        fixes = str(worst.get("suggested_fixes") or "")
        rubber_stamp = any(bool(j.get("rubber_stamp")) for j in judges)
    elif axes is not None:
        correctness, aesthetics, findings, fixes = _judge_facets_from_axes(axes)
    return format_rm_scores_as_judge_text(
        correctness=correctness,
        aesthetics=aesthetics,
        good_enough=good_enough,
        findings=findings,
        suggested_fixes=fixes,
        rubber_stamp=rubber_stamp,
        similarity=None,
    )
