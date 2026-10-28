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
"""Derive a recipe's `convergence.min_improvement` from its first reviewed baseline.

`min_improvement` ships as `0.0` because no reviewed baseline exists yet, and an
invented floor would be indistinguishable from a real one.  Once a release owner
has produced and reviewed a baseline, this tool turns the measured in-run
improvement of that baseline into a defensible floor:

    proposed_min_improvement = retention * baseline_improvement

with `retention` defaulting to 0.5: a run under test must still deliver at least
half of the learning progress the reviewed baseline delivered.

The tool refuses to invent a floor.  If the baseline did not improve (improvement
<= 0), or the driving metric is missing from the baseline, it exits non-zero and
says so rather than writing a number.

Usage::

    python3 -m tests.convergence.derive_thresholds \
        --recipe tests/convergence/recipes/qwen_image_flowgrpo.yaml \
        --baseline outputs/l4_convergence/baseline/qwen_image_flowgrpo/baseline.json

    # ... and with --write to update the recipe in place.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .recipe_registry import Recipe, RecipeError, load_recipe

DEFAULT_RETENTION = 0.5
EXIT_OK = 0
EXIT_CANNOT_DERIVE = 1
EXIT_USAGE = 2


@dataclass
class ThresholdProposal:
    """The floor proposed for one recipe, with the evidence behind it."""

    case_id: str
    metric: str
    direction: str
    baseline_improvement: float
    current_min_improvement: float
    retention: float
    proposed_min_improvement: float
    baseline_final_window_mean: float | None
    baseline_first_window_mean: float | None
    baseline_source: str

    def as_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "metric": self.metric,
            "direction": self.direction,
            "baseline_improvement": self.baseline_improvement,
            "current_min_improvement": self.current_min_improvement,
            "retention": self.retention,
            "proposed_min_improvement": self.proposed_min_improvement,
            "baseline_final_window_mean": self.baseline_final_window_mean,
            "baseline_first_window_mean": self.baseline_first_window_mean,
            "baseline_source": self.baseline_source,
        }


class ThresholdError(Exception):
    """Raised when a floor cannot be derived from the supplied baseline."""


def pick_driving_metric(recipe: Recipe, requested: str | None) -> dict:
    """Return the metric spec that should drive the floor.

    Defaults to the first required (non-optional) tracked metric, which is the
    signal the recipe is actually gated on.
    """
    tracked = recipe.metrics
    if requested:
        for spec in tracked:
            if str(spec["name"]) == requested:
                return spec
        raise ThresholdError(
            f"metric {requested!r} is not tracked by {recipe.case_id}; tracked metrics are "
            f"{[str(spec['name']) for spec in tracked]}"
        )
    for spec in tracked:
        if not spec.get("optional", False):
            return spec
    raise ThresholdError(f"{recipe.case_id} declares no required tracked metric")


def derive_min_improvement(
    recipe: Recipe,
    baseline: dict,
    *,
    retention: float = DEFAULT_RETENTION,
    metric: str | None = None,
    baseline_source: str = "<memory>",
) -> ThresholdProposal:
    """Propose a `min_improvement` floor from a reviewed baseline.

    Raises:
        ThresholdError: the baseline cannot support a floor.
    """
    if not 0.0 < retention <= 1.0:
        raise ThresholdError(f"retention must be in (0, 1], got {retention}")

    spec = pick_driving_metric(recipe, metric)
    name = str(spec["name"])
    summaries = (baseline.get("summary") or {}) if isinstance(baseline, dict) else {}
    summary = summaries.get(name)
    if not summary:
        raise ThresholdError(
            f"baseline {baseline_source} has no summary for the driving metric {name!r}; it cannot justify a floor"
        )

    improvement = summary.get("improvement")
    if improvement is None:
        raise ThresholdError(
            f"baseline {baseline_source} could not measure an in-run improvement for {name!r}; "
            "a floor cannot be derived from it"
        )
    improvement = float(improvement)
    if improvement <= 0.0:
        raise ThresholdError(
            f"baseline {baseline_source} did not improve {name!r} (improvement={improvement:.6g}); "
            "there is no learning signal to require a fraction of. Review the baseline before "
            "tightening this recipe."
        )

    convergence = recipe.data["convergence"]
    return ThresholdProposal(
        case_id=recipe.case_id,
        metric=name,
        direction=str(spec["direction"]),
        baseline_improvement=improvement,
        current_min_improvement=float(convergence["min_improvement"]),
        retention=retention,
        proposed_min_improvement=max(0.0, retention * improvement),
        baseline_final_window_mean=summary.get("final_window_mean"),
        baseline_first_window_mean=summary.get("first_window_mean"),
        baseline_source=baseline_source,
    )


_MIN_IMPROVEMENT_RE = re.compile(r"^(\s*min_improvement:\s*)(\S+)(\s*)$", re.MULTILINE)


def apply_to_recipe(recipe_path: Path, proposal: ThresholdProposal) -> str:
    """Rewrite one recipe's `convergence.min_improvement` and return the new text.

    Only that single line changes; comments and ordering are preserved so the diff
    a reviewer sees is exactly the threshold change.

    Raises:
        ThresholdError: the recipe does not contain exactly one `min_improvement` key.
    """
    recipe_path = Path(recipe_path)
    text = recipe_path.read_text(encoding="utf-8")
    matches = _MIN_IMPROVEMENT_RE.findall(text)
    if len(matches) != 1:
        raise ThresholdError(f"expected exactly one 'min_improvement:' line in {recipe_path}, found {len(matches)}")
    replaced = _MIN_IMPROVEMENT_RE.sub(
        lambda match: f"{match.group(1)}{proposal.proposed_min_improvement:g}{match.group(3)}",
        text,
        count=1,
    )
    return replaced


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive a recipe's convergence.min_improvement from its first reviewed baseline."
    )
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--metric", default=None, help="driving metric (default: first required metric)")
    parser.add_argument(
        "--retention",
        type=float,
        default=DEFAULT_RETENTION,
        help="fraction of the baseline improvement a run must retain (default: 0.5)",
    )
    parser.add_argument("--write", action="store_true", help="update the recipe file in place")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        recipe = load_recipe(args.recipe)
    except RecipeError as error:
        print(f"[derive] ERROR: {error}", file=sys.stderr)
        return EXIT_USAGE

    if not args.baseline.is_file():
        print(f"[derive] ERROR: baseline does not exist: {args.baseline}", file=sys.stderr)
        return EXIT_USAGE
    try:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        print(f"[derive] ERROR: baseline is not valid JSON: {error}", file=sys.stderr)
        return EXIT_USAGE

    try:
        proposal = derive_min_improvement(
            recipe,
            baseline,
            retention=args.retention,
            metric=args.metric,
            baseline_source=str(args.baseline),
        )
    except ThresholdError as error:
        print(f"[derive] cannot derive a floor: {error}", file=sys.stderr)
        return EXIT_CANNOT_DERIVE

    print(json.dumps(proposal.as_dict(), indent=2, sort_keys=True))
    print(
        f"[derive] {proposal.case_id}: min_improvement "
        f"{proposal.current_min_improvement:g} -> {proposal.proposed_min_improvement:g} "
        f"({proposal.retention:g} x measured baseline improvement "
        f"{proposal.baseline_improvement:.6g} on {proposal.metric})"
    )
    if args.write:
        updated = apply_to_recipe(args.recipe, proposal)
        args.recipe.write_text(updated, encoding="utf-8")
        print(f"[derive] updated {args.recipe}")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
