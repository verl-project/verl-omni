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
"""Score the paired per-prompt validation metric an L4 recipe declares primary.

An L4 recipe's ``protocol.json`` can declare a primary metric that is not a
per-step curve:

    paired per-prompt score difference step48 - step0; report all values and
    prompt-bootstrap 95% CI; step24 diagnostic only

That metric lives in the validation dumps the trainer already writes
(``validation/<step>.jsonl``), one row per fixed prompt.  It is worth scoring
separately from the tracked curves because it is a much better instrument: a
fixed prompt set with a greedy reward carries far less noise than a reward
averaged over a handful of sampled images per training step, so it can resolve
an effect the per-step curve cannot.

The statistic is deliberately narrow.  Prompts are paired by row index (the
protocol fixes the prompt set and disables shuffling), the interval comes from
resampling *prompts*, and a result is only reported as a signal when that
interval excludes zero.  A mean with an interval that spans zero is reported as
``inconclusive`` rather than as a small win, because at these prompt counts the
two are indistinguishable.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

DEFAULT_FIELD = "score"
DEFAULT_RESAMPLES = 20000
DEFAULT_SEED = 12345
DEFAULT_CONFIDENCE = 0.95

VERDICT_POSITIVE = "signal_positive"
VERDICT_NEGATIVE = "signal_negative"
VERDICT_INCONCLUSIVE = "inconclusive"


class PairedValidationError(Exception):
    """Raised when a validation pair cannot be scored."""


@dataclass(frozen=True)
class PairedResult:
    """Outcome of scoring one ``before``/``after`` validation pair."""

    field: str
    samples: int
    before_mean: float
    after_mean: float
    mean_difference: float
    ci_low: float
    ci_high: float
    confidence: float
    resamples: int
    seed: int
    improved: int
    unchanged: int
    worse: int

    @property
    def verdict(self) -> str:
        """``signal_positive``, ``signal_negative`` or ``inconclusive``."""
        if self.ci_low > 0.0:
            return VERDICT_POSITIVE
        if self.ci_high < 0.0:
            return VERDICT_NEGATIVE
        return VERDICT_INCONCLUSIVE

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable view of the result."""
        return {
            "field": self.field,
            "samples": self.samples,
            "before_mean": self.before_mean,
            "after_mean": self.after_mean,
            "mean_difference": self.mean_difference,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "confidence": self.confidence,
            "resamples": self.resamples,
            "seed": self.seed,
            "improved": self.improved,
            "unchanged": self.unchanged,
            "worse": self.worse,
            "verdict": self.verdict,
        }


def load_scores(path: Path | str, field: str = DEFAULT_FIELD) -> list[float]:
    """Read one score per row from a validation dump.

    Rows keep their file order: the protocol pairs by prompt, so reordering
    would silently compare different prompts.
    """
    target = Path(path)
    if not target.exists():
        raise PairedValidationError(f"validation dump not found: {target}")
    scores: list[float] = []
    for lineno, line in enumerate(target.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise PairedValidationError(f"{target}:{lineno} is not valid JSON: {error}") from error
        if field not in row:
            raise PairedValidationError(f"{target}:{lineno} has no {field!r} field")
        value = row[field]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise PairedValidationError(f"{target}:{lineno} has a non-numeric {field!r}: {value!r}")
        scores.append(float(value))
    if not scores:
        raise PairedValidationError(f"{target} contains no rows")
    return scores


def paired_differences(before: list[float], after: list[float]) -> list[float]:
    """Return ``after - before`` per prompt.

    A length mismatch means the two dumps do not cover the same fixed prompt
    set, which makes the pairing meaningless, so it is an error rather than a
    truncation.
    """
    if len(before) != len(after):
        raise PairedValidationError(
            f"prompt sets differ: before has {len(before)} rows, after has {len(after)}"
        )
    return [b - a for a, b in zip(before, after)]


def bootstrap_ci(
    differences: list[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> tuple[float, float]:
    """Resample prompts with replacement and return the two-sided interval.

    Deterministic for a given ``seed`` so a reported interval can be
    reproduced exactly.
    """
    if not differences:
        raise PairedValidationError("cannot bootstrap an empty sample")
    if resamples <= 0:
        raise PairedValidationError(f"resamples must be positive, got {resamples}")
    if not 0.0 < confidence < 1.0:
        raise PairedValidationError(f"confidence must be in (0, 1), got {confidence}")

    count = len(differences)
    rng = random.Random(seed)
    means = [
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    ]
    means.sort()
    tail = (1.0 - confidence) / 2.0
    low_index = int(tail * resamples)
    high_index = min(resamples - 1, int((1.0 - tail) * resamples))
    return means[low_index], means[high_index]


def score_pair(
    before_path: Path | str,
    after_path: Path | str,
    *,
    field: str = DEFAULT_FIELD,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> PairedResult:
    """Score one ``before``/``after`` validation pair end to end."""
    before = load_scores(before_path, field=field)
    after = load_scores(after_path, field=field)
    differences = paired_differences(before, after)
    ci_low, ci_high = bootstrap_ci(
        differences, resamples=resamples, seed=seed, confidence=confidence
    )
    return PairedResult(
        field=field,
        samples=len(differences),
        before_mean=sum(before) / len(before),
        after_mean=sum(after) / len(after),
        mean_difference=sum(differences) / len(differences),
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
        improved=sum(1 for value in differences if value > 0.0),
        unchanged=sum(1 for value in differences if value == 0.0),
        worse=sum(1 for value in differences if value < 0.0),
    )


def render(result: PairedResult) -> str:
    """Render a result as the short report the protocol asks for."""
    lines = [
        f"field            : {result.field}",
        f"samples          : {result.samples}",
        f"before mean      : {result.before_mean:.6f}",
        f"after mean       : {result.after_mean:.6f}",
        f"paired mean diff : {result.mean_difference:+.6f}",
        (
            f"{result.confidence:.0%} CI          : "
            f"[{result.ci_low:+.6f}, {result.ci_high:+.6f}]"
            f"  ({result.resamples} resamples, seed {result.seed})"
        ),
        f"improved/unchanged/worse : {result.improved}/{result.unchanged}/{result.worse}",
        f"verdict          : {result.verdict}",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Score the paired per-prompt validation metric of an L4 recipe. "
            "This is evidence, not a gate: it reports an interval and refuses to "
            "call a difference a signal when that interval spans zero."
        )
    )
    parser.add_argument("--before", required=True, help="validation/<step>.jsonl for the earlier step")
    parser.add_argument("--after", required=True, help="validation/<step>.jsonl for the later step")
    parser.add_argument("--field", default=DEFAULT_FIELD, help=f"score field (default: {DEFAULT_FIELD})")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="bootstrap seed")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE, help="interval confidence")
    parser.add_argument("--json", action="store_true", help="emit the result as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 for a signal, 1 for inconclusive, 2 on error."""
    args = parse_args(argv)
    try:
        result = score_pair(
            args.before,
            args.after,
            field=args.field,
            resamples=args.resamples,
            seed=args.seed,
            confidence=args.confidence,
        )
    except PairedValidationError as error:
        print(f"error: {error}")
        return 2
    print(json.dumps(result.as_dict(), indent=2) if args.json else render(result))
    return 1 if result.verdict == VERDICT_INCONCLUSIVE else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
