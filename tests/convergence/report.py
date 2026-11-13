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
"""Aggregate per-case L4 results into a release-readiness report.

The report is the only artifact a release manager is expected to read, so it is
deliberately conservative:

* ``ready`` requires every gated case to have ``status == "passed"`` *and* an
  evidence level of at least ``compared``; a case that only parsed a recipe
  cannot make a release ready;
* ``blocked`` means a gated case actually ran and failed;
* ``incomplete`` means at least one gated case is missing, skipped, timed out,
  invalid, or incomparable — i.e. the evidence does not exist.

There is no path that turns absent evidence into ``ready``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

#: Ordered from strongest to weakest evidence.
EVIDENCE_LEVELS = ("none", "static", "run", "compared")

CASE_STATUSES = (
    "passed",
    "failed",
    "incomparable",
    "timeout",
    "skipped",
    "invalid",
    "not_run",
    "baseline_created",
)

#: Statuses that make the overall release verdict ``blocked``.
BLOCKING_STATUSES = ("failed",)

#: Statuses that make the overall release verdict ``incomplete``.
INCOMPLETE_STATUSES = ("incomparable", "timeout", "skipped", "invalid", "not_run", "baseline_created")


class ReportError(Exception):
    """Raised when a per-case result cannot be read or is not a valid L4 result."""


@dataclass
class CaseResult:
    """The normalized part of one case result that the report consumes."""

    case_id: str
    status: str
    evidence_level: str
    converged: bool
    release_gate: bool = True
    title: str = ""
    duration_s: float | None = None
    completed_steps: int | None = None
    expected_steps: int | None = None
    failure_reason: str | None = None
    failed_metrics: list[str] = field(default_factory=list)
    log_path: str | None = None
    result_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "title": self.title,
            "status": self.status,
            "evidence_level": self.evidence_level,
            "converged": self.converged,
            "release_gate": self.release_gate,
            "duration_s": self.duration_s,
            "completed_steps": self.completed_steps,
            "expected_steps": self.expected_steps,
            "failure_reason": self.failure_reason,
            "failed_metrics": list(self.failed_metrics),
            "log_path": self.log_path,
            "result_path": self.result_path,
        }


def case_result_from_payload(payload: dict[str, Any], *, result_path: Path | None = None) -> CaseResult:
    """Normalize a ``result.json`` payload, rejecting records that are not L4 results."""
    if not isinstance(payload, dict):
        raise ReportError(f"case result at {result_path} is not an object")
    if payload.get("layer") != "L4":
        raise ReportError(f"case result at {result_path} has layer {payload.get('layer')!r}, expected 'L4'")
    status = payload.get("status")
    if status not in CASE_STATUSES:
        raise ReportError(f"case result at {result_path} has unknown status {status!r}")
    evidence = payload.get("evidence_level", "none")
    if evidence not in EVIDENCE_LEVELS:
        raise ReportError(f"case result at {result_path} has unknown evidence level {evidence!r}")
    case_id = payload.get("case_id")
    if not isinstance(case_id, str) or not case_id:
        raise ReportError(f"case result at {result_path} has no case_id")

    run = payload.get("run") or {}
    return CaseResult(
        case_id=case_id,
        title=str(payload.get("title", "")),
        status=str(status),
        evidence_level=str(evidence),
        converged=bool(payload.get("converged", False)),
        release_gate=bool(payload.get("release_gate", True)),
        duration_s=payload.get("duration_s"),
        completed_steps=run.get("completed_steps"),
        expected_steps=run.get("expected_steps"),
        failure_reason=payload.get("failure_reason"),
        failed_metrics=list((payload.get("comparison") or {}).get("failed_metrics", [])),
        log_path=run.get("log_path"),
        result_path=str(result_path) if result_path else None,
    )


def load_case_results(results_dir: Path) -> list[CaseResult]:
    """Load every ``*/result.json`` under a results directory.

    Raises:
        ReportError: the directory is missing or a result file is unreadable.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise ReportError(f"results directory does not exist: {results_dir}")
    results: list[CaseResult] = []
    for path in sorted(results_dir.glob("*/result.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ReportError(f"case result at {path} is not valid JSON: {error}") from error
        results.append(case_result_from_payload(payload, result_path=path))
    if not results:
        raise ReportError(f"no case results found under {results_dir}")
    return results


def _evidence_rank(level: str) -> int:
    return EVIDENCE_LEVELS.index(level) if level in EVIDENCE_LEVELS else -1


def summarize_cases(cases: list[CaseResult]) -> dict[str, Any]:
    """Compute the overall release verdict and the status histogram."""
    gated = [case for case in cases if case.release_gate]
    ungated = [case for case in cases if not case.release_gate]
    counts: dict[str, int] = {status: 0 for status in CASE_STATUSES}
    for case in cases:
        counts[case.status] = counts.get(case.status, 0) + 1

    if not gated:
        overall = "incomplete"
        rationales = ["no gated L4 case was reported; nothing was verified for release"]
    else:
        blocking = [case.case_id for case in gated if case.status in BLOCKING_STATUSES]
        incomplete = [
            case.case_id
            for case in gated
            if case.status in INCOMPLETE_STATUSES or _evidence_rank(case.evidence_level) < _evidence_rank("compared")
        ]
        not_passed = [case.case_id for case in gated if case.status != "passed"]
        if blocking:
            overall = "blocked"
            rationales = [f"gated case(s) failed: {', '.join(sorted(set(blocking)))}"]
        elif incomplete or not_passed:
            overall = "incomplete"
            missing = sorted(set(incomplete) | set(not_passed))
            rationales = [f"gated case(s) lack a passing, compared convergence result: {', '.join(missing)}"]
        else:
            overall = "ready"
            rationales = [f"all {len(gated)} gated case(s) passed against a reviewed baseline"]

    return {
        "overall_status": overall,
        "rationale": rationales,
        "gated_cases": [case.case_id for case in gated],
        "ungated_cases": [case.case_id for case in ungated],
        "status_counts": counts,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Render the release-readiness report as Markdown."""
    lines: list[str] = []
    lines.append("# L4 Convergence Release Readiness")
    lines.append("")
    lines.append(f"- Generated: {payload['generated_at']}")
    lines.append(f"- Commit: `{payload.get('commit_sha') or 'unknown'}`")
    lines.append(f"- Overall status: **{payload['overall_status'].upper()}**")
    for reason in payload["rationale"]:
        lines.append(f"- Reason: {reason}")
    lines.append("")
    lines.append("## Status counts")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("| --- | --- |")
    for status, count in payload["status_counts"].items():
        lines.append(f"| {status} | {count} |")
    lines.append("")
    lines.append("## Cases")
    lines.append("")
    lines.append("| Case | Gate | Status | Evidence | Converged | Steps | Failed metrics |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for case in payload["cases"]:
        steps = "n/a"
        if case["completed_steps"] is not None or case["expected_steps"] is not None:
            steps = f"{case['completed_steps']}/{case['expected_steps']}"
        failed = ", ".join(case["failed_metrics"]) or "-"
        lines.append(
            f"| `{case['case_id']}` | {'yes' if case['release_gate'] else 'no'} | {case['status']} | "
            f"{case['evidence_level']} | {'yes' if case['converged'] else 'no'} | {steps} | {failed} |"
        )
    lines.append("")
    lines.append("## How to read this")
    lines.append("")
    lines.append(
        "- `ready` means every gated case ran on real weights and a real dataset and matched a reviewed "
        "baseline within tolerance."
    )
    lines.append(
        "- `blocked` means a gated case ran and regressed. Inspect `failed_metrics` and the case log before "
        "refreshing any baseline."
    )
    lines.append(
        "- `incomplete` means the evidence does not exist yet (skipped for hardware, timed out, invalid, "
        "incomparable, or baseline creation only). It is never treated as a pass."
    )
    lines.append("")
    lines.append(
        "A case whose `evidence_level` is below `compared` cannot make a release ready, even if its `status` "
        "is `passed`."
    )
    return "\n".join(lines) + "\n"


def build_release_report(
    cases: list[CaseResult],
    *,
    commit_sha: str | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build the release-readiness payload for a set of case results."""
    summary = summarize_cases(cases)
    return {
        "schema_version": SCHEMA_VERSION,
        "layer": "L4",
        "generated_at": generated_at or datetime.now(UTC).isoformat(),
        "commit_sha": commit_sha,
        **summary,
        "cases": [case.as_dict() for case in cases],
    }


def write_release_report(payload: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write ``release_readiness.json`` and ``release_readiness.md``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "release_readiness.json"
    md_path = output_dir / "release_readiness.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    return json_path, md_path


def exit_code_for_status(overall_status: str) -> int:
    """Map an overall verdict to a process exit code.

    Only ``ready`` exits 0.  ``blocked`` and ``incomplete`` are both failures of
    the release gate, and they are reported distinctly in the artifact.
    """
    return 0 if overall_status == "ready" else 1
