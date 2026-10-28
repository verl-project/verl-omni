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
"""L4 release-readiness report semantics.

The single most important property tested here is that absent or inconclusive
evidence can never produce ``ready``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.convergence.report import (
    CaseResult,
    ReportError,
    build_release_report,
    case_result_from_payload,
    exit_code_for_status,
    load_case_results,
    render_markdown,
    summarize_cases,
    write_release_report,
)


def case(
    case_id: str,
    status: str,
    evidence: str = "compared",
    *,
    converged: bool | None = None,
    release_gate: bool = True,
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        title=f"{case_id} title",
        status=status,
        evidence_level=evidence,
        converged=(status == "passed") if converged is None else converged,
        release_gate=release_gate,
        completed_steps=10,
        expected_steps=10,
    )


def test_all_gated_cases_passing_is_ready() -> None:
    report = build_release_report([case("a", "passed"), case("b", "passed")])
    assert report["overall_status"] == "ready"
    assert exit_code_for_status(report["overall_status"]) == 0


@pytest.mark.parametrize(
    ("status", "evidence"),
    [
        ("skipped", "static"),
        ("timeout", "run"),
        ("invalid", "run"),
        ("not_run", "static"),
        ("baseline_created", "run"),
        ("incomparable", "compared"),
    ],
)
def test_inconclusive_statuses_are_incomplete(status: str, evidence: str) -> None:
    report = build_release_report([case("a", "passed"), case("b", status, evidence)])
    assert report["overall_status"] == "incomplete"
    assert exit_code_for_status(report["overall_status"]) == 1


def test_a_failed_gated_case_blocks_the_release() -> None:
    report = build_release_report([case("ok", "passed"), case("broken", "failed")])
    assert report["overall_status"] == "blocked"


def test_passed_without_comparison_evidence_is_not_ready() -> None:
    """A run that only reached ``run`` cannot certify convergence."""
    report = build_release_report([case("a", "passed", "run")])
    assert report["overall_status"] == "incomplete"


def test_ungated_cases_do_not_block_readiness() -> None:
    report = build_release_report([case("gated", "passed"), case("dev", "skipped", release_gate=False)])
    assert report["overall_status"] == "ready"
    assert report["gated_cases"] == ["gated"]
    assert report["ungated_cases"] == ["dev"]


def test_no_gated_case_is_incomplete() -> None:
    report = build_release_report([case("dev", "passed", release_gate=False)])
    assert report["overall_status"] == "incomplete"
    assert "no gated" in report["rationale"][0]


def test_status_counts_include_every_case() -> None:
    report = build_release_report([case("a", "passed"), case("b", "skipped", "static"), case("c", "failed")])
    assert report["status_counts"]["passed"] == 1
    assert report["status_counts"]["skipped"] == 1
    assert report["status_counts"]["failed"] == 1
    assert sum(report["status_counts"].values()) == 3


def test_report_is_json_serializable() -> None:
    report = build_release_report([case("a", "passed")], commit_sha="deadbeef")
    assert json.loads(json.dumps(report))["commit_sha"] == "deadbeef"


def test_markdown_mentions_the_verdict_and_cases() -> None:
    report = build_release_report([case("qwen_image_flowgrpo", "skipped", "static")])
    markdown = render_markdown(report)
    assert "INCOMPLETE" in markdown
    assert "qwen_image_flowgrpo" in markdown
    assert "How to read this" in markdown


def test_markdown_and_json_agree_on_every_status() -> None:
    cases = [case("a", "passed"), case("b", "failed"), case("c", "skipped", "static")]
    report = build_release_report(cases)
    markdown = render_markdown(report)
    for item in report["cases"]:
        assert f"`{item['case_id']}`" in markdown
        assert item["status"] in markdown


def test_write_release_report_creates_both_artifacts(tmp_path: Path) -> None:
    report = build_release_report([case("a", "passed")])
    json_path, md_path = write_release_report(report, tmp_path / "out")
    assert json_path.is_file() and md_path.is_file()
    assert json.loads(json_path.read_text(encoding="utf-8"))["overall_status"] == "ready"


# --------------------------------------------------------------------------------------
# Loading and rejecting malformed results
# --------------------------------------------------------------------------------------


def result_payload(case_id: str = "a", status: str = "passed") -> dict:
    return {
        "schema_version": 1,
        "layer": "L4",
        "case_id": case_id,
        "title": "synthetic",
        "status": status,
        "evidence_level": "compared",
        "converged": status == "passed",
        "release_gate": True,
        "duration_s": 1.0,
        "run": {"completed_steps": 3, "expected_steps": 4, "log_path": "/tmp/train.log"},
        "comparison": {"failed_metrics": [] if status == "passed" else ["critic/rewards/mean"]},
        "failure_reason": None,
    }


def test_load_case_results_reads_every_case_directory(tmp_path: Path) -> None:
    for case_id in ("a", "b"):
        target = tmp_path / case_id
        target.mkdir()
        (target / "result.json").write_text(json.dumps(result_payload(case_id)), encoding="utf-8")
    results = load_case_results(tmp_path)
    assert [item.case_id for item in results] == ["a", "b"]


def test_load_case_results_rejects_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ReportError):
        load_case_results(tmp_path)


def test_unknown_status_is_rejected() -> None:
    payload = result_payload()
    payload["status"] = "probably-fine"
    with pytest.raises(ReportError):
        case_result_from_payload(payload)


def test_non_l4_payload_is_rejected() -> None:
    payload = result_payload()
    payload["layer"] = "L3"
    with pytest.raises(ReportError):
        case_result_from_payload(payload)


def test_unknown_evidence_level_is_rejected() -> None:
    payload = result_payload()
    payload["evidence_level"] = "vibes"
    with pytest.raises(ReportError):
        case_result_from_payload(payload)


def test_summarize_cases_requires_gated_cases_to_have_compared_evidence() -> None:
    summary = summarize_cases([case("a", "passed", "run")])
    assert summary["overall_status"] == "incomplete"
