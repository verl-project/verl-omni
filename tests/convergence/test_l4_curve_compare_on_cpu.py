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
"""L4 curve extraction and convergence comparison semantics.

All curves here are synthetic fixtures; they prove the scoring rules, not any
real training result.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.convergence.compare import build_contract, compare_contracts, compare_curves, score_metric
from tests.convergence.curves import (
    CurveError,
    CurvePoint,
    build_curves,
    completed_steps,
    load_step_records,
    summarize_curve,
)

REWARD = "critic/rewards/mean"
LOSS = "actor/loss"


def console_line(step: int, rewards: float, loss: float = 1.0) -> str:
    """Render one synthetic ``verl``-style console step record."""
    payload = {
        "training/global_step": step,
        REWARD: rewards,
        LOSS: loss,
        "actor/grad_norm": 0.5,
    }
    return f"2026-01-01 00:00:0{step % 10} INFO step:{step} - {payload}\n"


def write_console_log(path: Path, steps: list[tuple[int, float]]) -> Path:
    path.write_text("".join(console_line(step, reward) for step, reward in steps), encoding="utf-8")
    return path


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def curve(points: list[tuple[int, float]], direction: str = "higher") -> list[CurvePoint]:
    return [CurvePoint(step=step, value=value) for step, value in points]


def rising_curve(start: float = 0.10, step_delta: float = 0.01, steps: int = 12) -> list[CurvePoint]:
    return curve([(step, start + step * step_delta) for step in range(1, steps + 1)])


def summary_of(points: list[CurvePoint], *, direction: str = "higher", warmup: int = 1, window: int = 3):
    return summarize_curve(
        points,
        metric=REWARD,
        direction=direction,
        warmup_steps=warmup,
        final_window=window,
    )


# --------------------------------------------------------------------------------------
# Curve extraction
# --------------------------------------------------------------------------------------


def test_console_records_are_parsed_from_dict_payload(tmp_path: Path) -> None:
    log = write_console_log(tmp_path / "train.log", [(1, 0.10), (2, 0.20), (3, 0.30)])
    records = load_step_records(log_file=log)
    assert [record["step"] for record in records] == [1, 2, 3]
    curves = build_curves(records, [REWARD])
    assert [point.value for point in curves[REWARD]] == pytest.approx([0.10, 0.20, 0.30])


def test_console_records_fall_back_to_key_value_pairs(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "[INFO] step:1 - training/global_step:1 - critic/rewards/mean:0.25 - actor/loss:0.5\n",
        encoding="utf-8",
    )
    records = load_step_records(log_file=log)
    assert len(records) == 1
    assert records[0]["step"] == 1
    assert build_curves(records, [REWARD])[REWARD][0].value == pytest.approx(0.25)


def test_structured_jsonl_is_preferred_over_console(tmp_path: Path) -> None:
    log = write_console_log(tmp_path / "train.log", [(1, 0.10)])
    side_channel = write_jsonl(
        tmp_path / "metrics.jsonl",
        [{"step": 1, "data": {REWARD: 0.9}}, {"step": 2, "data": {REWARD: 0.95}}],
    )
    records = load_step_records(metrics_jsonl=side_channel, log_file=log)
    assert [record["step"] for record in records] == [1, 2]
    assert completed_steps(records) == 2


def test_malformed_jsonl_is_rejected(tmp_path: Path) -> None:
    side_channel = tmp_path / "metrics.jsonl"
    side_channel.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(CurveError):
        load_step_records(metrics_jsonl=side_channel)


def test_missing_outputs_raise_instead_of_returning_an_empty_curve(tmp_path: Path) -> None:
    with pytest.raises(CurveError):
        load_step_records(log_file=tmp_path / "absent.log")


def test_non_finite_values_are_preserved_and_flagged(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        console_line(1, 0.10) + console_line(2, float("nan")) + console_line(3, 0.30),
        encoding="utf-8",
    )
    records = load_step_records(log_file=log)
    points = build_curves(records, [REWARD])[REWARD]
    assert len(points) == 3
    summary = summarize_curve(points, metric=REWARD, direction="higher", warmup_steps=0, final_window=2)
    assert summary.num_points == 3
    assert summary.num_finite_points == 2
    assert summary.non_finite_steps == [2]
    assert summary.finite_fraction == pytest.approx(2 / 3)


def test_summarize_uses_only_post_warmup_steps() -> None:
    points = curve([(1, 100.0), (2, 100.0), (3, 1.0), (4, 2.0), (5, 3.0)])
    summary = summarize_curve(points, metric=REWARD, direction="higher", warmup_steps=2, final_window=2)
    assert summary.steps == [1, 2, 3, 4, 5]
    assert summary.first_window_mean == pytest.approx(1.5)
    assert summary.final_window_mean == pytest.approx(2.5)
    assert summary.improvement == pytest.approx(1.0)


def test_improvement_sign_flips_for_lower_is_better() -> None:
    points = curve([(1, 5.0), (2, 4.0), (3, 3.0)])
    higher = summarize_curve(points, metric=LOSS, direction="higher", warmup_steps=0, final_window=2)
    lower = summarize_curve(points, metric=LOSS, direction="lower", warmup_steps=0, final_window=2)
    assert higher.improvement == pytest.approx(-1.0)
    assert lower.improvement == pytest.approx(1.0)


def test_summary_survives_json_round_trip() -> None:
    payload = summary_of(rising_curve()).as_dict()
    assert json.loads(json.dumps(payload)) == payload


# --------------------------------------------------------------------------------------
# Comparison semantics
# --------------------------------------------------------------------------------------


def convergence(atol: float = 0.0, rtol: float = 0.1, min_improvement: float = -1.0) -> dict:
    return {"atol": atol, "rtol": rtol, "min_improvement": min_improvement, "require_finite": True}


def make_report(current_points, baseline_points, *, specs=None, conv=None, contracts=None):
    specs = specs or [{"name": REWARD, "direction": "higher"}]
    if contracts is None:
        contracts = ({"algorithm": "flow_grpo"}, {"algorithm": "flow_grpo"})
    baseline_contract, current_contract = contracts
    return compare_curves(
        current_summaries={REWARD: summary_of(current_points)},
        baseline_summaries={REWARD: summary_of(baseline_points)},
        metric_specs=specs,
        convergence=conv or convergence(),
        min_points=3,
        baseline_contract=baseline_contract,
        current_contract=current_contract,
        baseline_present=True,
    )


def test_matching_run_passes() -> None:
    report = make_report(rising_curve(), rising_curve())
    assert report.comparable
    assert report.passed
    assert report.status == "passed"


def test_improvement_never_fails() -> None:
    report = make_report(rising_curve(start=0.5), rising_curve(start=0.1))
    assert report.passed


def test_regression_beyond_tolerance_fails() -> None:
    report = make_report(rising_curve(start=0.10), rising_curve(start=0.50))
    assert not report.passed
    assert report.status == "failed"
    assert report.failed_metrics == [REWARD]
    assert "dropped" in report.verdicts[REWARD].reasons[0]


def test_lower_is_better_regression_fails() -> None:
    baseline = summary_of(curve([(1, 1.0), (2, 1.0), (3, 1.0)]), direction="lower", warmup=0, window=3)
    current = summary_of(curve([(1, 1.0), (2, 1.5), (3, 2.0)]), direction="lower", warmup=0, window=3)
    verdict = score_metric(
        LOSS,
        direction="lower",
        baseline=baseline,
        current=current,
        min_points=3,
        require_finite=True,
        atol=0.0,
        rtol=0.1,
        min_improvement=-1.0,
    )
    assert not verdict.passed


def test_tolerance_boundary_is_inclusive() -> None:
    """A drop of exactly atol + rtol * |reference| still passes."""
    baseline = summary_of(curve([(1, 1.0), (2, 1.0), (3, 1.0)]), warmup=0, window=3)
    current = summary_of(curve([(1, 0.8), (2, 0.8), (3, 0.8)]), warmup=0, window=3)
    verdict = score_metric(
        REWARD,
        direction="higher",
        baseline=baseline,
        current=current,
        min_points=3,
        require_finite=True,
        atol=0.0,
        rtol=0.2,
        min_improvement=-1.0,
    )
    assert verdict.tolerance == pytest.approx(0.2)
    assert verdict.passed, verdict.reasons


def test_absolute_tolerance_covers_near_zero_baseline() -> None:
    """A relative-only rule would be unusable when the reference is ~0.

    A loss baseline of exactly 0 has no meaningful relative scale, so only the
    absolute floor can decide whether a small rise is a regression.
    """
    baseline = summary_of(curve([(1, 0.0), (2, 0.0), (3, 0.0)]), direction="lower", warmup=0, window=3)
    small = summary_of(curve([(1, 0.01), (2, 0.01), (3, 0.01)]), direction="lower", warmup=0, window=3)
    large = summary_of(curve([(1, 0.5), (2, 0.5), (3, 0.5)]), direction="lower", warmup=0, window=3)
    passing = score_metric(
        LOSS,
        direction="lower",
        baseline=baseline,
        current=small,
        min_points=3,
        require_finite=True,
        atol=0.02,
        rtol=0.05,
        min_improvement=-1.0,
    )
    failing = score_metric(
        LOSS,
        direction="lower",
        baseline=baseline,
        current=large,
        min_points=3,
        require_finite=True,
        atol=0.02,
        rtol=0.05,
        min_improvement=-1.0,
    )
    assert passing.passed, passing.reasons
    assert not failing.passed
    assert failing.relative_delta == float("inf")


def test_non_finite_values_fail_the_run() -> None:
    baseline = summary_of(rising_curve(), warmup=0, window=3)
    current = summarize_curve(
        curve([(1, 0.1), (2, float("nan")), (3, 0.3), (4, 0.3)]),
        metric=REWARD,
        direction="higher",
        warmup_steps=0,
        final_window=3,
    )
    verdict = score_metric(
        REWARD,
        direction="higher",
        baseline=baseline,
        current=current,
        min_points=3,
        require_finite=True,
        atol=0.0,
        rtol=0.1,
        min_improvement=-1.0,
    )
    assert not verdict.passed
    assert any("non-finite" in reason for reason in verdict.reasons)


def test_insufficient_points_fail_even_when_the_mean_looks_fine() -> None:
    baseline = summary_of(curve([(1, 0.1), (2, 0.2), (3, 0.3)]), warmup=0, window=3)
    current = summary_of(curve([(1, 0.3), (2, 0.3)]), warmup=0, window=2)
    verdict = score_metric(
        REWARD,
        direction="higher",
        baseline=baseline,
        current=current,
        min_points=3,
        require_finite=True,
        atol=0.0,
        rtol=0.5,
        min_improvement=-1.0,
    )
    assert not verdict.passed
    assert any("finite point" in reason for reason in verdict.reasons)


def test_optional_metric_absent_from_baseline_is_not_scored() -> None:
    report = compare_curves(
        current_summaries={LOSS: summary_of(curve([(1, 1.0), (2, 0.5)]))},
        baseline_summaries={},
        metric_specs=[{"name": LOSS, "direction": "lower", "optional": True}],
        convergence=convergence(),
        min_points=2,
        baseline_contract={"algorithm": "gspo"},
        current_contract={"algorithm": "gspo"},
        baseline_present=True,
    )
    assert report.verdicts[LOSS].scored is False
    assert report.passed


def test_required_metric_absent_from_baseline_fails() -> None:
    report = compare_curves(
        current_summaries={LOSS: summary_of(curve([(1, 1.0), (2, 0.5)]))},
        baseline_summaries={},
        metric_specs=[{"name": LOSS, "direction": "lower", "optional": False}],
        convergence=convergence(),
        min_points=2,
        baseline_contract={"algorithm": "gspo"},
        current_contract={"algorithm": "gspo"},
        baseline_present=True,
    )
    assert not report.passed
    assert report.status == "failed"


def test_missing_baseline_is_invalid_and_never_passed() -> None:
    report = compare_curves(
        current_summaries={REWARD: summary_of(rising_curve())},
        baseline_summaries={},
        metric_specs=[{"name": REWARD, "direction": "higher"}],
        convergence=convergence(),
        min_points=3,
        baseline_contract=None,
        current_contract=None,
        baseline_present=False,
    )
    assert report.status == "invalid"
    assert report.passed is False


def test_contract_mismatch_is_incomparable_not_a_pass() -> None:
    baseline_contract = {"algorithm": "flow_grpo", "precision": "bf16", "metric_set": ["rewards"], "hardware": "sm90"}
    current_contract = {"algorithm": "flow_grpo", "precision": "fp32", "metric_set": ["rewards"], "hardware": "sm90"}
    report = make_report(
        rising_curve(),
        rising_curve(),
        contracts=(baseline_contract, current_contract),
    )
    assert not report.comparable
    assert report.status == "incomparable"
    assert report.passed is False
    assert any(item["field"] == "precision" for item in report.mismatches)


def test_commit_sha_difference_does_not_break_comparability() -> None:
    baseline_contract = {"algorithm": "gspo", "commit_sha": "aaaa"}
    current_contract = {"algorithm": "gspo", "commit_sha": "bbbb"}
    comparable, mismatches = compare_contracts(baseline_contract, current_contract)
    assert comparable
    assert mismatches == []


def test_build_contract_records_provenance_without_paths() -> None:
    recipe_data = {
        "models": {"policy": {"source": "Qwen/Qwen-Image", "revision": "main"}},
        "dataset": {"train": "data/train.parquet", "val": "data/test.parquet"},
        "metrics": {"tracked": [{"name": REWARD, "direction": "higher"}], "warmup_steps": 1, "final_window": 2},
        "budget": {"total_training_steps": 10},
    }
    contract = build_contract(
        algorithm="flow_grpo",
        precision="bf16",
        recipe_data=recipe_data,
        observed_hardware={"gpu_architecture": ["sm89"], "gpu_count": 2},
        commit_sha="deadbeef",
    )
    assert contract["algorithm"] == "flow_grpo"
    assert contract["precision"] == "bf16"
    assert contract["steps"] == 10
    assert contract["hardware"] == {"gpu_architecture": ["sm89"], "gpu_count": 2}
    assert contract["commit_sha"] == "deadbeef"
    assert contract["models"]["policy"]["source"] == "Qwen/Qwen-Image"
