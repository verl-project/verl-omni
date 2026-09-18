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
"""CPU tests for the paired per-prompt validation metric.

The load-bearing case is the one the module exists for: a *positive mean* whose
interval still spans zero has to come back ``inconclusive``.  If that case ever
regressed to ``signal_positive``, the layer would start reporting noise as a
result.
"""

from __future__ import annotations

import json

import pytest

from tests.convergence.paired_validation import (
    VERDICT_INCONCLUSIVE,
    VERDICT_NEGATIVE,
    VERDICT_POSITIVE,
    PairedValidationError,
    bootstrap_ci,
    load_scores,
    main,
    paired_differences,
    render,
    score_pair,
)


def write_dump(path, scores, field="score"):
    """Write a validation dump with one row per score."""
    with open(path, "w") as handle:
        for index, value in enumerate(scores):
            handle.write(json.dumps({"input": f"prompt-{index}", field: value}) + "\n")
    return path


class TestLoadScores:
    def test_preserves_file_order(self, tmp_path):
        path = write_dump(tmp_path / "0.jsonl", [0.3, 0.1, 0.2])
        assert load_scores(path) == [0.3, 0.1, 0.2]

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "0.jsonl"
        path.write_text('{"score": 0.5}\n\n{"score": 0.25}\n')
        assert load_scores(path) == [0.5, 0.25]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(PairedValidationError, match="not found"):
            load_scores(tmp_path / "absent.jsonl")

    def test_missing_field_raises(self, tmp_path):
        path = tmp_path / "0.jsonl"
        path.write_text('{"other": 1.0}\n')
        with pytest.raises(PairedValidationError, match="no 'score' field"):
            load_scores(path)

    def test_non_numeric_raises(self, tmp_path):
        path = tmp_path / "0.jsonl"
        path.write_text('{"score": "high"}\n')
        with pytest.raises(PairedValidationError, match="non-numeric"):
            load_scores(path)

    def test_bool_is_not_a_score(self, tmp_path):
        path = tmp_path / "0.jsonl"
        path.write_text('{"score": true}\n')
        with pytest.raises(PairedValidationError, match="non-numeric"):
            load_scores(path)

    def test_empty_dump_raises(self, tmp_path):
        path = tmp_path / "0.jsonl"
        path.write_text("\n")
        with pytest.raises(PairedValidationError, match="no rows"):
            load_scores(path)

    def test_custom_field(self, tmp_path):
        path = write_dump(tmp_path / "0.jsonl", [0.75], field="reward")
        assert load_scores(path, field="reward") == [0.75]


class TestPairedDifferences:
    def test_pairs_by_index(self):
        assert paired_differences([0.1, 0.2], [0.4, 0.2]) == pytest.approx([0.3, 0.0])

    def test_length_mismatch_raises(self):
        with pytest.raises(PairedValidationError, match="prompt sets differ"):
            paired_differences([0.1, 0.2], [0.3])


class TestBootstrap:
    def test_is_deterministic_for_a_seed(self):
        diffs = [0.1, -0.2, 0.3, 0.0, 0.5, -0.1]
        assert bootstrap_ci(diffs, resamples=2000, seed=7) == bootstrap_ci(
            diffs, resamples=2000, seed=7
        )

    def test_interval_narrows_as_prompts_grow(self):
        """More sampled prompts must buy a tighter interval, not a looser one."""
        spread = [((i * 37) % 101) / 100.0 - 0.4 for i in range(64)]
        widths = []
        for count in (8, 16, 32, 64):
            low, high = bootstrap_ci(spread[:count], resamples=2000, seed=5)
            widths.append(high - low)
        assert widths == sorted(widths, reverse=True)
        assert widths[-1] < widths[0]

    def test_zero_variance_collapses_to_the_mean(self):
        low, high = bootstrap_ci([0.25] * 8, resamples=500, seed=3)
        assert low == pytest.approx(0.25)
        assert high == pytest.approx(0.25)

    def test_interval_brackets_the_mean(self):
        diffs = [0.4, 0.2, -0.1, 0.3, 0.0, 0.6, -0.3, 0.1]
        low, high = bootstrap_ci(diffs, resamples=2000, seed=11)
        mean = sum(diffs) / len(diffs)
        assert low <= mean <= high

    def test_empty_sample_raises(self):
        with pytest.raises(PairedValidationError, match="empty sample"):
            bootstrap_ci([])

    @pytest.mark.parametrize("resamples", [0, -1])
    def test_non_positive_resamples_raise(self, resamples):
        with pytest.raises(PairedValidationError, match="resamples must be positive"):
            bootstrap_ci([0.1], resamples=resamples)

    @pytest.mark.parametrize("confidence", [0.0, 1.0, -0.5])
    def test_invalid_confidence_raises(self, confidence):
        with pytest.raises(PairedValidationError, match="confidence must be in"):
            bootstrap_ci([0.1], confidence=confidence)


class TestVerdict:
    def test_consistent_gain_is_a_positive_signal(self, tmp_path):
        before = write_dump(tmp_path / "0.jsonl", [0.0] * 16)
        after = write_dump(tmp_path / "48.jsonl", [0.9] * 16)
        result = score_pair(before, after, resamples=1000)
        assert result.verdict == VERDICT_POSITIVE
        assert result.mean_difference == pytest.approx(0.9)

    def test_consistent_loss_is_a_negative_signal(self, tmp_path):
        before = write_dump(tmp_path / "0.jsonl", [0.8] * 16)
        after = write_dump(tmp_path / "48.jsonl", [0.1] * 16)
        result = score_pair(before, after, resamples=1000)
        assert result.verdict == VERDICT_NEGATIVE

    def test_positive_mean_with_wide_spread_is_inconclusive(self, tmp_path):
        """The case this module exists for: +mean, but the interval spans zero."""
        before = write_dump(tmp_path / "0.jsonl", [0.5] * 10)
        after = write_dump(
            tmp_path / "48.jsonl",
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        )
        result = score_pair(before, after, resamples=4000)
        assert result.mean_difference > 0.0
        assert result.ci_low < 0.0 < result.ci_high
        assert result.verdict == VERDICT_INCONCLUSIVE

    def test_cancelling_changes_are_inconclusive(self, tmp_path):
        before = write_dump(tmp_path / "0.jsonl", [0.5] * 8)
        after = write_dump(tmp_path / "48.jsonl", [1.0] * 4 + [0.0] * 4)
        result = score_pair(before, after, resamples=2000)
        assert result.mean_difference == pytest.approx(0.0)
        assert result.verdict == VERDICT_INCONCLUSIVE


class TestScorePair:
    def test_tallies_direction_counts(self, tmp_path):
        before = write_dump(tmp_path / "0.jsonl", [0.2, 0.2, 0.2, 0.2])
        after = write_dump(tmp_path / "48.jsonl", [0.5, 0.2, 0.0, 0.7])
        result = score_pair(before, after, resamples=500)
        assert (result.improved, result.unchanged, result.worse) == (2, 1, 1)

    def test_reports_means_and_sample_count(self, tmp_path):
        before = write_dump(tmp_path / "0.jsonl", [0.25, 0.75])
        after = write_dump(tmp_path / "48.jsonl", [0.75, 0.75])
        result = score_pair(before, after, resamples=500)
        assert result.samples == 2
        assert result.before_mean == pytest.approx(0.5)
        assert result.after_mean == pytest.approx(0.75)
        assert result.mean_difference == pytest.approx(0.25)

    def test_mismatched_prompt_counts_raise(self, tmp_path):
        before = write_dump(tmp_path / "0.jsonl", [0.2, 0.2])
        after = write_dump(tmp_path / "48.jsonl", [0.5])
        with pytest.raises(PairedValidationError, match="prompt sets differ"):
            score_pair(before, after)

    def test_as_dict_is_json_serialisable(self, tmp_path):
        """Two prompts cannot support a signal, so this pair must stay inconclusive."""
        before = write_dump(tmp_path / "0.jsonl", [0.2, 0.4])
        after = write_dump(tmp_path / "48.jsonl", [0.6, 0.4])
        payload = score_pair(before, after, resamples=500).as_dict()
        assert payload["mean_difference"] == pytest.approx(0.2)
        assert payload["verdict"] == VERDICT_INCONCLUSIVE
        assert json.loads(json.dumps(payload))["samples"] == 2


class TestRenderAndCli:
    def test_render_reports_the_interval(self, tmp_path):
        before = write_dump(tmp_path / "0.jsonl", [0.0, 0.0])
        after = write_dump(tmp_path / "48.jsonl", [1.0, 0.0])
        text = render(score_pair(before, after, resamples=500))
        assert "paired mean diff" in text
        assert "CI" in text
        assert "verdict" in text

    def test_cli_exit_zero_for_a_signal(self, tmp_path, capsys):
        before = write_dump(tmp_path / "0.jsonl", [0.0] * 12)
        after = write_dump(tmp_path / "48.jsonl", [0.9] * 12)
        code = main(["--before", str(before), "--after", str(after), "--resamples", "1000"])
        assert code == 0
        assert "signal_positive" in capsys.readouterr().out

    def test_cli_exit_one_for_inconclusive(self, tmp_path, capsys):
        before = write_dump(tmp_path / "0.jsonl", [0.5] * 10)
        after = write_dump(
            tmp_path / "48.jsonl",
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        )
        code = main(["--before", str(before), "--after", str(after), "--resamples", "4000"])
        assert code == 1
        assert "inconclusive" in capsys.readouterr().out

    def test_cli_exit_two_on_a_bad_pair(self, tmp_path, capsys):
        before = write_dump(tmp_path / "0.jsonl", [0.5, 0.5])
        after = write_dump(tmp_path / "48.jsonl", [0.5])
        code = main(["--before", str(before), "--after", str(after)])
        assert code == 2
        assert "error:" in capsys.readouterr().out

    def test_cli_json_mode(self, tmp_path, capsys):
        before = write_dump(tmp_path / "0.jsonl", [0.1])
        after = write_dump(tmp_path / "48.jsonl", [0.9])
        main(["--before", str(before), "--after", str(after), "--resamples", "500", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["samples"] == 1
        assert payload["verdict"] == VERDICT_POSITIVE
