"""Tests for choice_reward.compute_score, covering issue #643."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).parent.parent.parent.parent / "verl_omni" / "utils" / "reward_score" / "choice_reward.py"
    spec = importlib.util.spec_from_file_location("choice_reward", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


choice_reward = _load_module()
compute_score = choice_reward.compute_score


class TestCorrectBehavior:
    """Well-formed inputs still work exactly as before."""

    def test_tagged_response_matches_tagged_gt(self):
        assert compute_score("<answer>B</answer>", "<answer>B</answer>")["score"] == 1.0

    def test_tagged_response_differs_from_tagged_gt(self):
        assert compute_score("<answer>A</answer>", "<answer>B</answer>")["score"] == 0.0

    def test_tagged_response_matches_bare_label_gt(self):
        assert compute_score("<answer>B</answer>", "B")["score"] == 1.0

    def test_tagged_response_differs_from_bare_label_gt(self):
        assert compute_score("<answer>A</answer>", "B")["score"] == 0.0

    def test_first_tag_wins(self):
        assert compute_score("<answer>A</answer> <answer>B</answer>", "<answer>A</answer>")["score"] == 1.0


class TestIssue643Regression:
    """Empty or malformed responses must never score 1.0 (the bug from issue #643)."""

    def test_empty_response_bare_label_gt(self):
        assert compute_score("", "B")["score"] == 0.0

    def test_malformed_response_bare_label_gt(self):
        assert compute_score("I ran out of tokens before answering", "B")["score"] == 0.0

    def test_garbage_response_bare_label_gt(self):
        assert compute_score("asdkjfh", "B")["score"] == 0.0

    def test_empty_response_tagged_gt(self):
        assert compute_score("", "<answer>B</answer>")["score"] == 0.0

    def test_malformed_response_tagged_gt(self):
        assert compute_score("truncated generation without closing", "<answer>B</answer>")["score"] == 0.0

    def test_empty_response_empty_gt(self):
        assert compute_score("", "")["score"] == 0.0

    def test_none_response(self):
        assert compute_score(None, "B")["score"] == 0.0


class TestEdgeCases:
    """Graceful handling of unusual inputs."""

    def test_none_gt_falls_back_to_empty(self):
        assert compute_score("<answer>B</answer>", None)["score"] == 0.0

    def test_whitespace_gt(self):
        assert compute_score("<answer>B</answer>", "  ")["score"] == 0.0

    def test_both_untagged_non_matching(self):
        assert compute_score("The answer is A", "B")["score"] == 0.0

    def test_both_untagged_matching(self):
        assert compute_score("B", "B")["score"] == 0.0
