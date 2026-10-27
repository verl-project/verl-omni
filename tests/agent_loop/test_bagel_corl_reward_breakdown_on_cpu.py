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

"""Training must publish the reward breakdown, not just ``critic/score/mean``."""

import pytest

from verl_omni.agent_loop.bagel_corl_lib import (
    TRAIN_REWARD_DIM_KEYS,
    aggregate_reward_breakdown_metrics,
    und_record_field_keys,
)
from verl_omni.utils.reward_score.bagel_corl_multidim_reward import REWARD_COMPONENTS


def _record(
    *,
    score: float | None = None,
    und_reward: float | None = None,
    credit: bool | None = None,
    breakdown: dict | None = None,
    nested: bool = True,
    extra: dict | None = None,
) -> dict:
    """One UND TQ row in the shape ``_und_records_from_batch`` produces."""
    fields: dict = {}
    if score is not None:
        fields["token_level_scores"] = score
    if und_reward is not None:
        fields["und_reward"] = und_reward
    if credit is not None:
        fields["used_image_credit"] = credit
    if breakdown is not None:
        if nested:
            fields["extra_fields"] = {"reward_extra_info": breakdown}
        else:
            fields["reward_extra_info"] = breakdown
    if extra:
        fields.update(extra)
    return {"fields": fields}


def _authored_record(und_reward: float, credit: bool, breakdown: dict) -> dict:
    """A row exactly as ``bagel_corl.episode_from_agent_extra`` writes it."""
    return {
        "fields": {
            "und_reward": und_reward,
            "used_image_credit": credit,
            "reward_extra_info": breakdown,
            "bagel_role": "und",
            "episode_J": 4.0,
        }
    }


def test_dim_keys_stay_in_sync_with_the_reward_module():
    """The tuple is duplicated to keep this module import-light, so pin it."""
    assert set(TRAIN_REWARD_DIM_KEYS) == set(REWARD_COMPONENTS)


def test_breakdown_is_read_from_nested_extra_fields():
    metrics = aggregate_reward_breakdown_metrics(
        [
            _record(breakdown={"reward_image": 0.6, "reward_done": 1.0}),
            _record(breakdown={"reward_image": 0.9, "reward_done": 0.0}),
        ]
    )
    assert metrics["bagel_corl/reward_dim/image/mean"] == 0.75
    assert metrics["bagel_corl/reward_dim/image/min"] == 0.6
    assert metrics["bagel_corl/reward_dim/image/max"] == 0.9
    assert metrics["bagel_corl/reward_dim/done/mean"] == 0.5


def test_breakdown_is_read_from_the_flattened_row_too():
    """The TensorDict branch flattens ``extra_fields`` onto the row."""
    metrics = aggregate_reward_breakdown_metrics([_record(breakdown={"reward_image": 0.4}, nested=False)])
    assert metrics["bagel_corl/reward_dim/image/mean"] == 0.4


def test_used_image_credit_splits_the_episode_reward():
    """The failures and the successes must not be averaged into one number."""
    metrics = aggregate_reward_breakdown_metrics(
        [
            _record(score=0.0, credit=False),
            _record(score=0.0, credit=False),
            _record(score=0.8, credit=True),
            _record(score=0.6, credit=True),
        ]
    )
    assert metrics["bagel_corl/reward_valid/mean"] == 0.7
    assert metrics["bagel_corl/reward_valid/count"] == 2
    assert metrics["bagel_corl/reward_no_image/mean"] == 0.0
    assert metrics["bagel_corl/reward_no_image/count"] == 2


def test_the_split_works_on_the_row_shape_the_agent_loop_actually_writes():
    """Regression: a row carrying only ``und_reward`` must still split."""
    metrics = aggregate_reward_breakdown_metrics(
        [
            _authored_record(0.0, False, {"reward_image": 0.0}),
            _authored_record(0.9, True, {"reward_image": 0.8}),
            _authored_record(0.5, True, {"reward_image": 0.4}),
        ]
    )
    assert metrics["bagel_corl/reward_valid/mean"] == 0.7
    assert metrics["bagel_corl/reward_valid/count"] == 2
    assert metrics["bagel_corl/reward_no_image/mean"] == 0.0
    assert metrics["bagel_corl/reward_no_image/count"] == 1
    assert metrics["bagel_corl/reward_dim/image/mean"] == pytest.approx(0.4)


def test_token_level_scores_wins_when_both_spellings_are_present():
    """``token_level_scores`` is what ``critic/score/mean`` sums, so it is preferred."""
    metrics = aggregate_reward_breakdown_metrics([_record(score=0.8, und_reward=0.2, credit=True)])
    assert metrics["bagel_corl/reward_valid/mean"] == 0.8


class TestTrainedRewardPrecedence:
    """The split must summarise the reward the actor trains on, not a namesake field."""

    def test_rm_scores_is_summed_and_beats_every_other_spelling(self):
        metrics = aggregate_reward_breakdown_metrics(
            [
                _record(
                    credit=True, extra={"rm_scores": [0.0, 0.0, 0.7], "token_level_scores": 0.1, "und_reward": 0.2}
                ),
                _record(credit=True, extra={"rm_scores": [0.9], "token_level_scores": 0.1, "und_reward": 0.2}),
            ]
        )
        assert metrics["bagel_corl/reward_valid/mean"] == pytest.approx(0.8)  # (0.7 + 0.9) / 2

    def test_token_level_scores_is_summed_when_rm_scores_is_absent(self):
        metrics = aggregate_reward_breakdown_metrics(
            [_record(credit=True, extra={"token_level_scores": [0.25, 0.25], "und_reward": 0.9})]
        )
        assert metrics["bagel_corl/reward_valid/mean"] == pytest.approx(0.5)

    def test_an_unreadable_training_column_falls_through_to_the_next(self):
        """A present-but-unusable ``rm_scores`` must not shadow a readable one."""
        metrics = aggregate_reward_breakdown_metrics(
            [_record(credit=True, extra={"rm_scores": "not-a-vector", "token_level_scores": 0.6})]
        )
        assert metrics["bagel_corl/reward_valid/mean"] == pytest.approx(0.6)

    def test_rm_scores_is_not_mistaken_for_a_sequence_of_episodes(self):
        """Summing is what makes 16 tokens one episode; a mean would divide it by its length."""
        one_token = aggregate_reward_breakdown_metrics([_record(credit=True, extra={"rm_scores": [0.6]})])
        four_tokens = aggregate_reward_breakdown_metrics(
            [_record(credit=True, extra={"rm_scores": [0.15, 0.15, 0.15, 0.15]})]
        )
        assert one_token["bagel_corl/reward_valid/mean"] == pytest.approx(0.6)
        assert four_tokens["bagel_corl/reward_valid/mean"] == pytest.approx(0.6)

    def test_a_boolean_is_not_a_reward(self):
        """``bool`` is an ``int`` in Python; a flag must not be read as 1.0."""
        metrics = aggregate_reward_breakdown_metrics(
            [_record(credit=True, extra={"rm_scores": True, "und_reward": 0.4})]
        )
        assert metrics["bagel_corl/reward_valid/mean"] == pytest.approx(0.4)


def test_a_row_without_the_credit_flag_is_not_assumed_to_lack_credit():
    """Unknown is not the same as failed; counting it as failed inflates the failure rate."""
    metrics = aggregate_reward_breakdown_metrics([_record(und_reward=0.9, credit=True), _record(und_reward=0.1)])
    assert metrics["bagel_corl/reward_valid/count"] == 1
    assert "bagel_corl/reward_no_image/mean" not in metrics


def test_count_is_published_so_a_shrinking_subset_is_visible():
    """A mean over 1 of 16 episodes is not comparable to a mean over 15 of 16."""
    metrics = aggregate_reward_breakdown_metrics(
        [_record(score=0.5, credit=True), _record(score=0.5, credit=True), _record(score=0.0, credit=False)]
    )
    assert metrics["bagel_corl/reward_valid/count"] == 2
    assert metrics["bagel_corl/reward_no_image/count"] == 1


def test_absent_dim_is_not_reported_as_zero():
    """A missing scorer and a scorer that returned 0 are different measurements."""
    metrics = aggregate_reward_breakdown_metrics([_record(score=0.5, credit=True)])
    assert not any(key.startswith("bagel_corl/reward_dim/") for key in metrics)
    assert metrics["bagel_corl/reward_valid/mean"] == 0.5


def test_a_dim_present_on_only_some_rows_averages_only_those_rows():
    metrics = aggregate_reward_breakdown_metrics(
        [
            _record(breakdown={"reward_image": 1.0}),
            _record(breakdown={"reward_image": 0.0}),
            _record(breakdown={"reward_format": 0.5}),
        ]
    )
    assert metrics["bagel_corl/reward_dim/image/mean"] == 0.5
    assert metrics["bagel_corl/reward_dim/image/min"] == 0.0
    assert metrics["bagel_corl/reward_dim/image/max"] == 1.0
    assert metrics["bagel_corl/reward_dim/format/mean"] == 0.5


def test_rows_without_credit_or_score_are_skipped_not_counted():
    """The GEN-shaped rows carry neither field; they must not enter either mean."""
    metrics = aggregate_reward_breakdown_metrics(
        [
            _record(score=0.9, credit=True),
            _record(score=0.3),  # no credit flag
            _record(credit=True),  # no score
            {},
            {"fields": None},
            "not-a-record",
        ]
    )
    assert metrics["bagel_corl/reward_valid/mean"] == 0.9
    assert metrics["bagel_corl/reward_valid/count"] == 1
    assert "bagel_corl/reward_no_image/mean" not in metrics


def test_unreadable_values_are_dropped_rather_than_raised():
    metrics = aggregate_reward_breakdown_metrics(
        [
            _record(score=0.75, credit=True, breakdown={"reward_image": "not-a-number", "reward_done": 1}),
            _record(score="bad", credit=True),
        ]
    )
    assert metrics["bagel_corl/reward_dim/done/mean"] == 1.0
    assert "bagel_corl/reward_dim/image/mean" not in metrics
    assert metrics["bagel_corl/reward_valid/mean"] == 0.75


def test_empty_batch_publishes_nothing():
    assert aggregate_reward_breakdown_metrics([]) == {}


class TestUndRecordFieldKeys:
    """The empty-breakdown warning names the row's fields; that helper has to be total."""

    def test_returns_sorted_keys_of_the_wrapped_shape(self):
        assert und_record_field_keys([_record(score=0.5, credit=True)]) == [
            "token_level_scores",
            "used_image_credit",
        ]

    def test_accepts_a_bare_dict_row(self):
        assert und_record_field_keys([{"und_reward": 0.4, "episode_J": 3}]) == [
            "episode_J",
            "und_reward",
        ]

    def test_limit_truncates_without_reordering(self):
        keys = und_record_field_keys([_record(score=0.5, credit=True, extra={"z": 1, "a": 2})], limit=2)
        assert keys == ["a", "token_level_scores"]

    def test_empty_and_unreadable_records_yield_no_keys(self):
        assert und_record_field_keys([]) == []
        assert und_record_field_keys([{}]) == []
        assert und_record_field_keys([{"fields": None}]) == []
        assert und_record_field_keys(["not-a-record"]) == []
