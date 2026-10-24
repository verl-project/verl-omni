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

"""Aggregating the reward breakdown is not enough -- the keys have to survive the bus.

The dual-lane bookkeeping travels driver-side on the meta's ``extra_info``, and three
separate copies gate it: ``_compute_advantage`` publishes ``extra`` onto ``metrics``,
``super()._compute_advantage`` returns a *different* meta whose ``extra_info`` is rebuilt,
and ``_update_actor`` re-copies onto the object the step actually logs. A key that misses
any one of the three is computed and then thrown away, which is indistinguishable from
never having computed it.
"""

from verl_omni.agent_loop.bagel_corl_lib import aggregate_reward_breakdown_metrics
from verl_omni.trainer.omni.bagel_corl_trainer import (
    _BAGEL_CORL_STEP_METRIC_KEYS,
    _BAGEL_CORL_STEP_METRIC_PREFIXES,
    _bagel_corl_step_metric_keys,
)


def _record(score, credit, breakdown):
    return {
        "fields": {
            "token_level_scores": score,
            "used_image_credit": credit,
            "extra_fields": {"reward_extra_info": breakdown},
        }
    }


def test_every_key_the_aggregator_emits_survives_the_allowlist():
    """The regression that matters: a new metric family silently dropped by the bus."""
    records = [
        _record(0.8, True, {"reward_image": 0.7, "reward_done": 1.0}),
        _record(0.0, False, {"reward_image": 0.0, "reward_done": 0.0}),
    ]
    aggregated = aggregate_reward_breakdown_metrics(records)
    assert aggregated, "fixture must produce metrics"
    extra = dict(aggregated)
    extra["episode/J"] = 4.0

    published = set(_bagel_corl_step_metric_keys(extra))
    assert set(aggregated) <= published, sorted(set(aggregated) - published)
    assert "episode/J" in published


def test_prefix_families_match_the_aggregator_namespace():
    """Any ``bagel_corl/reward_*`` key the lib adds must be covered by a prefix."""
    records = [_record(0.5, True, {"reward_image": 0.5})]
    for key in aggregate_reward_breakdown_metrics(records):
        assert key.startswith(_BAGEL_CORL_STEP_METRIC_PREFIXES), key


def test_static_keys_are_reported_only_when_present():
    assert _bagel_corl_step_metric_keys({}) == []
    assert _bagel_corl_step_metric_keys({"episode/K_zero": 0.0}) == ["episode/K_zero"]


def test_a_key_held_as_none_is_not_published():
    """A copy of ``None`` reads as a measurement downstream; it must not be forwarded."""
    assert _bagel_corl_step_metric_keys({"episode/J": None}) == []
    assert _bagel_corl_step_metric_keys({"bagel_corl/reward_valid/mean": None}) == []


def test_static_keys_come_first_and_prefix_keys_are_sorted():
    extra = {
        "bagel_corl/reward_valid/mean": 0.7,
        "episode/K": 1.5,
        "bagel_corl/reward_dim/image/mean": 0.6,
        "und/no_image_credit": 0.0,
    }
    keys = _bagel_corl_step_metric_keys(extra)
    assert keys[:2] == ["und/no_image_credit", "episode/K"]
    assert keys[2:] == ["bagel_corl/reward_dim/image/mean", "bagel_corl/reward_valid/mean"]


def test_an_unrelated_key_is_never_published():
    """``extra`` carries the whole lane state; only the allowlist reaches the step log."""
    extra = {"bagel_corl_gen": {"advantages": 1}, "raw_prompt": ["secret"], "episode/K": 2.0}
    assert _bagel_corl_step_metric_keys(extra) == ["episode/K"]


def test_allowlist_static_keys_are_the_documented_five():
    assert _BAGEL_CORL_STEP_METRIC_KEYS == (
        "und/no_image_credit",
        "gen/dropped_incomplete_groups",
        "episode/J",
        "episode/K",
        "episode/K_zero",
    )
