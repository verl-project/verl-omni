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
"""Deterministic CPU replays for the reporting-only diffusion watchdog."""

from dataclasses import FrozenInstanceError

import pytest

from verl_omni.trainer.diffusion.training_watchdog import TrainingWatchdog


def _reward(*, zero=1.0, std=0.0, group=2.0, grad=0.0):
    return {
        "critic/rewards/zero_std_ratio": zero,
        "critic/rewards/std_mean": std,
        "critic/rewards/group_size": group,
        "actor/grad_norm": grad,
    }


def _replay(watchdog, rows):
    return [watchdog.observe(step, metrics) for step, metrics in rows]


def test_joint_reward_and_zero_gradient_requires_complete_window():
    watchdog = TrainingWatchdog("flow_grpo", window_size=3, warmup_steps=0)
    rows = _replay(watchdog, [(step, _reward()) for step in range(4)])
    assert [len(alerts) for alerts, _ in rows] == [0, 0, 1, 0]
    alert = rows[2][0][0]
    assert (alert.rule, alert.step, alert.first_step, alert.last_step, alert.observations) == (
        "reward_zero_grad",
        2,
        0,
        2,
        3,
    )
    assert {key: alert.evidence[key] for key in _reward()} == _reward()
    assert alert.evidence["actor/grad_norm/window_min"] == 0.0
    assert alert.evidence["actor/grad_norm/window_max"] == 0.0
    assert rows[2][1]["watchdog/coverage/reward_zero_grad"] == 1.0
    assert rows[2][1]["watchdog/alerts"] == 1.0


@pytest.mark.parametrize(
    "metrics",
    [
        _reward(std=1e-3),  # Healthy reward spread.
        _reward(group=1),  # A singleton group cannot establish collapse.
        _reward(zero=0.5, group=2.5),  # Mixed singleton groups are not all zero-std.
        _reward(grad=0.01),  # A reward plateau alone is insufficient.
        _reward(zero=0.0),
    ],
)
def test_reward_negatives_never_alert(metrics):
    watchdog = TrainingWatchdog("grpo_guard", window_size=2, warmup_steps=0)
    rows = _replay(watchdog, [(step, metrics) for step in range(20)])
    assert sum(len(alerts) for alerts, _ in rows) == 0
    assert all(result["watchdog/coverage/reward_zero_grad"] == 1.0 for _, result in rows)


def test_single_zero_gradient_and_missing_metric_break_window():
    watchdog = TrainingWatchdog("dance_grpo", window_size=3, warmup_steps=0)
    rows = _replay(
        watchdog,
        [
            (0, _reward()),
            (1, _reward(grad=0.1)),
            (2, _reward()),
            (3, {"actor/grad_norm": 0.0}),
            (4, _reward()),
            (5, _reward()),
            (6, _reward()),
        ],
    )
    assert [len(alerts) for alerts, _ in rows] == [0, 0, 0, 0, 0, 0, 1]
    assert rows[3][1]["watchdog/coverage/reward_zero_grad"] == 0.0
    assert rows[6][0][0].first_step == 4


@pytest.mark.parametrize("mode", ["flow_grpo", "dance_grpo", "flow_dppo", "grpo_guard"])
def test_ratio_rule_supported_modes_and_bounds(mode):
    watchdog = TrainingWatchdog(mode, window_size=2, warmup_steps=0)
    rows = _replay(
        watchdog,
        [
            (0, {"actor/ratio_mean": 0.5}),
            (1, {"actor/ratio_mean": 2.0}),
            (2, {"actor/ratio_mean": 0.1}),
            (3, {"actor/ratio_mean": 2.1}),
        ],
    )
    assert [len(alerts) for alerts, _ in rows] == [0, 0, 0, 1]
    assert rows[-1][0][0].rule == "ratio_out_of_bounds"
    assert (rows[-1][0][0].first_step, rows[-1][0][0].last_step) == (2, 3)


def test_unknown_loss_suppresses_reward_and_ratio_but_can_check_staleness():
    watchdog = TrainingWatchdog("flow_nft", window_size=2, warmup_steps=0, max_staleness=2)
    metrics = {**_reward(), "actor/ratio_mean": 10.0, "training/off_policy/trajectory_staleness_worst/max": 3.0}
    rows = _replay(watchdog, [(0, metrics), (1, metrics)])
    assert [alert.rule for alert in rows[1][0]] == ["trajectory_staleness"]
    assert rows[1][1]["watchdog/coverage/reward_zero_grad"] == 0.0
    assert rows[1][1]["watchdog/coverage/ratio_out_of_bounds"] == 0.0
    assert rows[1][1]["watchdog/coverage/trajectory_staleness"] == 1.0


def test_staleness_is_optional_and_measured_in_versions():
    metrics = {"training/off_policy/trajectory_staleness_worst/max": 2.1}
    disabled = TrainingWatchdog("unknown", window_size=1, warmup_steps=0)
    assert disabled.observe(0, metrics)[0] == []
    assert disabled.observe(1, metrics)[1]["watchdog/coverage/trajectory_staleness"] == 0.0
    enabled = TrainingWatchdog("unknown", window_size=1, warmup_steps=0, max_staleness=2)
    assert enabled.observe(0, metrics)[0][0].evidence == metrics


def test_nonfinite_gradient_is_immediate_even_during_warmup_and_latched():
    watchdog = TrainingWatchdog("unknown", warmup_steps=5, cooldown_steps=0)
    rows = _replay(
        watchdog,
        [
            (100, {"actor/grad_norm": float("nan")}),
            (101, {"actor/grad_norm": float("inf")}),
            (102, {}),
            (103, {"actor/grad_norm": float("-inf")}),
            (104, {"actor/grad_norm": 1.0}),
            (105, {"actor/grad_norm": float("nan")}),
        ],
    )
    assert [len(alerts) for alerts, _ in rows] == [1, 0, 0, 0, 0, 1]
    assert rows[0][0][0].rule == "nonfinite_grad"
    assert rows[0][1]["watchdog/warmup_remaining"] == 4.0
    assert rows[2][1]["watchdog/coverage/nonfinite_grad"] == 0.0


def test_warmup_uses_accepted_count_and_reset_handles_high_global_step():
    watchdog = TrainingWatchdog("flow_grpo", window_size=2, warmup_steps=2)
    faulty = {"actor/ratio_mean": 9.0}
    rows = _replay(watchdog, [(50, faulty), (51, faulty), (52, faulty), (53, faulty)])
    assert [len(alerts) for alerts, _ in rows] == [0, 0, 0, 1]
    assert [data["watchdog/warmup_remaining"] for _, data in rows] == [1.0, 0.0, 0.0, 0.0]
    watchdog.reset()
    rows = _replay(watchdog, [(500, faulty), (501, faulty), (502, faulty), (503, faulty)])
    assert [len(alerts) for alerts, _ in rows] == [0, 0, 0, 1]
    assert rows[-1][0][0].first_step == 502


def test_gaps_duplicates_and_out_of_order_do_not_create_evidence():
    watchdog = TrainingWatchdog("flow_dppo", window_size=2, warmup_steps=0)
    faulty = {"actor/ratio_mean": 9.0}
    assert watchdog.observe(10, faulty)[0] == []
    for step in (10, 9):
        alerts, data = watchdog.observe(step, faulty)
        assert alerts == []
        assert data["watchdog/ignored_step"] == 1.0
    assert watchdog.observe(12, faulty)[0] == []  # Step 11 is absent.
    alert = watchdog.observe(13, faulty)[0][0]
    assert (alert.first_step, alert.last_step, alert.observations) == (12, 13, 2)


def test_missing_after_alert_does_not_rearm_but_known_healthy_does():
    watchdog = TrainingWatchdog("flow_grpo", window_size=2, warmup_steps=0, cooldown_steps=3)
    faulty = {"actor/ratio_mean": 9.0}
    rows = _replay(
        watchdog,
        [
            (0, faulty),
            (1, faulty),
            (2, {}),
            (3, faulty),
            (4, faulty),
            (5, {"actor/ratio_mean": 1.0}),
            (6, faulty),
            (7, faulty),
        ],
    )
    assert [len(alerts) for alerts, _ in rows] == [0, 1, 0, 0, 0, 0, 0, 1]
    assert rows[-1][0][0].first_step == 6


def test_cooldown_is_accepted_observations_not_step_distance():
    watchdog = TrainingWatchdog("flow_grpo", window_size=1, warmup_steps=0, cooldown_steps=4)
    faulty = {"actor/ratio_mean": 9.0}
    rows = _replay(watchdog, [(1, faulty), (2, {"actor/ratio_mean": 1.0}), (100, faulty), (101, faulty), (102, faulty)])
    assert [len(alerts) for alerts, _ in rows] == [1, 0, 0, 0, 1]


@pytest.mark.parametrize("bad", [None, True, "1", object(), float("nan"), float("inf"), -1.0])
def test_invalid_metric_never_coerced_or_counted(bad):
    watchdog = TrainingWatchdog("flow_grpo", window_size=1, warmup_steps=0, max_staleness=0)
    metrics = {**_reward(), "actor/ratio_mean": bad, "training/off_policy/trajectory_staleness_worst/max": bad}
    metrics["critic/rewards/std_mean"] = bad
    _, data = watchdog.observe(0, metrics)
    assert data["watchdog/coverage/reward_zero_grad"] == 0.0
    assert data["watchdog/coverage/ratio_out_of_bounds"] == 0.0
    assert data["watchdog/coverage/trajectory_staleness"] == 0.0


def test_tensor_like_input_is_never_converted():
    class Bomb:
        def __float__(self):
            raise AssertionError("non-scalar conversion")

    watchdog = TrainingWatchdog("flow_grpo", window_size=1, warmup_steps=0)
    alerts, data = watchdog.observe(0, {"actor/ratio_mean": Bomb(), "actor/grad_norm": Bomb()})
    assert alerts == []
    assert data["watchdog/coverage/nonfinite_grad"] == 0.0
    assert data["watchdog/coverage/ratio_out_of_bounds"] == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_size": 0},
        {"window_size": True},
        {"window_size": 1.5},
        {"warmup_steps": -1},
        {"warmup_steps": True},
        {"cooldown_steps": -1},
        {"cooldown_steps": 1.0},
        {"reward_std_max": float("nan")},
        {"grad_norm_max": -1},
        {"ratio_lower": 0},
        {"ratio_lower": 1},
        {"ratio_upper": 1},
        {"ratio_upper": float("inf")},
        {"max_staleness": True},
        {"max_staleness": float("nan")},
        {"max_staleness": -1},
        {"loss_mode": None},
    ],
)
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        TrainingWatchdog("flow_grpo", **kwargs) if "loss_mode" not in kwargs else TrainingWatchdog(**kwargs)


def test_evidence_is_snapshot_and_alert_is_immutable():
    watchdog = TrainingWatchdog("flow_grpo", window_size=1, warmup_steps=0)
    metrics = {"actor/ratio_mean": 5.0}
    alert = watchdog.observe(0, metrics)[0][0]
    metrics["actor/ratio_mean"] = 6.0
    assert alert.evidence["actor/ratio_mean"] == 5.0
    with pytest.raises(TypeError):
        alert.evidence["actor/ratio_mean"] = 7.0
    with pytest.raises(FrozenInstanceError):
        alert.step = 3


def test_replay_is_deterministic_and_state_is_bounded():
    rows = [
        (
            step,
            {
                "actor/ratio_mean": 9.0 if step % 11 < 6 else 1.0,
                "training/off_policy/trajectory_staleness_worst/max": float(step % 7),
            },
        )
        for step in range(10_000)
    ]

    def run():
        watchdog = TrainingWatchdog("flow_grpo", window_size=3, warmup_steps=5, cooldown_steps=20, max_staleness=4)
        results = _replay(watchdog, rows)
        assert len(watchdog._windows) == 3
        assert sum(len(window) for window in watchdog._windows.values()) <= 9
        assert len(watchdog._active) == len(watchdog._last_alert_observation) == 4
        return [
            (tuple((alert.rule, alert.first_step, alert.last_step) for alert in alerts), data)
            for alerts, data in results
        ]

    assert run() == run()
    healthy = TrainingWatchdog("flow_grpo", window_size=3, warmup_steps=5)
    assert (
        sum(len(alerts) for alerts, _ in _replay(healthy, [(step, {"actor/ratio_mean": 1.0}) for step in range(100)]))
        == 0
    )
