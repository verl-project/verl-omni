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
"""CPU replay tests for bounded diffusion step-time regression detection."""

import pytest

from verl_omni.trainer.diffusion.performance_watchdog import (
    StepTimePerformanceWatchdog,
    validate_performance_watchdog_config,
)

_WORKLOAD = {
    "train_batch_size": 8,
    "rollout_n": 2,
    "height": 256,
    "width": 256,
    "num_inference_steps": 4,
    "parameter_sync_step": 1,
    "nnodes": 1,
    "n_gpus_per_node": 2,
}


def _metrics(step_time_s, *, images=16, gen_s=0.5, update_s=0.4):
    return {
        "perf/time_per_step": step_time_s,
        "perf/total_num_images": images,
        "timing_s/step": step_time_s,
        "timing_s/gen": gen_s,
        "timing_s/update_actor": update_s,
    }


def _watchdog(**kwargs):
    return StepTimePerformanceWatchdog(
        warmup_steps=kwargs.pop("warmup_steps", 0),
        baseline_window_size=kwargs.pop("baseline_window_size", 5),
        minimum_observations=kwargs.pop("minimum_observations", 3),
        regression_window_size=kwargs.pop("regression_window_size", 3),
        regression_ratio=kwargs.pop("regression_ratio", 1.5),
        cooldown_steps=kwargs.pop("cooldown_steps", 10),
        max_phase_deltas=kwargs.pop("max_phase_deltas", 3),
        **kwargs,
    )


def test_warmup_and_minimum_baseline_observations():
    watchdog = _watchdog(warmup_steps=2, baseline_window_size=4, minimum_observations=2)
    first = watchdog.observe(1, _metrics(1.0), _WORKLOAD)
    second = watchdog.observe(2, _metrics(1.0), _WORKLOAD)
    assert first.reason == "warmup"
    assert second.reason == "warmup"

    results = [watchdog.observe(step, _metrics(1.0), _WORKLOAD) for step in (3, 4, 5, 6, 7)]
    assert all(result.alert is None for result in results)
    assert any(result.reason == "collecting_regression_window" for result in results)
    assert any(result.reason == "collecting_baseline" for result in results)
    assert results[-1].state == "observing"


def test_noisy_steady_series_and_single_outlier_do_not_trigger():
    watchdog = _watchdog()
    times = [1.0, 0.94, 1.08, 0.98, 1.03, 0.96, 1.02, 9.0, 1.01, 1.0]
    results = [watchdog.observe(step, _metrics(value), _WORKLOAD) for step, value in enumerate(times, 1)]
    assert all(result.alert is None for result in results)


def test_sustained_regression_emits_compact_phase_delta_report():
    watchdog = _watchdog()
    for step, value in enumerate([1.0, 1.0, 1.0, 1.0, 1.0], 1):
        result = watchdog.observe(step, _metrics(value), _WORKLOAD)
        assert result.alert is None

    for step in (6, 7, 8):
        result = watchdog.observe(step, _metrics(2.0, gen_s=1.6, update_s=0.7), _WORKLOAD)

    alert = result.alert
    assert alert is not None
    assert (alert.first_step, alert.last_step) == (6, 8)
    assert alert.baseline_observations >= 3
    assert alert.current_observations == 3
    assert alert.baseline_median_s == pytest.approx(1.0)
    assert alert.current_median_s == pytest.approx(2.0)
    assert alert.slowdown_ratio == pytest.approx(2.0)
    assert len(alert.workload_fingerprint) == 12
    assert [delta.name for delta in alert.phase_deltas] == ["gen", "update_actor"]
    assert alert.phase_deltas[0].delta_s == pytest.approx(1.1)


def test_workload_change_resets_comparison_and_requires_new_baseline():
    watchdog = _watchdog(
        baseline_window_size=4,
        minimum_observations=2,
        regression_window_size=1,
    )
    for step in range(1, 7):
        result = watchdog.observe(step, _metrics(1.0), _WORKLOAD)
    assert result.state == "observing"

    changed_workload = {**_WORKLOAD, "width": 512}
    changed = watchdog.observe(7, _metrics(4.0), changed_workload)
    assert changed.insufficient_evidence
    assert changed.reason == "workload_changed"
    assert changed.alert is None

    first_new = watchdog.observe(8, _metrics(4.0), changed_workload)
    second_new = watchdog.observe(9, _metrics(4.0), changed_workload)
    assert first_new.alert is None
    assert second_new.alert is None
    assert first_new.reason == "collecting_baseline"
    assert second_new.state == "observing"

    third_new = watchdog.observe(10, _metrics(4.0), changed_workload)
    assert third_new.alert is None
    assert third_new.state == "observing"


@pytest.mark.parametrize(
    ("metrics", "workload", "reason"),
    [
        ({"perf/total_num_images": 8}, _WORKLOAD, "missing_or_invalid_step_metrics"),
        (_metrics(1.0, images=0), _WORKLOAD, "missing_or_invalid_step_metrics"),
        (_metrics(1.0), {**_WORKLOAD, "height": "prompt-shaped"}, "invalid_workload_signature"),
    ],
)
def test_missing_or_incompatible_evidence_is_explicit(metrics, workload, reason):
    result = _watchdog().observe(1, metrics, workload)
    assert result.insufficient_evidence
    assert result.reason == reason
    assert result.alert is None


def test_step_gap_resets_history():
    watchdog = _watchdog(regression_window_size=1, minimum_observations=1)
    watchdog.observe(1, _metrics(1.0), _WORKLOAD)
    result = watchdog.observe(3, _metrics(10.0), _WORKLOAD)
    assert result.insufficient_evidence
    assert result.reason == "step_gap"
    assert result.alert is None


def test_one_alert_per_episode_and_cooldown_between_episodes():
    watchdog = _watchdog(
        baseline_window_size=3,
        minimum_observations=1,
        regression_window_size=2,
        regression_ratio=1.75,
        cooldown_steps=6,
    )
    for step in range(1, 7):
        watchdog.observe(step, _metrics(1.0), _WORKLOAD)
    first_high = watchdog.observe(7, _metrics(2.0), _WORKLOAD)
    first_alert = watchdog.observe(8, _metrics(2.0), _WORKLOAD)
    assert first_high.alert is None
    assert first_alert.alert is not None

    still_slow = watchdog.observe(9, _metrics(2.0), _WORKLOAD)
    assert still_slow.alert is None
    assert still_slow.state == "cooldown"

    # A recovered episode inside the cooldown is suppressed.
    watchdog.observe(10, _metrics(1.0), _WORKLOAD)
    watchdog.observe(11, _metrics(1.0), _WORKLOAD)
    first_suppressed_high = watchdog.observe(12, _metrics(2.0), _WORKLOAD)
    suppressed = watchdog.observe(13, _metrics(2.0), _WORKLOAD)
    assert first_suppressed_high.alert is None
    assert suppressed.alert is None
    assert suppressed.state == "cooldown"

    # Once the episode recovers and the cooldown elapses, a later episode alerts.
    watchdog.observe(14, _metrics(1.0), _WORKLOAD)
    watchdog.observe(15, _metrics(1.0), _WORKLOAD)
    watchdog.observe(16, _metrics(2.0), _WORKLOAD)
    recovered_alert = watchdog.observe(17, _metrics(2.0), _WORKLOAD)
    assert recovered_alert.alert is not None


@pytest.fixture
def valid_profile_config():
    return {
        "global_profiler": {
            "tool": "torch",
            "profile_continuous_steps": False,
            "save_path": "outputs/profile",
        },
        "actor_rollout_ref": {
            "actor": {"profiler": {"enable": True, "tool": "torch"}},
            "ref": {"profiler": {"enable": False, "tool": "torch"}},
            "rollout": {"profiler": {"enable": False, "tool": "torch"}},
        },
    }


def test_profile_config_returns_tool_and_role_selection(valid_profile_config):
    assert validate_performance_watchdog_config(
        valid_profile_config,
        trainer_mode="sync",
        use_reference_policy=False,
    ) == ("torch", ("actor",))


@pytest.mark.parametrize(
    ("mutate", "trainer_mode", "use_reference_policy", "message"),
    [
        (lambda config: None, "separate_async", False, "only with trainer_mode='sync'"),
        (lambda config: config["global_profiler"].update(tool="nsys"), "sync", False, "does not support"),
        (
            lambda config: config["global_profiler"].update(profile_continuous_steps=True),
            "sync",
            False,
            "requires global_profiler.profile_continuous_steps=false",
        ),
        (
            lambda config: config["actor_rollout_ref"]["actor"]["profiler"].update(enable=False),
            "sync",
            False,
            "requires at least one active role profiler",
        ),
    ],
)
def test_invalid_profile_configuration_is_rejected(
    valid_profile_config, mutate, trainer_mode, use_reference_policy, message
):
    mutate(valid_profile_config)
    with pytest.raises(ValueError, match=message):
        validate_performance_watchdog_config(
            valid_profile_config,
            trainer_mode=trainer_mode,
            use_reference_policy=use_reference_policy,
        )
