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
"""Exercise profiling through the actual V1 fit loop with mocked compute/RPCs."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from verl_omni.trainer.diffusion.performance_watchdog import StepTimePerformanceWatchdog
from verl_omni.trainer.diffusion.v1 import trainer_base, trainer_separate_async, trainer_sync
from verl_omni.trainer.diffusion.v1.trainer_separate_async import PolicyGradientDiffusionTrainerV1SeparateAsync
from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync


def make_trainer(monkeypatch, *, mode="sync", continuous=False, fail_at=None, resumed_step=0, val_only=False):
    events = []
    cls = PolicyGradientDiffusionTrainerV1Sync if mode == "sync" else PolicyGradientDiffusionTrainerV1SeparateAsync
    trainer = object.__new__(cls)
    trainer.config = OmegaConf.create(
        {
            "global_profiler": {"steps": [1, 2, 4], "profile_continuous_steps": continuous},
            "actor_rollout_ref": {"actor": {"profiler": {"enable": True}}, "rollout": {"profiler": {"enable": True}}},
            "trainer": {
                "project_name": "test",
                "experiment_name": "test",
                "logger": [],
                "val_before_train": val_only,
                "val_only": val_only,
                "total_epochs": 1,
                "save_freq": -1,
                "test_freq": -1,
            },
        }
    )
    trainer.use_reference_policy = False
    trainer.use_critic = False
    trainer.sync_compatible = False
    trainer._training_watchdog = None
    trainer._performance_watchdog = None
    trainer.global_steps = resumed_step
    trainer.steps_per_epoch = 4
    trainer.total_training_steps = 4
    trainer.hybrid_rollout_config = SimpleNamespace(enable_switch=False)
    trainer.profile_stop_calls = []

    def backend(role):
        def stop_profile(**kwargs):
            trainer.profile_stop_calls.append((role, kwargs))
            events.append((role, "stop"))

        return SimpleNamespace(
            start_profile=lambda **kwargs: events.append((role, "start", kwargs)),
            stop_profile=stop_profile,
        )

    trainer.actor_rollout_wg = backend("actor")
    trainer.llm_server_manager = backend("rollout")
    trainer.standalone_server_manager = backend("standalone")
    trainer.checkpoint_manager = SimpleNamespace(sleep_replicas=lambda: events.append(("sleep", trainer.global_steps)))
    trainer._reissue_inflight_prompts = lambda: events.append(("reissue", trainer.global_steps))
    trainer.on_train_begin = lambda: events.append(("warmup", trainer.global_steps))
    trainer.on_step_begin = lambda: None
    trainer.checkpoint_manager.update_weights = Mock(return_value={})
    trainer.standalone_checkpoint_manager = SimpleNamespace(update_weights=Mock(return_value={}))
    trainer.on_validate_begin = lambda: None
    trainer.on_validate_end = lambda: None
    trainer._validate = lambda: {"validation": 1}
    trainer._compute_metrics = lambda *args: None
    trainer._shutdown_dump_executor = lambda: None
    trainer._shutdown_dataloaders = lambda: None

    def step(metrics, timing):
        events.append(("feed", trainer.global_steps))
        if trainer.global_steps == fail_at:
            raise RuntimeError("compute failed")
        if mode == "sync":
            trainer.on_sample_end()
        return SimpleNamespace(keys=[], partition_id="train")

    trainer.step = step
    monkeypatch.setattr(trainer_base, "SkipManager", Mock())
    monkeypatch.setattr(trainer_base, "init_skip_manager", Mock())
    monkeypatch.setattr(trainer_base, "Tracking", Mock())
    monkeypatch.setattr(trainer_base, "ValidationGenerationsLogger", Mock())
    monkeypatch.setattr(trainer_base, "tq", Mock())
    monkeypatch.setattr(trainer_base, "tqdm", Mock())
    monkeypatch.setattr(trainer_base, "marked_timer", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(trainer_sync, "marked_timer", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(trainer_separate_async, "marked_timer", lambda *args, **kwargs: nullcontext())
    return trainer, events


@pytest.mark.parametrize("mode", ["sync", "separate_async"])
def test_fit_selected_steps_and_mode_hooks(monkeypatch, mode):
    trainer, events = make_trainer(monkeypatch, mode=mode)
    trainer.fit(Mock())
    assert [e[2]["profile_step"] for e in events if e[:2] == ("actor", "start")] == [1, 2, 4]
    role = "rollout" if mode == "sync" else "standalone"
    assert events.count((role, "start", {})) == 3
    assert events.count((role, "stop")) == 3
    assert events.index(("warmup", 1)) < events.index((role, "start", {}))
    if mode == "sync":
        for step in (1, 2, 4):
            assert events[events.index(("sleep", step)) + 1] == ("rollout", "stop")
    assert [kwargs["run_command"] for role, kwargs in trainer.profile_stop_calls if role == "actor"] == [
        False,
        False,
        True,
    ]


@pytest.mark.parametrize("mode", ["sync", "separate_async"])
def test_fit_resumes_selected_step(monkeypatch, mode):
    trainer, events = make_trainer(monkeypatch, mode=mode, resumed_step=1)
    trainer.fit(Mock())
    assert [e[2]["profile_step"] for e in events if e[:2] == ("actor", "start")] == [2, 4]


def test_validation_only_does_not_start_profiler(monkeypatch):
    trainer, events = make_trainer(monkeypatch, val_only=True)
    trainer.fit(Mock())
    assert events == []


def test_async_without_hybrid_only_profiles_standalone(monkeypatch):
    trainer, events = make_trainer(monkeypatch, mode="separate_async")
    trainer.fit(Mock())
    assert not any(e[0] == "rollout" for e in events)
    assert events.count(("standalone", "start", {})) == 3


def test_performance_watchdog_profiles_only_the_next_step(monkeypatch, caplog):
    trainer, events = make_trainer(monkeypatch)
    trainer.config.global_profiler.steps = []
    trainer.config.global_profiler.profile_continuous_steps = False
    trainer.config.global_profiler.tool = "torch"
    trainer.config.global_profiler.save_path = "outputs/profile"
    trainer.config.trainer.v1 = {"performance_watchdog": {"regression_ratio": 1.5}}
    trainer._performance_watchdog = StepTimePerformanceWatchdog(
        warmup_steps=0,
        baseline_window_size=2,
        minimum_observations=1,
        regression_window_size=1,
        regression_ratio=1.5,
        cooldown_steps=10,
    )
    trainer._performance_profile_step = None
    trainer._pending_performance_profile_report = None
    trainer._completed_performance_profile_report = None
    trainer._profiled_step_this_iteration = False
    trainer._last_performance_insufficient_reason = None
    trainer._performance_profile_tool = "torch"
    trainer._performance_profile_roles = ("actor", "rollout")
    trainer.parameter_sync_step = 1
    trainer._performance_watchdog_workload = {"train_batch_size": 4, "parameter_sync_step": 1}

    def compute_metrics(_batch, metrics, _timing, step, _epoch):
        step_time = {1: 1.0, 2: 1.0, 3: 2.0, 4: 30.0}[step]
        metrics.update(
            {
                "perf/time_per_step": step_time,
                "perf/total_num_images": 4,
                "timing_s/gen": step_time / 2,
                "training/global_step": step,
            }
        )

    monkeypatch.setattr(trainer, "_compute_metrics", compute_metrics)
    with caplog.at_level("WARNING", logger=trainer_base.logger.name):
        trainer.fit(Mock())

    actor_profiles = [event for event in events if event[:2] == ("actor", "start")]
    assert [event[2]["profile_step"] for event in actor_profiles] == [4]
    assert len([event for event in events if event[:2] == ("actor", "stop")]) == 1
    assert events.count(("rollout", "start", {})) == 1
    assert events.count(("rollout", "stop")) == 1
    assert trainer.profile_stop_calls == [("actor", {"run_command": True})]
    assert "status=captured" in caplog.text
    assert "capture_step=4" in caplog.text
    assert "artifact_ref=outputs/profile" in caplog.text
    assert "excludes checkpoint" in caplog.text
