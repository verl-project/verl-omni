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
from unittest.mock import Mock, call

import pytest
from omegaconf import OmegaConf

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
    trainer._init_profile_state()
    trainer.use_teacher_policy = False
    trainer.ref_policy_wg = None
    trainer.reward_loop_manager = None
    trainer.gpu_metrics_profiler = None
    trainer.sync_compatible = False
    trainer._training_watchdog = None
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


def test_fit_profiles_each_distinct_worker_group(monkeypatch):
    trainer, _ = make_trainer(monkeypatch)
    actor = trainer.actor_rollout_wg = Mock(wraps=trainer.actor_rollout_wg)
    ref = trainer.ref_policy_wg = Mock()
    teacher = Mock()
    trainer.use_reference_policy = True
    trainer.use_teacher_policy = True
    trainer.teacher_model_manager = SimpleNamespace(
        teacher_wg={"actor_alias": actor, "ref_alias": ref, "teacher1": teacher, "teacher2": teacher}
    )

    trainer.fit(Mock())

    for worker, role in ((actor, "train"), (ref, "ref"), (teacher, "teacher")):
        assert worker.mock_calls == [
            call.start_profile(role=role, profile_step=1),
            call.stop_profile(run_command=False),
            call.start_profile(role=role, profile_step=2),
            call.stop_profile(run_command=False),
            call.start_profile(role=role, profile_step=4),
            call.stop_profile(run_command=True),
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


@pytest.mark.parametrize("continuous", [False, True])
def test_failed_step_closes_all_started_targets(monkeypatch, continuous):
    trainer, events = make_trainer(monkeypatch, continuous=continuous, fail_at=1)
    with pytest.raises(RuntimeError, match="compute failed"):
        trainer.fit(Mock())
    assert events.count(("rollout", "start", {})) == events.count(("rollout", "stop")) == 1
    assert events.count(("actor", "stop")) == 1
    assert not trainer._active_profile_targets
    assert not trainer._active_rollout_profile_targets
    assert not trainer._profile_window_active


def test_cleanup_failure_preserves_training_error(monkeypatch):
    trainer, events = make_trainer(monkeypatch, continuous=True)
    training_error = ValueError("compute failed")
    trainer.step = Mock(side_effect=training_error)
    trainer.actor_rollout_wg.stop_profile = Mock(side_effect=RuntimeError("cleanup failed"))

    with pytest.raises(ValueError) as exc:
        trainer.fit(Mock())

    assert exc.value is training_error
    trainer.actor_rollout_wg.stop_profile.assert_called_once_with(run_command=False)
    assert events.count(("rollout", "stop")) == 1


@pytest.mark.parametrize(("failing_target", "stop_kwargs"), [("ref", {"run_command": False}), ("rollout", {})])
def test_start_failure_cleans_up_started_and_failing_targets(monkeypatch, failing_target, stop_kwargs):
    trainer, events = make_trainer(monkeypatch)
    trainer.use_reference_policy = True
    ref = trainer.ref_policy_wg = Mock()
    trainer.global_steps = 1
    trainer.prev_step_profile = False
    trainer.curr_step_profile = True
    target = {"ref": ref, "rollout": trainer.llm_server_manager}[failing_target]
    start_error = RuntimeError("start failed")
    target.start_profile = Mock(side_effect=start_error)
    target.stop_profile = Mock()

    with pytest.raises(RuntimeError) as exc:
        trainer._on_profiling_step_start()

    assert exc.value is start_error
    assert events.count(("actor", "stop")) == 1
    ref.stop_profile.assert_called_once_with(run_command=False)
    target.stop_profile.assert_called_once_with(**stop_kwargs)
    assert not trainer._active_profile_targets
    assert not trainer._active_rollout_profile_targets
    assert not trainer._profile_window_active


def test_stop_failure_still_closes_remaining_targets(monkeypatch):
    trainer, events = make_trainer(monkeypatch, continuous=True)
    trainer.config.global_profiler.steps = [1]
    trainer.use_reference_policy = True
    stop_error = RuntimeError("stop failed")
    trainer.ref_policy_wg = Mock()
    trainer.ref_policy_wg.stop_profile.side_effect = stop_error
    trainer.global_steps = 1
    trainer.prev_step_profile = False
    trainer.curr_step_profile = True
    trainer._on_profiling_step_start()

    with pytest.raises(RuntimeError) as exc:
        trainer._on_profiling_step_end()

    assert exc.value is stop_error
    trainer.ref_policy_wg.stop_profile.assert_called_once_with(run_command=True)
    assert events.count(("actor", "stop")) == 1
    assert events.count(("rollout", "stop")) == 1
    assert not trainer._active_profile_targets
    assert not trainer._active_rollout_profile_targets
    assert not trainer._profile_window_active


def test_failed_step_end_closes_continuous_window(monkeypatch):
    trainer, events = make_trainer(monkeypatch, continuous=True)
    trainer.on_step_end = Mock(side_effect=RuntimeError("weight sync failed"))
    with pytest.raises(RuntimeError, match="weight sync failed"):
        trainer.fit(Mock())
    assert events.count(("rollout", "start", {})) == events.count(("rollout", "stop")) == 1
    assert events.count(("actor", "stop")) == 1
    assert not trainer._profile_window_active
