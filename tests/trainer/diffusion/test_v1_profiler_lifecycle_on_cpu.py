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

from verl_omni.trainer.diffusion.v1 import trainer_base
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
    trainer.global_steps = resumed_step
    trainer.steps_per_epoch = 4
    trainer.total_training_steps = 4
    trainer.hybrid_rollout_config = SimpleNamespace(enable_switch=True)
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
    trainer.on_step_end = lambda: None
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
    monkeypatch.setattr(trainer_base, "Tracking", Mock())
    monkeypatch.setattr(trainer_base, "ValidationGenerationsLogger", Mock())
    monkeypatch.setattr(trainer_base, "tq", Mock())
    monkeypatch.setattr(trainer_base, "tqdm", Mock())
    monkeypatch.setattr(trainer_base, "marked_timer", lambda *args, **kwargs: nullcontext())
    return trainer, events


@pytest.mark.parametrize("mode", ["sync", "separate_async"])
@pytest.mark.parametrize("continuous", [False, True])
def test_fit_selected_steps_and_cleanup(monkeypatch, mode, continuous):
    trainer, events = make_trainer(monkeypatch, mode=mode, continuous=continuous)
    trainer.fit(Mock())
    expected_steps = [1, 4] if continuous else [1, 2, 4]
    assert [e[2]["profile_step"] for e in events if e[:2] == ("actor", "start")] == expected_steps
    assert events.count(("actor", "stop")) == len(expected_steps)
    assert events.index(("rollout", "start", {})) < events.index(("reissue", 1))
    assert events.index(("rollout", "start", {})) < events.index(("warmup", 1))
    if mode == "sync":
        for step in (1, 2, 4):
            assert events[events.index(("sleep", step)) - 1] == ("rollout", "stop")
    else:
        assert events.count(("standalone", "start", {})) == len(expected_steps)
        assert events.count(("standalone", "stop")) == len(expected_steps)
    assert events[-1] == ("actor", "stop")


@pytest.mark.parametrize("mode", ["sync", "separate_async"])
def test_fit_exception_flushes_profile(monkeypatch, mode):
    trainer, events = make_trainer(monkeypatch, mode=mode, continuous=True, fail_at=2)
    with pytest.raises(RuntimeError, match="compute failed"):
        trainer.fit(Mock())
    assert events[-1] == ("actor", "stop")
    assert events.count(("actor", "stop")) == 1
    if mode == "separate_async":
        assert events.count(("standalone", "stop")) == 1


def test_fit_resumes_inside_profile_window(monkeypatch):
    trainer, events = make_trainer(monkeypatch, continuous=True, resumed_step=1)
    trainer.fit(Mock())
    assert events[0] == ("actor", "start", {"role": "train", "profile_step": 2})


def test_validation_only_does_not_start_profiler(monkeypatch):
    trainer, events = make_trainer(monkeypatch, val_only=True)
    trainer.fit(Mock())
    assert events == []


def test_async_without_hybrid_only_profiles_standalone(monkeypatch):
    trainer, events = make_trainer(monkeypatch, mode="separate_async")
    trainer.hybrid_rollout_config.enable_switch = False
    trainer.fit(Mock())
    assert not any(e[0] == "rollout" for e in events)
    assert events.count(("standalone", "start", {})) == 3


@pytest.mark.parametrize("mode", ["sync", "separate_async"])
@pytest.mark.parametrize("total_steps,steps_per_epoch", [(3, 4), (8, 3)])
def test_fit_finish_hook_respects_step_and_epoch_limits(monkeypatch, mode, total_steps, steps_per_epoch):
    trainer, events = make_trainer(monkeypatch, mode=mode)
    trainer.config.global_profiler.steps = [1, 5]
    trainer.total_training_steps = total_steps
    trainer.steps_per_epoch = steps_per_epoch
    trainer.fit(Mock())
    assert [kwargs for role, kwargs in trainer.profile_stop_calls if role == "actor"] == [{"run_command": True}]
    assert [event[2]["profile_step"] for event in events if event[:2] == ("actor", "start")] == [1]
