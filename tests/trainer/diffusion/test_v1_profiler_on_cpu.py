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
"""Profiler RPC contracts without importing Ray, engines, or model registries."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from omegaconf import OmegaConf

# This controller has no engine dependencies. Load it directly so these CPU
# tests also run without verl_omni's package-level model/engine registration.
ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "diffusion_v1_profiling", ROOT / "verl_omni/trainer/diffusion/v1/profiling.py"
)
profiling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profiling)


def config(*, steps=(1, 2, 4), continuous=False, actor=True, rollout=True, tool="torch"):
    return OmegaConf.create(
        {
            "global_profiler": {
                "steps": steps,
                "tool": tool,
                "profile_continuous_steps": continuous,
                "global_tool_config": {
                    "nsys": {
                        "controller_nsight_options": {"capture-range": "cudaProfilerApi"},
                        "worker_nsight_options": {"capture-range": "cudaProfilerApi"},
                    }
                },
            },
            "actor_rollout_ref": {
                "actor": {"profiler": {"enable": actor}},
                "rollout": {"profiler": {"enable": rollout}},
            },
        }
    )


class Recorder:
    def __init__(self, events, name, *, fail_start=False, fail_stop=False):
        self.events = events
        self.name = name
        self.fail_start = fail_start
        self.fail_stop = fail_stop
        self.stop_kwargs = []

    def start_profile(self, **kwargs):
        self.events.append((self.name, "start", kwargs))
        if self.fail_start:
            raise RuntimeError(f"{self.name} start failed")

    def stop_profile(self, **kwargs):
        self.stop_kwargs.append(kwargs)
        self.events.append((self.name, "stop"))
        if self.fail_stop:
            raise RuntimeError(f"{self.name} stop failed")


def make_profiler(cfg, events, *, hybrid=False):
    managers = [Recorder(events, "rollout")]
    if hybrid:
        managers.append(Recorder(events, "hybrid"))
    return profiling.DiffusionV1Profiler(cfg, Recorder(events, "actor"), managers)


@pytest.mark.parametrize("steps", [None, [], [5]])
def test_disabled_or_unselected_step(steps):
    events = []
    profiler = make_profiler(config(steps=steps), events)
    profiler.start_step(1)
    profiler.end_step(1)
    profiler.close()
    assert events == []


@pytest.mark.parametrize("actor,rollout", [(True, False), (False, True), (True, True), (False, False)])
def test_independent_roles(actor, rollout):
    events = []
    profiler = make_profiler(config(actor=actor, rollout=rollout), events)
    profiler.start_step(1)
    profiler.start_step(1)
    profiler.end_step(1)
    profiler.close()
    expected = []
    if actor:
        expected.append(("actor", "start", {"role": "train", "profile_step": 1}))
    if rollout:
        expected += [("rollout", "start", {}), ("rollout", "stop")]
    if actor:
        expected.append(("actor", "stop"))
    assert events == expected


@pytest.mark.parametrize("continuous,expected_starts", [(False, [1, 2, 4]), (True, [1, 4])])
def test_async_windows_and_hybrid(continuous, expected_starts):
    events = []
    profiler = make_profiler(config(continuous=continuous), events, hybrid=True)
    for step in range(1, 5):
        profiler.start_step(step)
        profiler.end_step(step, last_step=step == 4)
    assert [e[2]["profile_step"] for e in events if e[:2] == ("actor", "start")] == expected_starts
    for role in ("actor", "rollout", "hybrid"):
        assert sum(e[:2] == (role, "start") for e in events) == len(expected_starts)
        assert sum(e == (role, "stop") for e in events) == len(expected_starts)


def test_sync_closes_rollout_before_sleep_but_keeps_actor_window():
    events = []
    profiler = make_profiler(config(continuous=True), events)
    for step in (1, 2):
        profiler.start_step(step)
        events.append(("feed", step))
        profiler.stop_rollout()
        events.append(("sleep", step))
        profiler.end_step(step)
    assert [e for e in events if e[:2] == ("actor", "start")] == [
        ("actor", "start", {"role": "train", "profile_step": 1})
    ]
    assert events.count(("rollout", "start", {})) == 2
    for step in (1, 2):
        sleep_index = events.index(("sleep", step))
        assert events[sleep_index - 1] == ("rollout", "stop")
    assert events[-1] == ("actor", "stop")


def test_resume_and_last_step_close_a_continuous_window():
    events = []
    profiler = make_profiler(config(steps=[1, 2, 3], continuous=True), events)
    profiler.start_step(2)  # Resumed after a checkpoint at step 1.
    profiler.end_step(2, last_step=True)  # Total steps can be below the profile range.
    profiler.close()
    assert events[0] == ("actor", "start", {"role": "train", "profile_step": 2})
    assert events[-1] == ("actor", "stop")
    assert len(events) == 4


@pytest.mark.parametrize("fail_stop", [False, True])
def test_partial_start_failure_cleans_all_targets_and_preserves_error(fail_stop):
    events = []
    actor = Recorder(events, "actor", fail_stop=fail_stop)
    rollout = Recorder(events, "rollout", fail_start=True)
    profiler = profiling.DiffusionV1Profiler(config(), actor, [rollout])
    with pytest.raises(RuntimeError, match="rollout start failed"):
        profiler.start_step(1)
    profiler.close()
    assert events[-2:] == [("rollout", "stop"), ("actor", "stop")]


def test_stop_failure_does_not_skip_other_backends():
    events = []
    profiler = profiling.DiffusionV1Profiler(
        config(), Recorder(events, "actor"), [Recorder(events, "rollout", fail_stop=True)]
    )
    profiler.start_step(1)
    with pytest.raises(RuntimeError, match="rollout stop failed"):
        profiler.end_step(1)
    profiler.close()
    assert events[-1] == ("actor", "stop")


def test_controller_stops_after_workers_even_on_failure(monkeypatch):
    events = []
    platform = ModuleType("verl.plugin.platform")
    platform.get_platform = lambda: type(
        "Platform",
        (),
        {
            "profiler_start": lambda self: events.append(("controller", "start")),
            "profiler_stop": lambda self: events.append(("controller", "stop")),
        },
    )()
    monkeypatch.setitem(sys.modules, "verl.plugin.platform", platform)
    profiler = profiling.DiffusionV1Profiler(config(tool="nsys"), Recorder(events, "actor", fail_start=True), [])
    with pytest.raises(RuntimeError, match="actor start failed"):
        profiler.start_step(1)
    assert events[0] == ("controller", "start")
    assert events[-2:] == [("actor", "stop"), ("controller", "stop")]


@pytest.mark.parametrize("tool", ["torch", "npu", "torch_memory", "nsys"])
def test_worker_group_launch_options(tool):
    cfg = config(tool=tool)
    kwargs = profiling.profiler_worker_group_kwargs(cfg)
    assert kwargs["profile_steps"] == [1, 2, 4]
    assert ("worker_nsight_options" in kwargs) == (tool == "nsys")
    if tool == "nsys":
        assert isinstance(kwargs["worker_nsight_options"], dict)
    cfg.global_profiler.steps = []
    assert profiling.profiler_worker_group_kwargs(cfg) == {}


def test_missing_nsight_worker_options():
    cfg = config(tool="nsys")
    del cfg.global_profiler.global_tool_config.nsys.worker_nsight_options
    with pytest.raises(ValueError, match="worker_nsight_options"):
        profiling.profiler_worker_group_kwargs(cfg)


@pytest.mark.parametrize("continuous", [False, True])
def test_finish_hook_only_runs_after_final_window(continuous):
    actor = Recorder([], "actor")
    profiler = profiling.DiffusionV1Profiler(config(continuous=continuous), actor, [])
    for step in range(1, 5):
        profiler.start_step(step)
        profiler.end_step(step)
    profiler.close()
    assert actor.stop_kwargs == ([{"run_command": False}] * (1 if continuous else 2) + [{"run_command": True}])


def test_failure_cleanup_does_not_run_finish_hook():
    actor = Recorder([], "actor")
    profiler = profiling.DiffusionV1Profiler(config(), actor, [])
    profiler.start_step(1)
    profiler.close(suppress_errors=True)
    assert actor.stop_kwargs == [{"run_command": False}]


def test_training_final_step_runs_finish_hook():
    actor = Recorder([], "actor")
    profiler = profiling.DiffusionV1Profiler(config(steps=[1, 2, 3], continuous=True), actor, [])
    profiler.start_step(1)
    profiler.end_step(1, last_step=True)
    assert actor.stop_kwargs == [{"run_command": True}]


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("start_step", [1, 2])
def test_finish_hook_uses_last_reachable_profile_step(continuous, start_step):
    events = []
    actor = Recorder(events, "actor")
    profiler = profiling.DiffusionV1Profiler(
        config(steps=[1, 2, 5], continuous=continuous, tool="npu"), actor, [], total_training_steps=3
    )
    for step in range(start_step, 4):
        profiler.start_step(step)
        profiler.end_step(step, last_step=step == 3)
    profiler.close()
    assert actor.stop_kwargs[-1] == {"run_command": True}
    assert sum(call["run_command"] for call in actor.stop_kwargs) == 1
    assert len(actor.stop_kwargs) == (2 if start_step == 1 and not continuous else 1)


def test_profile_steps_outside_training_do_not_start_or_run_hook():
    actor = Recorder([], "actor")
    profiler = profiling.DiffusionV1Profiler(config(steps=[5], tool="npu"), actor, [], total_training_steps=3)
    for step in range(1, 4):
        profiler.start_step(step)
        profiler.end_step(step, last_step=step == 3)
    profiler.close()
    assert actor.events == []


@pytest.mark.parametrize("engine", ["dp_diffusion", "veomni_diffusion"])
@pytest.mark.parametrize("mode,hybrid", [("sync", False), ("separate_async", False), ("separate_async", True)])
@pytest.mark.parametrize("actor,rollout", [(True, False), (False, True), (True, True)])
def test_compose_npu_discrete_manual_ranks(engine, mode, hybrid, actor, rollout):
    from hydra import compose, initialize_config_dir

    rollout_rank = 0 if mode == "sync" else 8
    with initialize_config_dir(config_dir=str(ROOT / "verl_omni/trainer/config"), version_base=None):
        cfg = compose(
            config_name="diffusion_trainer",
            overrides=[
                f"diffusion/model_engine={engine}",
                "trainer.use_v1=true",
                f"trainer.v1.trainer_mode={mode}",
                f"trainer.v1.separate_async.hybrid_rollout.enable_switch={str(hybrid).lower()}",
                "global_profiler.tool=npu",
                "global_profiler.steps=[1,3]",
                "global_profiler.profile_continuous_steps=false",
                "actor_rollout_ref.actor.profiler.tool=npu",
                f"actor_rollout_ref.actor.profiler.enable={str(actor).lower()}",
                "actor_rollout_ref.actor.profiler.all_ranks=false",
                "actor_rollout_ref.actor.profiler.ranks=[0]",
                "actor_rollout_ref.actor.profiler.tool_config.npu.discrete=true",
                "actor_rollout_ref.rollout.profiler.tool=npu",
                f"actor_rollout_ref.rollout.profiler.enable={str(rollout).lower()}",
                "actor_rollout_ref.rollout.profiler.all_ranks=false",
                f"actor_rollout_ref.rollout.profiler.ranks=[{rollout_rank}]",
                "actor_rollout_ref.rollout.profiler.tool_config.npu.discrete=true",
            ],
        )
    for role, enabled, rank in [("actor", actor, 0), ("rollout", rollout, rollout_rank)]:
        role_cfg = OmegaConf.to_container(cfg.actor_rollout_ref[role].profiler, resolve=True)
        assert role_cfg["tool"] == "npu"
        assert role_cfg["enable"] is enabled
        assert role_cfg["all_ranks"] is False
        assert role_cfg["ranks"] == [rank]
        assert role_cfg["tool_config"]["npu"]["discrete"] is True
    events = []
    profiler = make_profiler(cfg, events, hybrid=hybrid)
    profiler.start_step(1)
    profiler.end_step(1)
    assert any(e[:2] == ("actor", "start") for e in events) is actor
    assert any(e[:2] == ("rollout", "start") for e in events) is rollout
    assert any(e[:2] == ("hybrid", "start") for e in events) is (hybrid and rollout)


@pytest.mark.parametrize("engine", ["dp_diffusion", "veomni_diffusion"])
@pytest.mark.parametrize("tool", ["torch", "npu"])
def test_compose_actor_and_rollout_profiling(engine, tool):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "verl_omni/trainer/config"), version_base=None):
        cfg = compose(
            config_name="diffusion_trainer",
            overrides=[
                f"diffusion/model_engine={engine}",
                "trainer.use_v1=true",
                f"global_profiler.tool={tool}",
                "global_profiler.steps=[1,2,4]",
                "global_profiler.save_path=outputs/profile_v1",
                "global_profiler.relocate_results=true",
                "global_profiler.finish_hook_cmd=echo done",
                "global_profiler.finish_hook_ranks=[0]",
                f"actor_rollout_ref.actor.profiler.tool={tool}",
                "actor_rollout_ref.actor.profiler.enable=true",
                "actor_rollout_ref.actor.profiler.all_ranks=true",
                f"actor_rollout_ref.rollout.profiler.tool={tool}",
                "actor_rollout_ref.rollout.profiler.enable=true",
                "actor_rollout_ref.rollout.profiler.all_ranks=true",
            ],
        )
    for role in ("actor", "rollout"):
        role_cfg = OmegaConf.to_container(cfg.actor_rollout_ref[role].profiler, resolve=True)
        assert role_cfg["enable"] is True
        assert role_cfg["tool"] == tool
        assert role_cfg["save_path"] == "outputs/profile_v1"
        assert role_cfg["relocate_results"] is True
        assert role_cfg["finish_hook_cmd"] == "echo done"
        assert role_cfg["finish_hook_ranks"] == [0]
        assert role_cfg["tool_config"][tool]["discrete"] is False
