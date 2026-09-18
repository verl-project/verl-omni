# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""CPU tests for controller-side Nsight Systems capture control."""

import pytest
from omegaconf import OmegaConf

from verl_omni.trainer.diffusion.ray_diffusion_trainer import PolicyGradientRayTrainer
from verl_omni.trainer.main_diffusion import (
    _count_controller_capture_ranges,
    _resolve_controller_nsight_options,
)


class _RecordingPlatform:
    def __init__(self, events):
        self.events = events

    def profiler_start(self):
        self.events.append("controller_start")

    def profiler_stop(self):
        self.events.append("controller_stop")


class _RecordingWorkerGroup:
    def __init__(self, events, role):
        self.events = events
        self.role = role

    def start_profile(self, **kwargs):
        self.events.append(f"{self.role}_start")

    def stop_profile(self):
        self.events.append(f"{self.role}_stop")


class _FailingStartWorkerGroup(_RecordingWorkerGroup):
    def start_profile(self, **kwargs):
        self.events.append(f"{self.role}_start")
        raise RuntimeError("worker start failed")


def _make_trainer(events, *, controller_enabled=True, use_reference_policy=True):
    trainer = PolicyGradientRayTrainer.__new__(PolicyGradientRayTrainer)
    trainer.global_steps = 3
    trainer.actor_rollout_wg = _RecordingWorkerGroup(events, "actor")
    trainer.ref_policy_wg = _RecordingWorkerGroup(events, "ref")
    trainer.use_reference_policy = use_reference_policy
    trainer.ref_in_actor = False
    trainer.use_teacher_policy = False
    trainer._controller_nsys_profile_enabled = controller_enabled
    trainer._controller_nsys_profile_active = False
    return trainer


def _make_config(steps, *, continuous, capture_range_end=None):
    return OmegaConf.create(
        {
            "global_profiler": {
                "steps": steps,
                "profile_continuous_steps": continuous,
                "global_tool_config": {
                    "nsys": {
                        "controller_nsight_options": {
                            "capture-range": "cudaProfilerApi",
                            "capture-range-end": capture_range_end,
                            "kill": "none",
                        }
                    }
                },
            }
        }
    )


def test_count_controller_capture_ranges():
    assert _count_controller_capture_ranges([1, 2, 5], profile_continuous_steps=False) == 3
    assert _count_controller_capture_ranges([1, 2, 5], profile_continuous_steps=True) == 2
    assert _count_controller_capture_ranges([5, 2, 1, 2], profile_continuous_steps=True) == 2


def test_resolve_controller_capture_range_end():
    options = _resolve_controller_nsight_options(_make_config([1, 2, 5], continuous=True))
    assert options["capture-range-end"] == "repeat-shutdown:2"


def test_preserve_explicit_controller_capture_range_end():
    options = _resolve_controller_nsight_options(
        _make_config([1, 2, 5], continuous=False, capture_range_end="repeat-shutdown:7")
    )
    assert options["capture-range-end"] == "repeat-shutdown:7"


def test_preserve_controller_options_without_capture_range():
    config = _make_config([1, 2, 5], continuous=False)
    controller_options = {
        "trace": "cuda,nvtx,cublas,ucx",
        "cuda-memory-usage": "true",
        "cuda-graph-trace": "graph",
    }
    config.global_profiler.global_tool_config.nsys.controller_nsight_options = controller_options

    options = _resolve_controller_nsight_options(config)

    assert options == controller_options


def test_controller_wraps_worker_profile_calls(monkeypatch):
    events = []
    trainer = _make_trainer(events)
    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.ray_diffusion_trainer.get_platform",
        lambda: _RecordingPlatform(events),
    )

    trainer._start_profiling(True)
    trainer._stop_profiling(True)

    assert events == ["controller_start", "actor_start", "ref_start", "actor_stop", "ref_stop", "controller_stop"]
    assert not trainer._controller_nsys_profile_active


def test_controller_profile_control_can_be_disabled(monkeypatch):
    events = []
    trainer = _make_trainer(events, controller_enabled=False, use_reference_policy=False)
    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.ray_diffusion_trainer.get_platform",
        lambda: _RecordingPlatform(events),
    )

    trainer._start_profiling(True)
    trainer._stop_profiling(True)

    assert events == ["actor_start", "actor_stop"]


def test_worker_start_failure_stops_controller(monkeypatch):
    events = []
    trainer = _make_trainer(events, use_reference_policy=False)
    trainer.actor_rollout_wg = _FailingStartWorkerGroup(events, "actor")
    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.ray_diffusion_trainer.get_platform",
        lambda: _RecordingPlatform(events),
    )

    with pytest.raises(RuntimeError, match="worker start failed"):
        trainer._start_profiling(True)

    assert events == ["controller_start", "actor_start", "controller_stop"]
    assert not trainer._controller_nsys_profile_active


def test_disabled_step_does_not_start_any_profiler(monkeypatch):
    events = []
    trainer = _make_trainer(events)
    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.ray_diffusion_trainer.get_platform",
        lambda: _RecordingPlatform(events),
    )

    trainer._start_profiling(False)
    trainer._stop_profiling(False)

    assert events == []
