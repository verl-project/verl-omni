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
"""Reject unsupported V1 profiling combinations before distributed startup."""

from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from verl_omni.trainer import main_diffusion_v1 as entrypoint


class StartupReached(Exception):
    """Stop accepted configurations before any distributed initialization."""


@pytest.fixture
def startup(monkeypatch):
    start = Mock(side_effect=StartupReached)
    ray_init = Mock()
    monkeypatch.setattr(entrypoint, "enable_rl_insight", start)
    monkeypatch.setattr(entrypoint.ray, "init", ray_init)
    yield start
    ray_init.assert_not_called()


def config(*, tool="torch", mode="sync", steps=(1, 2), continuous=False, rollout=True):
    return OmegaConf.create(
        {
            "trainer": {"v1": {"trainer_mode": mode}},
            "transfer_queue": {"enable": False},
            "global_profiler": {
                "tool": tool,
                "steps": steps,
                "profile_continuous_steps": continuous,
                "global_tool_config": {
                    "nsys": {
                        "controller_nsight_options": {},
                        "worker_nsight_options": {"capture-range": "cudaProfilerApi"},
                    }
                },
            },
            "actor_rollout_ref": {
                "actor": {"profiler": {"enable": True}},
                "rollout": {"profiler": {"enable": rollout}},
            },
        }
    )


@pytest.mark.parametrize("mode", ["sync", "separate_async"])
def test_reject_controller_cuda_capture_before_startup(startup, mode):
    cfg = config(tool="nsys", mode=mode)
    cfg.global_profiler.global_tool_config.nsys.controller_nsight_options["capture-range"] = "cudaProfilerApi"
    with pytest.raises(ValueError, match="controller-side Nsight capture-range=cudaProfilerApi"):
        entrypoint.run_diffusion_v1(cfg)
    startup.assert_not_called()


@pytest.mark.parametrize("mode", ["sync", "separate_async"])
@pytest.mark.parametrize("tool", ["nsys", "npu", "torch"])
def test_allow_default_worker_capture_and_noncontinuous_profiling(startup, mode, tool):
    with pytest.raises(StartupReached):
        entrypoint.run_diffusion_v1(config(tool=tool, mode=mode))
    startup.assert_called_once()


@pytest.mark.parametrize("tool", ["npu", "torch"])
def test_inactive_nsight_options_do_not_reject_other_tools(startup, tool):
    cfg = config(tool=tool)
    cfg.global_profiler.global_tool_config.nsys.controller_nsight_options["capture-range"] = "cudaProfilerApi"
    with pytest.raises(StartupReached):
        entrypoint.run_diffusion_v1(cfg)
    startup.assert_called_once()


@pytest.mark.parametrize("mode", ["sync", "separate_async"])
@pytest.mark.parametrize("tool", ["npu", "torch"])
def test_reject_continuous_rollout_before_startup(startup, mode, tool):
    with pytest.raises(ValueError, match="global_profiler.profile_continuous_steps=False"):
        entrypoint.run_diffusion_v1(config(tool=tool, mode=mode, continuous=True))
    startup.assert_not_called()


@pytest.mark.parametrize("tool", ["npu", "torch"])
def test_allow_actor_only_continuous_profiling(startup, tool):
    with pytest.raises(StartupReached):
        entrypoint.run_diffusion_v1(config(tool=tool, continuous=True, rollout=False))
    startup.assert_called_once()


@pytest.mark.parametrize("steps", [None, []])
def test_inactive_profiling_does_not_reject_unused_options(startup, steps):
    cfg = config(tool="nsys", steps=steps, continuous=True)
    cfg.global_profiler.global_tool_config.nsys.controller_nsight_options["capture-range"] = "cudaProfilerApi"
    with pytest.raises(StartupReached):
        entrypoint.run_diffusion_v1(cfg)
    startup.assert_called_once()
