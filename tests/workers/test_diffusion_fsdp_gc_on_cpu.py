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

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
from omegaconf import OmegaConf

from verl_omni.workers.config.diffusion.actor import DiffusionFSDPEngineConfig
from verl_omni.workers.engine.fsdp.diffusers_impl import DiffusersFSDPEngine, PPODiffusersFSDPEngine


def make_engine(*, mode, regional_compile=False, diagnostics=False):
    engine = object.__new__(PPODiffusersFSDPEngine)
    engine.engine_config = DiffusionFSDPEngineConfig(
        forward_only=False,
        gc_diagnostics=diagnostics,
    )
    engine.model_config = SimpleNamespace(use_regional_compile=regional_compile)
    engine.optimizer = None
    engine.mode = mode
    return engine


@pytest.mark.parametrize(
    ("mode", "regional_compile", "diagnostics", "expected_call"),
    [
        ("train", False, False, call(diagnostics_point=None)),
        ("eval", False, True, call(diagnostics_point="eval_device_load")),
        ("train", True, False, None),
        ("eval", True, True, None),
    ],
)
def test_device_load_uses_regional_compile_gc_policy(
    monkeypatch,
    mode,
    regional_compile,
    diagnostics,
    expected_call,
):
    collect = Mock()
    engine = make_engine(mode=mode, regional_compile=regional_compile, diagnostics=diagnostics)
    monkeypatch.setattr("verl_omni.workers.engine.fsdp.diffusers_impl.get_device_name", lambda: "cuda")
    monkeypatch.setattr("verl_omni.workers.engine.fsdp.diffusers_impl.collect_garbage", collect)

    DiffusersFSDPEngine.to(engine, device="cuda", model=False, optimizer=False, grad=False)

    assert collect.call_args == expected_call


@pytest.mark.parametrize(
    ("regional_compile", "diagnostics", "expected_kwargs"),
    [
        (
            False,
            False,
            {"force_sync": True, "gc_diagnostics_point": None},
        ),
        (
            True,
            True,
            {"force_sync": True, "gc_setting": False, "gc_diagnostics_point": "actor_offload"},
        ),
        (None, True, {"force_sync": True, "gc_diagnostics_point": "actor_offload"}),
    ],
)
def test_actor_offload_uses_regional_compile_gc_policy(monkeypatch, regional_compile, diagnostics, expected_kwargs):
    from verl_omni.workers import engine_workers

    cleanup = Mock()
    model_config = {}
    if regional_compile is not None:
        model_config["use_regional_compile"] = regional_compile
    worker = SimpleNamespace(
        actor=SimpleNamespace(engine=SimpleNamespace(is_param_offload_enabled=False)),
        config=OmegaConf.create(
            {
                "model": model_config,
                "rollout": {},
            }
        ),
        gc_diagnostics=diagnostics,
    )
    monkeypatch.setattr(engine_workers, "aggressive_empty_cache", cleanup)

    engine_workers.ActorRolloutRefWorker._offload_actor_and_empty_cache(worker)

    cleanup.assert_called_once_with(**expected_kwargs)


def test_manual_accelerator_load_uses_default_gc(monkeypatch):
    collect = Mock()
    engine = make_engine(mode=None, diagnostics=True)
    monkeypatch.setattr("verl_omni.workers.engine.fsdp.diffusers_impl.get_device_name", lambda: "cuda")
    monkeypatch.setattr("verl_omni.workers.engine.fsdp.diffusers_impl.collect_garbage", collect)

    DiffusersFSDPEngine.to(engine, device="cuda", model=False, optimizer=False, grad=False)

    collect.assert_called_once_with()
