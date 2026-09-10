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
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from verl_omni.workers.config.diffusion.actor import DiffusionFSDPEngineConfig
from verl_omni.workers.engine.fsdp.diffusers_impl import DiffusersFSDPEngine, PPODiffusersFSDPEngine


def make_engine(*, mode, train_gc=True, eval_gc=True, diagnostics=False):
    engine = object.__new__(PPODiffusersFSDPEngine)
    engine.engine_config = DiffusionFSDPEngineConfig(
        forward_only=False,
        gc_diagnostics=diagnostics,
        gc_on_train_device_load=train_gc,
        gc_on_eval_device_load=eval_gc,
    )
    engine.optimizer = None
    engine.mode = mode
    return engine


@pytest.mark.parametrize(
    ("mode", "train_gc", "eval_gc", "diagnostics", "expected_setting", "expected_point"),
    [
        ("train", False, 1, False, False, None),
        ("eval", 0, False, False, False, None),
        ("train", 0, True, True, 0, "train_device_load"),
        ("eval", 0, 1, True, 1, "eval_device_load"),
    ],
)
def test_device_load_forwards_gc_configuration(
    monkeypatch,
    mode,
    train_gc,
    eval_gc,
    diagnostics,
    expected_setting,
    expected_point,
):
    collect = Mock()
    engine = make_engine(mode=mode, train_gc=train_gc, eval_gc=eval_gc, diagnostics=diagnostics)
    monkeypatch.setattr("verl_omni.workers.engine.fsdp.diffusers_impl.get_device_name", lambda: "cuda")
    monkeypatch.setattr("verl_omni.workers.engine.fsdp.diffusers_impl.collect_garbage", collect)

    DiffusersFSDPEngine.to(engine, device="cuda", model=False, optimizer=False, grad=False)

    forwarded_setting = collect.call_args.args[0]
    assert forwarded_setting == expected_setting
    assert type(forwarded_setting) is type(expected_setting)
    assert collect.call_args.kwargs == {"diagnostics_point": expected_point}


@pytest.mark.parametrize(
    ("rollout_config", "diagnostics", "expected_kwargs"),
    [
        (
            {"gc_on_actor_offload": 1},
            False,
            {"force_sync": True, "gc_setting": 1, "gc_diagnostics_point": None},
        ),
        (
            {"gc_on_actor_offload": 1},
            True,
            {"force_sync": True, "gc_setting": 1, "gc_diagnostics_point": "actor_offload"},
        ),
        ({}, True, {"force_sync": True}),
    ],
)
def test_actor_offload_forwards_gc_configuration(monkeypatch, rollout_config, diagnostics, expected_kwargs):
    from verl_omni.workers import engine_workers

    cleanup = Mock()
    worker = SimpleNamespace(
        actor=SimpleNamespace(engine=SimpleNamespace(is_param_offload_enabled=False)),
        config=OmegaConf.create({"rollout": rollout_config}),
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
