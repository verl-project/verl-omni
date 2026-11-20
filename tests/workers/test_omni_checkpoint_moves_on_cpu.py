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
"""Checkpoint lifecycle guards on the native engine; no CUDA or file writes."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")

from verl.workers.engine.megatron import transformer_impl

from verl_omni.workers import engine_workers
from verl_omni.workers.engine.megatron import omni_impl


@pytest.fixture
def checkpoint(monkeypatch):
    engine = object.__new__(omni_impl.OmniMegatronEngine)
    engine.module = object()
    engine._is_offload_param = True
    tracker = engine.enable_parameter_tracking()
    worker = object.__new__(engine_workers.TrainingWorker)
    worker.engine = engine
    state = SimpleNamespace(engine=engine, tracker=tracker, worker=worker, events=[], failure=None, origin="cpu")

    def event(name, *args, **kwargs):
        state.events.append(name)
        if state.failure == name:
            raise OSError(f"injected {name} failure")

    def save(**kwargs):
        state.saved_arguments = kwargs
        event("save")

    engine.checkpoint_mananager = SimpleNamespace(save_checkpoint=save)
    for module in (transformer_impl, omni_impl):
        monkeypatch.setattr(module, "get_megatron_module_device", lambda _: state.origin, raising=False)
        monkeypatch.setattr(module, "load_megatron_model_to_gpu", lambda *a, **k: event("load"), raising=False)
        monkeypatch.setattr(module, "offload_megatron_model_to_cpu", lambda *a, **k: event("offload"), raising=False)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: event("barrier"))
    return state


@pytest.mark.parametrize("failure", ["load", "offload"])
def test_partial_checkpoint_move_fences_later_access(checkpoint, failure):
    checkpoint.failure = failure
    with pytest.raises(OSError, match=f"injected {failure}"):
        checkpoint.worker.save_checkpoint("unused")
    assert checkpoint.tracker._poisoned
    events = list(checkpoint.events)
    with pytest.raises(RuntimeError, match="Partial actor operation"):
        checkpoint.worker.save_checkpoint("unused")
    assert checkpoint.events == events


@pytest.mark.parametrize("failure", ["save", "barrier"])
def test_nonmutating_checkpoint_failure_preserves_revision(checkpoint, failure):
    before = checkpoint.tracker.version()
    checkpoint.failure = failure
    with pytest.raises(OSError, match=f"injected {failure}"):
        checkpoint.worker.save_checkpoint("unused")
    assert checkpoint.tracker.version() == before
    assert not checkpoint.tracker._poisoned
    # This is a local guard assertion, not permission to retry an uncertain
    # distributed checkpoint operation on a live worker group.
    with checkpoint.tracker.mutation("optimizer"):
        pass
    assert checkpoint.tracker.version().revision == before.revision + 1


@pytest.mark.parametrize("tracked", [False, True])
def test_success_preserves_native_checkpoint_order_and_arguments(checkpoint, tracked):
    before = checkpoint.tracker.version()
    if not tracked:
        del checkpoint.engine._parameter_revision
    checkpoint.worker.save_checkpoint("local", "remote", 7, 2)
    assert checkpoint.events == ["load", "save", "barrier", "offload"]
    assert checkpoint.saved_arguments == {
        "local_path": "local",
        "hdfs_path": "remote",
        "global_step": 7,
        "max_ckpt_to_keep": 2,
    }
    assert checkpoint.tracker.version() == before


def test_resident_model_without_offload_does_not_move(checkpoint):
    checkpoint.engine._is_offload_param = False
    checkpoint.origin = "cuda"
    checkpoint.worker.save_checkpoint("unused")
    assert checkpoint.events == ["save", "barrier"]


def test_direct_engine_save_cannot_interleave_with_parameter_writer(checkpoint):
    with checkpoint.tracker.mutation("restore"):
        with pytest.raises(RuntimeError, match="lifecycle operation"):
            checkpoint.engine.save_checkpoint("unused")
    assert checkpoint.events == []
