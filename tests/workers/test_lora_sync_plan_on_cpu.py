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
"""CPU tests for the one LoRA sync decision table (``lora_sync.resolve_lora_sync_plan``).

Ports upstream verl's sync-ordering rows onto the real omni worker: every row
is pinned twice — as a pure plan, and as the behavior the real
``update_weights`` executes:
- adapter mode, first sync: base weights then adapter deltas, level-1 sleep;
- adapter mode, steady state: adapter-only deltas on the direct-IPC fast path;
- merge mode / non-LoRA: a single full sync at level-2 sleep.

Impossible combinations raise before any weight is gathered.
"""

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

import verl_omni.workers.engine_workers as ew
from verl_omni.utils.config import LoRASettings
from verl_omni.workers.lora_sync import SyncExport, SyncRolloutAction, resolve_lora_sync_plan

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings(*, enabled=True, merge=False, adapters=("default",)) -> LoRASettings:
    return LoRASettings(
        rank=8 if enabled else 0,
        alpha=16,
        adapter_path=None,
        merge=merge,
        adapters=adapters,
        enabled=enabled,
    )


def _peft_module():
    module = torch.nn.Module()
    module.proj = torch.nn.Linear(4, 4, bias=False)
    module.peft_config = {"default": SimpleNamespace(to_dict=lambda: {"r": 8})}
    return module


class _NoopRLInsight:
    @staticmethod
    def trace_state(*args, **kwargs):
        return contextlib.nullcontext()


def _bare_worker(
    *,
    has_lora,
    enabled=None,
    merge=False,
    base_sync_done=True,
    adapters=("default",),
    rollout_adapter="default",
):
    """A bare worker whose engine/rollout mocks cover both colocated sync paths.

    ``enabled`` overrides the resolved ``LoRASettings.enabled``; it defaults to
    the engine's wrap state. Pass ``enabled=True`` with a plain engine to model
    a config/engine mismatch.
    """
    worker = object.__new__(ew.ActorRolloutRefWorker)

    # Adapter metadata exists exactly when the engine is PEFT-wrapped; the
    # export's wire peft_config is None for merged payloads (a plain full update).
    engine = MagicMock(spec=["module", "get_per_tensor_param", "get_lora_peft_config", "is_param_offload_enabled"])
    engine.module = _peft_module() if has_lora else torch.nn.Linear(4, 4, bias=False)
    engine.is_param_offload_enabled = False
    engine.get_lora_peft_config = MagicMock(return_value={"r": 8} if has_lora else None)
    wire_peft_config = {"r": 8} if has_lora and not merge else None

    def _get_per_tensor_param(*, layered_summon=False, base_sync_done=True, adapter_name="default", **kwargs):
        tensors = [("adapter.lora_A.weight", torch.zeros(1))] if base_sync_done else [("base.weight", torch.zeros(1))]
        return (iter(tensors), wire_peft_config)

    engine.get_per_tensor_param = MagicMock(side_effect=_get_per_tensor_param)
    worker.actor = SimpleNamespace(engine=engine)

    rollout = AsyncMock()
    rollout.resume = AsyncMock()
    rollout.update_weights = AsyncMock()
    rollout.sleep_level = 2
    worker.rollout = rollout

    worker.config = SimpleNamespace(rollout=SimpleNamespace(free_cache_engine=False))
    worker.lora_settings = _settings(enabled=has_lora if enabled is None else enabled, merge=merge, adapters=adapters)
    worker.base_sync_done = base_sync_done
    worker.layered_summon = False
    worker.rollout_adapter = rollout_adapter
    worker._rank = 0  # update_weights stamps timings/metrics with f"rank_{self.rank}"
    worker._zmq_update_seq = 0
    worker._offload_actor_and_empty_cache = lambda *args, **kwargs: None
    return worker


def _run_naive_update(monkeypatch, worker, global_steps=3):
    monkeypatch.setattr(ew, "RLInsightLogger", _NoopRLInsight)
    monkeypatch.setattr(ew, "set_expandable_segments", MagicMock())
    monkeypatch.setattr(ew, "log_gpu_memory_usage", MagicMock())
    monkeypatch.setattr(ew, "get_torch_device", MagicMock(is_available=MagicMock(return_value=False)))
    sender = MagicMock()
    sender.async_send_weights = AsyncMock()
    with patch.object(ew, "BucketedWeightSender", return_value=sender):
        asyncio.run(ew.ActorRolloutRefWorker.update_weights(worker, mode="naive", global_steps=global_steps))
    return worker


def _wire_fast_path_rollout(worker):
    """Swap in the mock surface the direct-IPC path needs (handles, RPC, sender)."""
    server_handle = MagicMock()
    server_handle.clear_kv_cache.remote = AsyncMock()
    server_handle.set_global_steps.remote = AsyncMock()
    rollout = MagicMock()
    rollout.zmq_handle = "ipc:///tmp/test-lora-sync-plan.sock"
    rollout.use_shm = False
    rollout.rollout_rank = 1
    rollout._ensure_server_handle.return_value = True
    rollout.server_handle = server_handle
    rollout._execute_method = AsyncMock(return_value=None)
    rollout.resume = AsyncMock()
    rollout.update_weights = AsyncMock()
    rollout.sleep_level = 2
    worker.rollout = rollout
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(
            free_cache_engine=False,
            checkpoint_engine=SimpleNamespace(backend="naive", update_weights_bucket_megabytes=16),
        )
    )
    return worker


# ---------------------------------------------------------------------------
# Pure plan rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base_sync_done,expected_export,expected_action",
    [
        (False, SyncExport.FULL, SyncRolloutAction.LOAD_WEIGHTS),
        (True, SyncExport.ADAPTER, SyncRolloutAction.REPLACE_ADAPTER),
    ],
)
def test_adapter_mode_rows_send_peft_config_at_level_1(base_sync_done, expected_export, expected_action):
    plan = resolve_lora_sync_plan(_settings(), adapter_name="default", has_adapter=True, base_sync_done=base_sync_done)

    assert plan.export is expected_export
    assert plan.rollout_action is expected_action
    assert plan.send_peft_config is True
    assert plan.sleep_level == 1


def test_adapter_delta_row_carries_the_named_adapter():
    plan = resolve_lora_sync_plan(
        _settings(adapters=("default", "old")), adapter_name="old", has_adapter=True, base_sync_done=True
    )

    assert plan.adapter_name == "old"
    assert plan.export is SyncExport.ADAPTER
    assert plan.rollout_action is SyncRolloutAction.REPLACE_ADAPTER


def test_merged_row_is_a_full_level_2_sync():
    plan = resolve_lora_sync_plan(_settings(merge=True), adapter_name="default", has_adapter=True, base_sync_done=True)

    assert plan.export is SyncExport.MERGED
    assert plan.rollout_action is SyncRolloutAction.LOAD_WEIGHTS
    assert plan.send_peft_config is False
    assert plan.sleep_level == 2


def test_non_lora_row_is_a_full_level_2_sync():
    plan = resolve_lora_sync_plan(
        _settings(enabled=False), adapter_name="default", has_adapter=False, base_sync_done=True
    )

    assert plan.export is SyncExport.FULL
    assert plan.rollout_action is SyncRolloutAction.LOAD_WEIGHTS
    assert plan.send_peft_config is False
    assert plan.sleep_level == 2


def test_merge_rejects_a_named_adapter():
    with pytest.raises(ValueError, match="exports the active 'default' adapter only"):
        resolve_lora_sync_plan(_settings(merge=True), adapter_name="old", has_adapter=True, base_sync_done=True)


def test_enabled_without_the_requested_adapter_raises():
    with pytest.raises(ValueError, match="holds no 'old' adapter"):
        resolve_lora_sync_plan(_settings(), adapter_name="old", has_adapter=False, base_sync_done=True)


# ---------------------------------------------------------------------------
# Colocated standard path (real worker, mocked engine/rollout)
# ---------------------------------------------------------------------------


def test_naive_adapter_first_sync_sends_base_then_adapter(monkeypatch):
    worker = _run_naive_update(monkeypatch, _bare_worker(has_lora=True, base_sync_done=False))

    fetches = worker.actor.engine.get_per_tensor_param.call_args_list
    assert [call.kwargs["base_sync_done"] for call in fetches] == [False, True]
    assert worker.rollout.update_weights.await_count == 2
    first, second = worker.rollout.update_weights.await_args_list
    assert first.kwargs["base_sync_done"] is False  # base weights first
    assert first.kwargs["peft_config"] == {"r": 8}  # the rollout recreates the adapter
    assert second.kwargs["base_sync_done"] is True  # then adapter deltas
    assert second.kwargs["peft_config"] == {"r": 8}
    assert worker.rollout.sleep_level == 1  # adapter mode pins level-1 sleep
    assert worker.base_sync_done is True


def test_naive_merged_mode_sends_single_full_sync(monkeypatch):
    worker = _run_naive_update(monkeypatch, _bare_worker(has_lora=True, merge=True, base_sync_done=False))

    assert worker.actor.engine.get_per_tensor_param.call_count == 1
    worker.rollout.update_weights.assert_awaited_once()
    assert worker.rollout.update_weights.await_args.kwargs["base_sync_done"] is True
    assert worker.rollout.update_weights.await_args.kwargs["peft_config"] is None
    assert worker.rollout.sleep_level == 2


def test_naive_non_lora_sends_single_sync(monkeypatch):
    worker = _run_naive_update(monkeypatch, _bare_worker(has_lora=False, base_sync_done=False))

    worker.rollout.update_weights.assert_awaited_once()
    assert worker.rollout.update_weights.await_args.kwargs["base_sync_done"] is True
    assert worker.rollout.update_weights.await_args.kwargs["peft_config"] is None
    assert worker.rollout.sleep_level == 2


def test_naive_adapter_delta_rides_the_direct_ipc_fast_path(monkeypatch):
    worker = _wire_fast_path_rollout(_bare_worker(has_lora=True))
    worker = _run_naive_update(monkeypatch, worker, global_steps=7)

    fetch = worker.actor.engine.get_per_tensor_param.call_args
    assert fetch.kwargs["adapter_name"] == "default"
    assert fetch.kwargs["base_sync_done"] is True
    worker.rollout.update_weights.assert_not_awaited()  # standard path skipped
    ipc_call = worker.rollout._execute_method.await_args
    assert ipc_call.args[0] == "update_weights_from_ipc"
    assert ipc_call.kwargs["kwargs"]["peft_config"] == {"r": 8}
    assert ipc_call.kwargs["kwargs"]["base_sync_done"] is True
    assert worker.rollout.sleep_level == 1


def test_naive_old_adapter_delta_ships_the_old_policy(monkeypatch):
    worker = _wire_fast_path_rollout(_bare_worker(has_lora=True, adapters=("default", "old"), rollout_adapter="old"))
    worker = _run_naive_update(monkeypatch, worker, global_steps=7)

    fetch = worker.actor.engine.get_per_tensor_param.call_args
    assert fetch.kwargs["adapter_name"] == "old"
    worker.rollout.update_weights.assert_not_awaited()


def test_naive_merge_named_adapter_raises_before_any_gather(monkeypatch):
    worker = _bare_worker(has_lora=True, merge=True, adapters=("default", "old"), rollout_adapter="old")

    with pytest.raises(ValueError, match="exports the active 'default' adapter only"):
        _run_naive_update(monkeypatch, worker)

    worker.actor.engine.get_per_tensor_param.assert_not_called()
    worker.rollout.update_weights.assert_not_awaited()


def test_naive_enabled_without_requested_adapter_raises(monkeypatch):
    worker = _bare_worker(has_lora=False, enabled=True, rollout_adapter="old")

    with pytest.raises(ValueError, match="holds no 'old' adapter"):
        _run_naive_update(monkeypatch, worker)

    worker.actor.engine.get_per_tensor_param.assert_not_called()
    worker.rollout.update_weights.assert_not_awaited()


# ---------------------------------------------------------------------------
# Standalone checkpoint-engine path
# ---------------------------------------------------------------------------


def test_standalone_first_adapter_sync_raises():
    # The standalone rollout loads real base weights via load_format; with a
    # dummy load_format the adapter deltas would land on a base-less model.
    worker = _bare_worker(has_lora=True, base_sync_done=False)
    worker.config = SimpleNamespace(rollout=SimpleNamespace(checkpoint_engine=SimpleNamespace(backend="nccl")))
    worker.checkpoint_engine = SimpleNamespace(send_weights=AsyncMock())

    with pytest.raises(ValueError, match="cannot stage the first adapter sync"):
        asyncio.run(worker.update_weights(mode="nccl", global_steps=1))

    worker.checkpoint_engine.send_weights.assert_not_awaited()


# ---------------------------------------------------------------------------
# Standalone replica metadata RPC
# ---------------------------------------------------------------------------


def test_peft_config_rpc_raises_when_lora_enabled_but_engine_is_plain():
    worker = object.__new__(ew.ActorRolloutRefWorker)
    worker.role = "actor"
    worker.lora_settings = _settings()
    worker.rollout_adapter = "default"
    worker.base_sync_done = True
    worker.actor = SimpleNamespace(engine=SimpleNamespace(module=torch.nn.Linear(2, 2)))

    with pytest.raises(ValueError, match="holds no 'default' adapter"):
        worker.get_lora_peft_config()


def test_peft_config_rpc_none_for_non_actor_role():
    worker = object.__new__(ew.ActorRolloutRefWorker)
    worker.role = "rollout"

    assert worker.get_lora_peft_config() is None
