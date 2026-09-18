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
"""CPU checks for the rollout side of ``delta_sharded`` weight sync.

Drives the real code paths end to end on CPU: the diffusers engine's shard export
(seed + steady deltas) is encoded into wire flushes exactly as the delta checkpoint
engine encodes them (``verl.checkpoint_engine.delta_sync.encode``), then applied
through ``vLLMOmniColocateWorkerExtension.update_weights_from_ipc`` (delta_flush=True)
and verl's delta loader into a toy rollout model whose ``load_weights`` lands on
``param.copy_`` like vllm's loaders. Only the ZMQ/NCCL transport is faked.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from verl.checkpoint_engine.delta_sync.encode import DeltaParam, checksum
from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

import verl_omni.workers.rollout.base  # noqa: F401  -- rollout class registration side effect
import verl_omni.workers.rollout.vllm_rollout.server_adapter as server_adapter
from verl_omni.workers.config.diffusion import DiffusionModelConfig
from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine
from verl_omni.workers.rollout.vllm_rollout.delta_apply import (
    POSITIONS_NAME,
    SPEC_NAME,
    VALUES_NAME,
    DeltaFlushReceiver,
)
from verl_omni.workers.rollout.vllm_rollout.utils import vLLMOmniColocateWorkerExtension

# ---------------------------------------------------------------------------
# Trainer side: real diffusers engine export over a toy DiT
# ---------------------------------------------------------------------------


class _ToyDiT(torch.nn.Module):
    _checkpoint_conversion_mapping = {"^transformer_blocks": "blocks"}

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=False) for _ in range(2)])


def _make_engine(module) -> PPODiffusersFSDPEngine:
    engine = object.__new__(PPODiffusersFSDPEngine)
    engine.module = module
    engine._is_offload_param = False
    engine._uses_fsdp2_cpu_offload_policy = False
    model_config = object.__new__(DiffusionModelConfig)
    object.__setattr__(model_config, "lora", {})
    engine.model_config = model_config
    return engine


def _patch_engine_helpers(monkeypatch):
    import verl_omni.workers.engine.fsdp.diffusers_impl as diffusers_impl

    monkeypatch.setattr(diffusers_impl, "log_gpu_memory_usage", MagicMock())
    monkeypatch.setattr(diffusers_impl, "load_fsdp_model_to_gpu", MagicMock())
    monkeypatch.setattr(diffusers_impl, "offload_fsdp_model_to_cpu", MagicMock())
    monkeypatch.setattr(diffusers_impl, "get_device_id", lambda: torch.device("cpu"))


def _seed_export(engine) -> list:
    """The delta engine's seed: full export, every floating tensor cast to bf16."""
    full, _ = engine.get_per_tensor_param()
    return [(name, t.to(torch.bfloat16) if t.is_floating_point() else t) for name, t in full]


# ---------------------------------------------------------------------------
# Rollout side: toy model whose load_weights lands on copy_ like vllm's loaders
# ---------------------------------------------------------------------------


class _ToyRolloutModel:
    def __init__(self, state: dict):
        self.state = state

    def load_weights(self, weights):
        for name, tensor in weights:
            self.state[name].copy_(tensor)


# ---------------------------------------------------------------------------
# Wire encoding, mirroring the delta engine's flush assembly (world-1: gather is identity)
# ---------------------------------------------------------------------------


def _spec_tensor(spec: dict) -> torch.Tensor:
    return torch.frombuffer(bytearray(json.dumps(spec).encode()), dtype=torch.uint8)


def _encode_dense_flush(named_params: list) -> list:
    params, val_pieces = [], []
    val_off = 0
    for name, tensor in named_params:
        flat = tensor.reshape(-1)
        n = flat.numel()
        params.append(
            DeltaParam(
                name=name,
                dtype=str(flat.dtype).replace("torch.", ""),
                shape=list(tensor.shape),
                pos_start=0,
                pos_end=0,
                pos_width=4,
                val_start=val_off,
                val_end=val_off + n,
            )
        )
        val_pieces.append(flat)
        val_off += n
    values = torch.cat(val_pieces)
    empty_pos = torch.empty(0, dtype=torch.uint8)
    spec = {
        "encoding": "dense",
        "verify": False,
        "is_last": True,
        "params": [vars(p) for p in params],
        "checksum": checksum(empty_pos, values),
    }
    return [(SPEC_NAME, _spec_tensor(spec)), (VALUES_NAME, values)]


def _encode_indices_flush(deltas) -> list:
    params, idx_pieces, val_pieces = [], [], []
    pos_off = val_off = 0
    for slots, dtype_str, counts, hf_idx, hf_val, _pg in deltas:
        off = 0
        for (name, shape), count in zip(slots, counts.tolist(), strict=True):
            idx = hf_idx[off : off + count]
            val = hf_val[off : off + count]
            off += count
            if count == 0:
                continue
            idx_pieces.append(idx.to(torch.int32))
            val_pieces.append(val)
            params.append(
                DeltaParam(
                    name=name,
                    dtype=dtype_str,
                    shape=list(shape),
                    pos_start=pos_off,
                    pos_end=pos_off + count * 4,
                    pos_width=4,
                    val_start=val_off,
                    val_end=val_off + count,
                )
            )
            pos_off += count * 4
            val_off += count
    values = torch.cat(val_pieces) if val_pieces else torch.empty(0, dtype=torch.bfloat16)
    positions = torch.cat(idx_pieces).view(torch.uint8) if idx_pieces else torch.empty(0, dtype=torch.uint8)
    spec = {"encoding": "indices", "params": [vars(p) for p in params], "checksum": checksum(positions, values)}
    return [(SPEC_NAME, _spec_tensor(spec)), (POSITIONS_NAME, positions), (VALUES_NAME, values)]


def _run_delta_ipc(worker, buckets: list, monkeypatch):
    """Drive update_weights_from_ipc(delta_flush=True) with a fake bucketed receiver."""

    class _FakeReceiver:
        def __init__(self, zmq_handle, device, use_shm):
            pass

        def receive_weights(self, on_bucket_received):
            for i, bucket in enumerate(buckets):
                on_bucket_received(bucket, i == len(buckets) - 1)

    monkeypatch.setattr(bucketed_weight_transfer, "BucketedWeightReceiver", _FakeReceiver)
    vLLMOmniColocateWorkerExtension.update_weights_from_ipc(worker, delta_flush=True)


def _delta_worker(target):
    worker = SimpleNamespace(
        _pending_lora_peft_config=None,
        device=torch.device("cpu"),
        _get_zmq_handle=lambda: "ipc:///tmp/test-delta.sock",
        _get_standard_weight_model_and_config=lambda: None,
        model_runner=SimpleNamespace(pipeline=target),
    )
    worker._update_weights_from_delta_ipc = (
        lambda receiver: vLLMOmniColocateWorkerExtension._update_weights_from_delta_ipc(worker, receiver)
    )
    return worker


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_seed_plus_two_deltas_bit_exact(monkeypatch):
    """End-to-end-style: seed export + two delta rounds reconstruct the reference state."""
    torch.manual_seed(0)
    module = _ToyDiT()
    _patch_engine_helpers(monkeypatch)
    engine = _make_engine(module)

    # Session 1: the seed streams the full export values-only into dummy weights.
    seed_params = _seed_export(engine)
    target = _ToyRolloutModel({name: torch.zeros_like(t) for name, t in seed_params})
    _run_delta_ipc(_delta_worker(target), [_encode_dense_flush(seed_params)], monkeypatch)
    for name, tensor in seed_params:
        assert torch.equal(target.state[name], tensor), name
    engine.prime_delta_snapshots()

    # Session 2: perturb, export delta round 1, apply.
    with torch.no_grad():
        module.blocks[0].weight.view(-1)[3] += 0.5
    deltas, _ = engine.get_per_tensor_param_delta_shard()
    _run_delta_ipc(_delta_worker(target), [_encode_indices_flush(deltas)], monkeypatch)

    # Session 3: perturb again, export delta round 2, apply.
    with torch.no_grad():
        module.blocks[1].weight.view(-1)[7] -= 1.0
    deltas, _ = engine.get_per_tensor_param_delta_shard()
    _run_delta_ipc(_delta_worker(target), [_encode_indices_flush(deltas)], monkeypatch)

    reference = dict(_seed_export(engine))
    for name, ref in reference.items():
        assert torch.equal(target.state[name].view(torch.int16), ref.view(torch.int16)), name


def test_flush_split_across_buckets(monkeypatch):
    """One flush's tensors may land in different buckets; reassembly must hold."""
    torch.manual_seed(0)
    module = _ToyDiT()
    _patch_engine_helpers(monkeypatch)
    engine = _make_engine(module)

    seed_params = _seed_export(engine)
    target = _ToyRolloutModel({name: torch.zeros_like(t) for name, t in seed_params})
    _run_delta_ipc(_delta_worker(target), [_encode_dense_flush(seed_params)], monkeypatch)
    engine.prime_delta_snapshots()

    with torch.no_grad():
        module.blocks[0].weight.view(-1)[1] += 0.25
    deltas, _ = engine.get_per_tensor_param_delta_shard()
    spec, positions, values = _encode_indices_flush(deltas)

    _run_delta_ipc(_delta_worker(target), [[spec, positions], [values]], monkeypatch)

    reference = dict(_seed_export(engine))
    for name, ref in reference.items():
        assert torch.equal(target.state[name], ref), name


def test_checksum_mismatch_raises(monkeypatch):
    torch.manual_seed(0)
    module = _ToyDiT()
    _patch_engine_helpers(monkeypatch)
    engine = _make_engine(module)

    seed_params = _seed_export(engine)
    target = _ToyRolloutModel({name: torch.zeros_like(t) for name, t in seed_params})
    _run_delta_ipc(_delta_worker(target), [_encode_dense_flush(seed_params)], monkeypatch)
    engine.prime_delta_snapshots()

    with torch.no_grad():
        module.blocks[0].weight.view(-1)[2] += 0.5
    deltas, _ = engine.get_per_tensor_param_delta_shard()
    spec, positions, values = _encode_indices_flush(deltas)
    corrupted = values[1].clone()
    corrupted[0] = 42.0

    receiver = DeltaFlushReceiver(target)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        receiver.on_bucket([spec, positions, (VALUES_NAME, corrupted)], is_last=True)


def test_lora_with_delta_flush_raises():
    worker = SimpleNamespace(_pending_lora_peft_config=None)
    with pytest.raises(ValueError, match="does not apply LoRA"):
        vLLMOmniColocateWorkerExtension.update_weights_from_ipc(worker, peft_config={"r": 8}, delta_flush=True)


def test_delta_stream_protocol_violations_raise():
    target = _ToyRolloutModel({})
    values = (VALUES_NAME, torch.zeros(4, dtype=torch.bfloat16))

    # values before the flush spec
    receiver = DeltaFlushReceiver(target)
    with pytest.raises(RuntimeError, match="before the flush spec"):
        receiver.on_bucket([values])

    # stream ends between spec and values
    spec = {
        "encoding": "indices",
        "params": [],
        "checksum": 0,
    }
    receiver = DeltaFlushReceiver(target)
    with pytest.raises(RuntimeError, match="partial flush"):
        receiver.on_bucket([(SPEC_NAME, _spec_tensor(spec))], is_last=True)

    # unknown tensor name in the stream
    receiver = DeltaFlushReceiver(target)
    with pytest.raises(ValueError, match="unexpected tensor"):
        receiver.on_bucket([("weights.0", torch.zeros(1))])


def test_delta_apply_rejects_npu_platform(monkeypatch):
    from verl_omni.workers.rollout.vllm_rollout import npu_utils

    monkeypatch.setattr(npu_utils, "_is_npu_platform", lambda: True)
    worker = _delta_worker(_ToyRolloutModel({}))
    with pytest.raises(NotImplementedError, match="Ascend NPU"):
        vLLMOmniColocateWorkerExtension._update_weights_from_delta_ipc(worker, MagicMock())


def test_registry_points_at_delta_capable_adapter():
    from verl.workers.rollout.base import get_rollout_class

    assert get_rollout_class("vllm_omni", "async") is server_adapter.VLLMOmniServerAdapter


def test_server_adapter_flattens_delta_flushes(monkeypatch):
    adapter = object.__new__(server_adapter.VLLMOmniServerAdapter)
    adapter.use_shm = False
    adapter.zmq_handle = "ipc:///tmp/test-adapter.sock"
    adapter.config = SimpleNamespace(checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=512))
    adapter._has_server = False
    adapter.replica_rank = 1
    adapter.rollout_rank = 1

    calls = []

    async def _fake_execute(method, non_block=False, kwargs=None):
        calls.append((method, kwargs))
        return None

    adapter._execute_method = _fake_execute
    sent = []

    class _FakeSender:
        def __init__(self, zmq_handle, bucket_size_mb, use_shm):
            pass

        async def async_send_weights(self, weights):
            sent.extend(list(weights))

    monkeypatch.setattr(server_adapter, "BucketedWeightSender", _FakeSender)

    flushes = iter(
        [
            ([(SPEC_NAME, torch.zeros(4, dtype=torch.uint8)), (VALUES_NAME, torch.zeros(2))], False),
            (
                [
                    (SPEC_NAME, torch.zeros(4, dtype=torch.uint8)),
                    (POSITIONS_NAME, torch.zeros(4, dtype=torch.uint8)),
                    (VALUES_NAME, torch.zeros(1)),
                ],
                True,
            ),
        ]
    )
    import asyncio

    asyncio.run(adapter.update_weights(flushes, global_steps=3, wire_format="delta_flush"))

    assert calls == [("update_weights_from_ipc", {"use_shm": False, "delta_flush": True})]
    assert [name for name, _ in sent] == [SPEC_NAME, VALUES_NAME, SPEC_NAME, POSITIONS_NAME, VALUES_NAME]


def test_checkpoint_engine_worker_rejects_delta_for_non_vllm_omni():
    from verl_omni.workers.checkpoint_engine import OmniCheckpointEngineWorker

    rollout_config = SimpleNamespace(
        checkpoint_engine=SimpleNamespace(backend="delta_sharded"),
        name="vllm",
    )
    with pytest.raises(NotImplementedError, match="vllm_omni"):
        OmniCheckpointEngineWorker(rollout_config, model_config=SimpleNamespace())
