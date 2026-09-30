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
"""CPU checks for the rollout side of ``omni_delta_sharded`` weight sync.

The wire is verl's stock ``named_tensors`` bucketed channel end to end (verl's
unmodified vLLM ServerAdapter drives it): the omni engine subclass flattens
each delta flush's sentinel tensors (``__delta_spec__`` / ``__positions__`` /
``__values__``) into the stream with a ``#<flush>`` suffix, and the omni worker
extension routes the stream by sniffing the first bucket. These tests drive
both ends on CPU: the diffusers engine's shard export (seed + steady deltas) is
encoded into wire flushes exactly as the delta checkpoint engine encodes them
(``verl.checkpoint_engine.delta_sync.encode``), then applied through the
extension and verl's delta loader into a toy rollout model whose
``load_weights`` lands on ``param.copy_`` like vllm's loaders. Only the
ZMQ/NCCL transport is faked.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from verl.checkpoint_engine.delta_sync.encode import DeltaParam, checksum
from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

import verl_omni.workers.rollout.base  # noqa: F401  -- rollout class registration side effect
from verl_omni.workers.config.diffusion import DiffusionModelConfig
from verl_omni.workers.engine.fsdp.diffusers_impl import PPODiffusersFSDPEngine
from verl_omni.workers.rollout.vllm_rollout.utils import vLLMOmniColocateWorkerExtension

SPEC_NAME = "__delta_spec__"
POSITIONS_NAME = "__positions__"
VALUES_NAME = "__values__"

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


def _suffix(flush: list, flush_idx: int) -> list:
    """The engine-side flatten: every tensor name gains its flush index."""
    return [(f"{name}#{flush_idx}", tensor) for name, tensor in flush]


def _run_ipc(worker, buckets: list, monkeypatch, delta_flush=True):
    """Drive one ``update_weights_from_ipc`` call with a fake bucketed receiver."""

    class _FakeReceiver:
        def __init__(self, zmq_handle, device, use_shm):
            pass

        def receive_weights(self, on_bucket_received):
            for i, bucket in enumerate(buckets):
                on_bucket_received(bucket, i == len(buckets) - 1)

    monkeypatch.setattr(bucketed_weight_transfer, "BucketedWeightReceiver", _FakeReceiver)
    vLLMOmniColocateWorkerExtension.update_weights_from_ipc(worker, delta_flush=delta_flush)


def _as_worker(**attrs):
    """A namespace standing in for ``self``: the delta helpers are bound onto it so
    the unbound ``update_weights_from_ipc(worker, ...)`` call style works."""
    worker = SimpleNamespace(**attrs)
    for name in ("_apply_delta_bucket", "_finish_delta_stream"):
        setattr(worker, name, vLLMOmniColocateWorkerExtension.__dict__[name].__get__(worker))
    return worker


def _delta_worker(target):
    return _as_worker(
        device=torch.device("cpu"),
        _pending_lora_peft_config=None,
        _get_zmq_handle=lambda: "ipc:///tmp/test-delta.sock",
        _get_standard_weight_model_and_config=lambda: None,
        model_runner=SimpleNamespace(pipeline=target),
    )


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
    _run_ipc(_delta_worker(target), [_suffix(_encode_dense_flush(seed_params), 0)], monkeypatch)
    for name, tensor in seed_params:
        assert torch.equal(target.state[name], tensor), name
    engine.prime_delta_snapshots()

    # Session 2: perturb, export delta round 1, apply.
    with torch.no_grad():
        module.blocks[0].weight.view(-1)[3] += 0.5
    deltas, _ = engine.get_per_tensor_param_delta_shard()
    _run_ipc(_delta_worker(target), [_suffix(_encode_indices_flush(deltas), 0)], monkeypatch)

    # Session 3: perturb again, export delta round 2, apply.
    with torch.no_grad():
        module.blocks[1].weight.view(-1)[7] -= 1.0
    deltas, _ = engine.get_per_tensor_param_delta_shard()
    _run_ipc(_delta_worker(target), [_suffix(_encode_indices_flush(deltas), 0)], monkeypatch)

    reference = dict(_seed_export(engine))
    for name, ref in reference.items():
        assert torch.equal(target.state[name].view(torch.int16), ref.view(torch.int16)), name


def test_flush_split_across_buckets(monkeypatch):
    """One flush's tensors may land in different buckets; the payload must hold."""
    torch.manual_seed(0)
    module = _ToyDiT()
    _patch_engine_helpers(monkeypatch)
    engine = _make_engine(module)

    seed_params = _seed_export(engine)
    target = _ToyRolloutModel({name: torch.zeros_like(t) for name, t in seed_params})
    _run_ipc(_delta_worker(target), [_suffix(_encode_dense_flush(seed_params), 0)], monkeypatch)
    engine.prime_delta_snapshots()

    with torch.no_grad():
        module.blocks[0].weight.view(-1)[1] += 0.25
    deltas, _ = engine.get_per_tensor_param_delta_shard()
    (_sn, spec_t), (_pn, pos_t), (_vn, val_t) = _encode_indices_flush(deltas)

    _run_ipc(
        _delta_worker(target),
        [[(f"{SPEC_NAME}#0", spec_t), (f"{POSITIONS_NAME}#0", pos_t)], [(f"{VALUES_NAME}#0", val_t)]],
        monkeypatch,
    )

    reference = dict(_seed_export(engine))
    for name, ref in reference.items():
        assert torch.equal(target.state[name], ref), name


def test_two_flushes_sharing_one_bucket_both_apply(monkeypatch):
    """The bucketed sender keys each bucket's metadata by tensor name, which is why
    the engine suffixes sentinels with the flush index; two whole flushes landing in
    one bucket must still apply independently (review round-2 regression)."""
    torch.manual_seed(0)
    module = _ToyDiT()
    _patch_engine_helpers(monkeypatch)
    engine = _make_engine(module)

    seed_params = _seed_export(engine)
    target = _ToyRolloutModel({name: torch.zeros_like(t) for name, t in seed_params})
    _run_ipc(_delta_worker(target), [_suffix(_encode_dense_flush(seed_params), 0)], monkeypatch)
    engine.prime_delta_snapshots()

    with torch.no_grad():
        module.blocks[0].weight.view(-1)[3] += 0.5
    deltas_a, _ = engine.get_per_tensor_param_delta_shard()
    with torch.no_grad():
        module.blocks[1].weight.view(-1)[7] -= 1.0
    deltas_b, _ = engine.get_per_tensor_param_delta_shard()

    one_bucket = [*_suffix(_encode_indices_flush(deltas_a), 0), *_suffix(_encode_indices_flush(deltas_b), 1)]
    _run_ipc(_delta_worker(target), [one_bucket], monkeypatch)

    reference = dict(_seed_export(engine))
    for name, ref in reference.items():
        assert torch.equal(target.state[name].view(torch.int16), ref.view(torch.int16)), name


def test_sniffed_stream_routes_by_first_bucket(monkeypatch):
    """``delta_flush=None`` (the stock ServerAdapter path, which passes no flag)
    must route sentinel-led streams to the delta apply and dense-named streams to
    the ordinary bucketed load."""
    torch.manual_seed(0)
    module = _ToyDiT()
    _patch_engine_helpers(monkeypatch)
    engine = _make_engine(module)

    seed_params = _seed_export(engine)
    target = _ToyRolloutModel({name: torch.zeros_like(t) for name, t in seed_params})

    # Dense names through the same worker: loads through the pipeline's load_weights.
    _run_ipc(_delta_worker(target), [list(seed_params)], monkeypatch, delta_flush=None)
    for name, tensor in seed_params:
        assert torch.equal(target.state[name], tensor), name

    # Sentinel-led stream: applies through the delta receiver instead.
    engine.prime_delta_snapshots()
    with torch.no_grad():
        module.blocks[0].weight.view(-1)[2] += 0.5
    deltas, _ = engine.get_per_tensor_param_delta_shard()
    _run_ipc(_delta_worker(target), [_suffix(_encode_indices_flush(deltas), 0)], monkeypatch, delta_flush=None)

    reference = dict(_seed_export(engine))
    for name, ref in reference.items():
        assert torch.equal(target.state[name].view(torch.int16), ref.view(torch.int16)), name


def test_empty_terminal_bucket_is_noop(monkeypatch):
    """A zero-flush sync arrives as one empty is_last bucket. Sniffing that as a
    dense load calls diffusion load_weights([]) and, on AR, reruns
    process_weights_after_loading."""
    from verl_omni.workers.rollout.vllm_rollout import npu_utils

    monkeypatch.setattr(npu_utils, "_is_npu_platform", lambda: False)

    diffusion_calls = []
    target = _ToyRolloutModel({})
    target.load_weights = lambda weights: diffusion_calls.append(list(weights))
    _run_ipc(_delta_worker(target), [[]], monkeypatch, delta_flush=None)
    assert diffusion_calls == []

    class _AR(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(1, 1)
            self.loads = []

        def load_weights(self, weights):
            self.loads.append(list(weights))

    model = _AR()
    post = []
    import vllm.model_executor.model_loader.utils as loader_utils

    monkeypatch.setattr(loader_utils, "process_weights_after_loading", lambda *args, **kwargs: post.append(args))
    worker = _as_worker(
        device=torch.device("cpu"),
        _pending_lora_peft_config=None,
        _get_zmq_handle=lambda: "ipc:///tmp/test-delta.sock",
        _get_standard_weight_model_and_config=lambda: (model, SimpleNamespace()),
    )
    _run_ipc(worker, [[]], monkeypatch, delta_flush=None)
    assert model.loads == []
    assert post == []


def test_explicit_delta_flush_rejects_lora():
    worker = _delta_worker(_ToyRolloutModel({}))
    with pytest.raises(ValueError, match="does not apply LoRA adapters"):
        vLLMOmniColocateWorkerExtension.update_weights_from_ipc(worker, peft_config={"r": 8}, delta_flush=True)


def test_partial_flush_at_stream_end_raises(monkeypatch):
    """A stream ending between a spec and its values is corrupt; fail closed."""
    spec_entry = _encode_indices_flush([])[0]
    worker = _delta_worker(_ToyRolloutModel({}))
    with pytest.raises(RuntimeError, match="partial flush"):
        _run_ipc(worker, [[(f"{SPEC_NAME}#0", spec_entry[1])]], monkeypatch)


def test_delta_apply_rejects_fused_moe_rollout(monkeypatch):
    from verl_omni.workers.rollout.vllm_rollout import npu_utils
    from verl_omni.workers.rollout.vllm_rollout import utils as rollout_utils

    monkeypatch.setattr(npu_utils, "_is_npu_platform", lambda: False)

    class _FakeRoutedExperts(torch.nn.Module):
        pass

    import vllm.model_executor.layers.fused_moe.routed_experts as routed_experts_module

    monkeypatch.setattr(routed_experts_module, "RoutedExperts", _FakeRoutedExperts)
    assert not rollout_utils._model_has_fused_moe(torch.nn.Linear(2, 2))
    model = torch.nn.Module()
    model.moe = _FakeRoutedExperts()
    assert rollout_utils._model_has_fused_moe(model)

    worker = _as_worker(
        device=torch.device("cpu"),
        _pending_lora_peft_config=None,
        _get_zmq_handle=lambda: "ipc:///tmp/test-delta.sock",
        _get_standard_weight_model_and_config=lambda: (model, SimpleNamespace()),
    )
    import vllm.model_executor.model_loader.reload as reload_mod

    started = []
    monkeypatch.setattr(reload_mod, "initialize_layerwise_reload", lambda model: started.append("init"))
    monkeypatch.setattr(reload_mod, "finalize_layerwise_reload", lambda *args, **kwargs: started.append("fin"))
    # delta_flush=None is the stock ServerAdapter path. The explicit True path
    # never calls initialize_layerwise_reload, so it cannot catch this bug.
    with pytest.raises(NotImplementedError, match="fused-MoE"):
        _run_ipc(
            worker,
            [[(f"{SPEC_NAME}#0", torch.zeros(1, dtype=torch.uint8))]],
            monkeypatch,
            delta_flush=None,
        )
    assert started == []

    # A real dense bucket on the same path still reloads, after the sniff.
    model.load_weights = lambda weights: None
    _run_ipc(worker, [[("blocks.0.weight", torch.zeros(1))]], monkeypatch, delta_flush=None)
    assert started == ["init", "fin"]


def test_checksum_mismatch_raises(monkeypatch):
    torch.manual_seed(0)
    module = _ToyDiT()
    _patch_engine_helpers(monkeypatch)
    engine = _make_engine(module)

    seed_params = _seed_export(engine)
    target = _ToyRolloutModel({name: torch.zeros_like(t) for name, t in seed_params})
    _run_ipc(_delta_worker(target), [_suffix(_encode_dense_flush(seed_params), 0)], monkeypatch)
    engine.prime_delta_snapshots()

    with torch.no_grad():
        module.blocks[0].weight.view(-1)[2] += 0.5
    deltas, _ = engine.get_per_tensor_param_delta_shard()
    (_sn, spec_t), (_pn, pos_t), (_vn, val_t) = _encode_indices_flush(deltas)
    corrupted = val_t.clone()
    corrupted[0] = 42.0

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _run_ipc(
            _delta_worker(target),
            [[(f"{SPEC_NAME}#0", spec_t), (f"{POSITIONS_NAME}#0", pos_t), (f"{VALUES_NAME}#0", corrupted)]],
            monkeypatch,
        )


def test_delta_apply_rejects_npu_platform(monkeypatch):
    from verl_omni.workers.rollout.vllm_rollout import npu_utils

    monkeypatch.setattr(npu_utils, "_is_npu_platform", lambda: True)
    worker = _delta_worker(_ToyRolloutModel({}))
    with pytest.raises(NotImplementedError, match="Ascend NPU"):
        _run_ipc(worker, [[(f"{SPEC_NAME}#0", torch.zeros(1, dtype=torch.uint8))]], monkeypatch)


def test_registry_keeps_verl_server_adapter():
    """The delta wire rides verl's stock named_tensors bucketed channel, so the
    rollout class is verl's own ServerAdapter -- no omni subclass at any pin."""
    from verl.workers.rollout.base import get_rollout_class
    from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter

    assert get_rollout_class("vllm_omni", "async") is ServerAdapter


def _flattened_pairs(flushes):
    """Expected ``(name#flush, tensor)`` pairs. ``flushes[i]`` is ``(named, is_last)``;
    the tensor is ``flushes[i][0][j][1]``, not the ``is_last`` bool at ``[i][1]``."""
    return [
        (f"{name}#{flush_idx}", tensor) for flush_idx, (named, _is_last) in enumerate(flushes) for name, tensor in named
    ]


def _assert_flattened(got, expected):
    assert [(name, tensor.tolist()) for name, tensor in got] == [(name, tensor.tolist()) for name, tensor in expected]


def test_omni_delta_sharded_registers_flattening_engine(monkeypatch):
    """The omni backend is a thin subclass of verl's DeltaShardedCheckpointEngine
    that re-declares the wire as named_tensors and flattens flushes into
    sentinel-suffixed pairs, so verl's unmodified worker and ServerAdapter drive
    the whole sync. Without the transport deps (e.g. CPU envs without cupy) the
    alias is absent and asking for it fails closed. The flatten itself still runs
    here: the parent class is what needs cupy, not the suffix loop."""
    from verl.checkpoint_engine import CheckpointEngineRegistry, DeltaShardedCheckpointEngine

    from verl_omni.workers.checkpoint_engine import _flatten_flush_stream

    flushes = [
        ([(SPEC_NAME, torch.zeros(1)), (VALUES_NAME, torch.zeros(1))], True),
        ([(SPEC_NAME, torch.zeros(1)), (POSITIONS_NAME, torch.zeros(1)), (VALUES_NAME, torch.zeros(1))], True),
    ]
    expected = _flattened_pairs(flushes)
    # Always executed, including the CPU path where the parent engine is None.
    _assert_flattened(list(_flatten_flush_stream(flushes)), expected)

    if DeltaShardedCheckpointEngine is None:
        with pytest.raises(ValueError, match="not registered"):
            CheckpointEngineRegistry.get("omni_delta_sharded")
        return

    engine_cls = CheckpointEngineRegistry.get("omni_delta_sharded")
    assert engine_cls is not DeltaShardedCheckpointEngine
    assert issubclass(engine_cls, DeltaShardedCheckpointEngine)
    assert engine_cls.wire_format == "named_tensors"

    monkeypatch.setattr(DeltaShardedCheckpointEngine, "receive_weights", lambda self, global_steps=None: iter(flushes))
    engine = object.__new__(engine_cls)
    _assert_flattened(list(engine.receive_weights()), expected)


def test_ar_strategy_routes_only_weight_sync_rpcs():
    """The delta stream rides ``update_weights_from_ipc``; no other RPC may be
    broadcast to the weight-sync stages."""
    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy import ARStrategy

    strategy = SimpleNamespace(_weight_sync_stage_ids=[2])
    for name in ("set_pending_lora_peft_config", "update_weights_from_ipc", "monkey_patch_model"):
        assert ARStrategy.collective_rpc_stage_ids(strategy, name) == [2], name
    for name in ("init_weight_transfer_engine", "update_verl_delta_weights"):
        assert ARStrategy.collective_rpc_stage_ids(strategy, name) is None, name
