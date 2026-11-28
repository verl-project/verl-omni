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
"""CPU tests for per-repeat validation seeding when val_kwargs.n > 1."""

import asyncio
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import NonTensorData, NonTensorStack, TensorDict
from verl.protocol import DataProto

from verl_omni.agent_loop import composite_agent_loop, diffusion_agent_loop, diffusion_agent_loop_tq
from verl_omni.agent_loop.composite_agent_loop import CompositeAgentLoopWorker
from verl_omni.agent_loop.diffusion_agent_loop import DiffusionAgentLoopWorker
from verl_omni.agent_loop.diffusion_agent_loop_tq import DiffusionAgentLoopWorkerTQ
from verl_omni.agent_loop.utils import derive_validation_seed, maybe_validation_seeds


def test_derive_validation_seed_preserves_repeat_zero_and_offsets_repeats():
    assert derive_validation_seed(42, 0) == 42
    assert [derive_validation_seed(42, idx) for idx in range(4)] == [42, 43, 44, 45]
    assert derive_validation_seed(0, 0) == 0


def test_maybe_validation_seeds_handles_none_and_validates_length():
    assert maybe_validation_seeds(None, 2, [0, 1]) is None
    assert maybe_validation_seeds(42, 2, None) is None
    assert maybe_validation_seeds(42, 4, [0, 1, 0, 1]) == [42, 43, 42, 43]
    entropy = maybe_validation_seeds(None, 4, [0, 1, 0, 1], val_n=2)
    assert entropy is not None and len(entropy) == 4
    assert all(isinstance(seed, int) and 0 <= seed < (1 << 63) for seed in entropy)
    assert len(set(entropy)) == 4

    with pytest.raises(ValueError, match="Expected 3 validation repeat indices, got 2"):
        maybe_validation_seeds(42, 3, [0, 1])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("val_n", "val_seed", "expected"),
    [
        (1, 42, {("a", 0): 42, ("b", 0): 42}),
        (
            4,
            100,
            {
                ("a", 0): 100,
                ("a", 1): 101,
                ("a", 2): 102,
                ("a", 3): 103,
                ("b", 0): 100,
                ("b", 1): 101,
                ("b", 2): 102,
                ("b", 3): 103,
            },
        ),
        (1, None, {("a", 0): None, ("b", 0): None}),
    ],
)
async def test_tq_worker_derives_distinct_validation_seeds_per_repeat(monkeypatch, val_n, val_seed, expected):
    worker_cls = DiffusionAgentLoopWorkerTQ.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    worker.background_tasks = set()
    worker.rollout_config = SimpleNamespace(
        n=2,
        pipeline={},
        algo={},
        calculate_log_probs=False,
        agent=SimpleNamespace(default_agent_loop="diffusion_single_turn_agent"),
        val_kwargs=SimpleNamespace(n=val_n, seed=val_seed, pipeline={}, algo={}),
    )

    async def fake_kv_put(*, key, partition_id, tag):
        del key, partition_id, tag

    captured_seeds: dict[tuple[str, int], int | None] = {}

    async def fake_run_agent_loop(self, sampling_params, *, session_id, uid, **kwargs):
        del self, kwargs
        captured_seeds[(uid, session_id)] = sampling_params.get("seed")

    monkeypatch.setattr(diffusion_agent_loop_tq, "_config_to_sampling_dict", lambda _cfg: {})
    monkeypatch.setattr(diffusion_agent_loop_tq.tq, "async_kv_put", fake_kv_put)
    worker._run_agent_loop = MethodType(fake_run_agent_loop, worker)

    chunk = TensorDict(
        {
            "index": torch.tensor([0, 1]),
            "uid": NonTensorStack("a", "b"),
            "global_steps": NonTensorData(5),
            "validate": NonTensorData(True),
        },
        batch_size=[2],
    )

    await worker.generate_sequences(chunk)
    await asyncio.gather(*worker.background_tasks)

    assert captured_seeds == expected


@pytest.mark.asyncio
async def test_tq_worker_unseeded_multi_repeat_validation_uses_entropy_seeds(monkeypatch):
    worker_cls = DiffusionAgentLoopWorkerTQ.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    worker.background_tasks = set()
    worker.rollout_config = SimpleNamespace(
        n=2,
        pipeline={},
        algo={},
        calculate_log_probs=False,
        agent=SimpleNamespace(default_agent_loop="diffusion_single_turn_agent"),
        val_kwargs=SimpleNamespace(n=3, seed=None, pipeline={}, algo={}),
    )

    async def fake_kv_put(*, key, partition_id, tag):
        del key, partition_id, tag

    captured_seeds: dict[tuple[str, int], int | None] = {}

    async def fake_run_agent_loop(self, sampling_params, *, session_id, uid, **kwargs):
        del self, kwargs
        captured_seeds[(uid, session_id)] = sampling_params.get("seed")

    monkeypatch.setattr(diffusion_agent_loop_tq, "_config_to_sampling_dict", lambda _cfg: {})
    monkeypatch.setattr(diffusion_agent_loop_tq.tq, "async_kv_put", fake_kv_put)
    worker._run_agent_loop = MethodType(fake_run_agent_loop, worker)

    chunk = TensorDict(
        {
            "index": torch.tensor([0, 1]),
            "uid": NonTensorStack("a", "b"),
            "global_steps": NonTensorData(5),
            "validate": NonTensorData(True),
        },
        batch_size=[2],
    )

    await worker.generate_sequences(chunk)
    await asyncio.gather(*worker.background_tasks)

    seeds = list(captured_seeds.values())
    assert set(captured_seeds) == {(uid, session) for uid in "ab" for session in range(3)}
    assert all(isinstance(seed, int) and 0 <= seed < (1 << 63) for seed in seeds)
    assert len(set(seeds)) == 6


@pytest.mark.asyncio
async def test_v0_worker_derives_validation_seeds_from_repeat_indices_across_chunks(monkeypatch):
    worker = object.__new__(DiffusionAgentLoopWorker)
    worker.rollout_config = SimpleNamespace(
        pipeline={},
        algo={},
        calculate_log_probs=False,
        agent=SimpleNamespace(default_agent_loop="diffusion_single_turn_agent"),
        val_kwargs=SimpleNamespace(n=3, seed=50, pipeline={}, algo={}),
    )

    captured: list[tuple[str, int | None]] = []

    async def fake_run_agent_loop(self, sampling_params, *, validate=False, uid=None, **kwargs):
        del self, validate, kwargs
        captured.append((uid, sampling_params.get("seed")))
        return SimpleNamespace()

    monkeypatch.setattr(diffusion_agent_loop, "_config_to_sampling_dict", lambda _cfg: {})
    worker._run_agent_loop = MethodType(fake_run_agent_loop, worker)
    worker._postprocess = lambda outputs, input_non_tensor_batch=None: outputs

    # Two prompts repeated 3 times, split unevenly across two worker chunks (2 + 4).
    chunk_0 = DataProto(
        non_tensor_batch={
            "uid": np.array(["p0", "p0"], dtype=object),
            "_val_repeat_idx": np.array([0, 1], dtype=np.int64),
        },
        meta_info={"validate": True},
    )
    chunk_1 = DataProto(
        non_tensor_batch={
            "uid": np.array(["p0", "p1", "p1", "p1"], dtype=object),
            "_val_repeat_idx": np.array([2, 0, 1, 2], dtype=np.int64),
        },
        meta_info={"validate": True},
    )

    await worker.generate_sequences(chunk_0)
    await worker.generate_sequences(chunk_1)

    assert captured == [
        ("p0", 50),
        ("p0", 51),
        ("p0", 52),
        ("p1", 50),
        ("p1", 51),
        ("p1", 52),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("val_seed", "expected_seeds"),
    [
        (42, [42, 42]),
        (None, [None, None]),
    ],
)
async def test_v0_worker_validation_without_repeat_idx_keeps_val_seed(monkeypatch, val_seed, expected_seeds):
    worker = object.__new__(DiffusionAgentLoopWorker)
    worker.rollout_config = SimpleNamespace(
        pipeline={},
        algo={},
        calculate_log_probs=False,
        agent=SimpleNamespace(default_agent_loop="diffusion_single_turn_agent"),
        val_kwargs=SimpleNamespace(n=1, seed=val_seed, pipeline={}, algo={}),
    )

    captured: list[int | None] = []

    async def fake_run_agent_loop(self, sampling_params, *, validate=False, **kwargs):
        del self, validate, kwargs
        captured.append(sampling_params.get("seed"))
        return SimpleNamespace()

    monkeypatch.setattr(diffusion_agent_loop, "_config_to_sampling_dict", lambda _cfg: {})
    worker._run_agent_loop = MethodType(fake_run_agent_loop, worker)
    worker._postprocess = lambda outputs, input_non_tensor_batch=None: outputs

    batch = DataProto(
        non_tensor_batch={"uid": np.array(["p0", "p1"], dtype=object)},
        meta_info={"validate": True},
    )

    await worker.generate_sequences(batch)

    assert captured == expected_seeds


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("val_n", "val_seed", "repeat_idx", "expected_ar", "expected_dit"),
    [
        (1, 42, None, [42], [[42]]),
        (3, 42, None, [42], [[42, 43, 44]]),
        (2, 42, [0, 1], [42, 43], [[42, 43], [44, 45]]),
        (1, None, None, [None], [None]),
    ],
)
async def test_composite_worker_derives_validation_seeds_per_repeat(
    monkeypatch,
    val_n,
    val_seed,
    repeat_idx,
    expected_ar,
    expected_dit,
):
    worker = object.__new__(CompositeAgentLoopWorker)
    ar_cfg = SimpleNamespace(top_p=1.0, top_k=0, temperature=0.0, repetition_penalty=1.0, response_length=32)
    worker.rollout_config = SimpleNamespace(
        n=2,
        pipeline={},
        algo={},
        ar={},
        calculate_log_probs=False,
        ar_calculate_log_probs=False,
        agent=SimpleNamespace(default_agent_loop="composite_agent"),
        val_kwargs=SimpleNamespace(n=val_n, seed=val_seed, pipeline={}, algo={}, ar=ar_cfg),
    )

    captured_ar: list[int | None] = []
    captured_dit: list[list[int] | None] = []

    async def fake_run_agent_loop(self, sampling_params, *, validate=False, per_rollout_seeds=None, **kwargs):
        del self, validate, kwargs
        captured_ar.append(sampling_params.get("seed"))
        captured_dit.append(per_rollout_seeds)
        return SimpleNamespace(), [SimpleNamespace() for _ in range(val_n)]

    monkeypatch.setattr(composite_agent_loop, "_config_to_sampling_dict", lambda _cfg: {})
    worker._run_agent_loop = MethodType(fake_run_agent_loop, worker)
    worker._postprocess_ar = lambda inputs, input_non_tensor_batch=None: inputs
    monkeypatch.setattr(
        DiffusionAgentLoopWorker,
        "_postprocess",
        lambda self, inputs, input_non_tensor_batch=None: inputs,
    )

    num_rows = len(repeat_idx) if repeat_idx is not None else 1
    non_tensor_batch = {"uid": np.array([f"p{i}" for i in range(num_rows)], dtype=object)}
    if repeat_idx is not None:
        non_tensor_batch["_val_repeat_idx"] = np.array(repeat_idx, dtype=np.int64)
    batch = DataProto(non_tensor_batch=non_tensor_batch, meta_info={"validate": True})

    await worker.generate_sequences(batch)

    assert captured_ar == expected_ar
    assert captured_dit == expected_dit


@pytest.mark.asyncio
async def test_composite_worker_unseeded_multi_repeat_validation_uses_dit_entropy_seeds(monkeypatch):
    worker = object.__new__(CompositeAgentLoopWorker)
    ar_cfg = SimpleNamespace(top_p=1.0, top_k=0, temperature=0.0, repetition_penalty=1.0, response_length=32)
    worker.rollout_config = SimpleNamespace(
        n=2,
        pipeline={},
        algo={},
        ar={},
        calculate_log_probs=False,
        ar_calculate_log_probs=False,
        agent=SimpleNamespace(default_agent_loop="composite_agent"),
        val_kwargs=SimpleNamespace(n=2, m=2, seed=None, pipeline={}, algo={}, ar=ar_cfg),
    )

    captured_ar: list[int | None] = []
    captured_dit: list[list[int] | None] = []

    async def fake_run_agent_loop(self, sampling_params, *, validate=False, per_rollout_seeds=None, **kwargs):
        del self, validate, kwargs
        captured_ar.append(sampling_params.get("seed"))
        captured_dit.append(per_rollout_seeds)
        return SimpleNamespace(), [SimpleNamespace() for _ in range(2)]

    monkeypatch.setattr(composite_agent_loop, "_config_to_sampling_dict", lambda _cfg: {})
    worker._run_agent_loop = MethodType(fake_run_agent_loop, worker)
    worker._postprocess_ar = lambda inputs, input_non_tensor_batch=None: inputs
    monkeypatch.setattr(
        DiffusionAgentLoopWorker,
        "_postprocess",
        lambda self, inputs, input_non_tensor_batch=None: inputs,
    )

    batch = DataProto(
        non_tensor_batch={
            "uid": np.array(["p0", "p0"], dtype=object),
            "_val_repeat_idx": np.array([0, 1], dtype=np.int64),
        },
        meta_info={"validate": True},
    )

    await worker.generate_sequences(batch)

    assert captured_ar == [None, None]
    flat_dit = [seed for row in captured_dit for seed in (row or [])]
    assert len(flat_dit) == 4
    assert all(isinstance(seed, int) and 0 <= seed < (1 << 63) for seed in flat_dit)
    assert len(set(flat_dit)) == 4
