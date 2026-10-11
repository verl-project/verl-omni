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

import asyncio
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch
from tensordict import NonTensorData, NonTensorStack, TensorDict
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics
from verl.protocol import DataProto
from verl.utils import tensordict_utils as tu

from verl_omni.agent_loop import diffusion_agent_loop_tq
from verl_omni.agent_loop.diffusion_agent_loop import (
    DiffusionAgentLoopOutput,
    DiffusionAgentLoopWorker,
    _pad_prompt_extra_field,
    _pad_reference_rows,
)
from verl_omni.agent_loop.diffusion_agent_loop_tq import (
    DiffusionAgentLoopManagerTQ,
    DiffusionAgentLoopWorkerTQ,
    create_diffusion_agent_loop_manager,
)
from verl_omni.agent_loop.single_turn_agent_loop import DiffusionSingleTurnAgentLoop
from verl_omni.reward_loop.streaming import StreamingRewardClient
from verl_omni.trainer.diffusion.v1 import tq_utils
from verl_omni.workers.config.reward import StreamingRewardConfig


class _FakeRemoteComputeScore:
    def __init__(self):
        self.received_data: DataProto | None = None

    async def remote(self, data: DataProto) -> dict:
        self.received_data = data
        return {"reward_score": 1.0, "reward_extra_info": {}}


@pytest.mark.asyncio
@pytest.mark.parametrize("create", [create_diffusion_agent_loop_manager, DiffusionAgentLoopManagerTQ.create])
async def test_factory_and_class_creation_use_diffusion_tq_workers(monkeypatch, create):
    def initialize(self):
        self.agent_loop_workers_class = object

    async def init_workers(self):
        assert self.agent_loop_workers_class is DiffusionAgentLoopWorkerTQ
        self.agent_loop_workers = ["diffusion_worker"]

    monkeypatch.setattr(diffusion_agent_loop_tq.AgentLoopManagerTQ, "__init__", initialize)
    monkeypatch.setattr(DiffusionAgentLoopManagerTQ, "_init_agent_loop_workers", init_workers)
    manager = await create()
    assert manager.agent_loop_workers == ["diffusion_worker"]


class _FakeRewardLoopWorkerHandle:
    def __init__(self):
        self.compute_score = _FakeRemoteComputeScore()


class _DummyDiffusionAgentLoopWorker:
    _compute_score = DiffusionAgentLoopWorker._compute_score

    def __init__(self, reward_loop_worker_handle: _FakeRewardLoopWorkerHandle):
        self.reward_loop_worker_handles = [reward_loop_worker_handle]


@pytest.mark.parametrize(
    ("cache_enabled", "affinity_enabled", "expected_request_id"),
    [
        (True, True, "sample-uid"),
        (True, False, None),
        (False, True, None),
    ],
)
def test_prompt_cache_routing_affinity(cache_enabled, affinity_enabled, expected_request_id):
    agent_loop = object.__new__(DiffusionSingleTurnAgentLoop)
    agent_loop.rollout_config = SimpleNamespace(
        enable_prompt_embed_cache=cache_enabled,
        enable_prompt_embed_cache_routing_affinity=affinity_enabled,
    )

    request_id = agent_loop._get_routing_request_id("sample-uid")

    if expected_request_id is None:
        assert request_id != "sample-uid"
    else:
        assert request_id == expected_request_id


def test_prompt_cache_routing_affinity_requires_sample_uid():
    agent_loop = object.__new__(DiffusionSingleTurnAgentLoop)
    agent_loop.rollout_config = SimpleNamespace(
        enable_prompt_embed_cache=True,
        enable_prompt_embed_cache_routing_affinity=True,
    )

    first_request_id = agent_loop._get_routing_request_id(None)
    second_request_id = agent_loop._get_routing_request_id(None)

    assert first_request_id != second_request_id


@pytest.mark.asyncio
async def test_single_turn_agent_forwards_all_multimodal_inputs():
    agent_loop = object.__new__(DiffusionSingleTurnAgentLoop)
    agent_loop.rollout_config = SimpleNamespace(
        enable_prompt_embed_cache=False,
        enable_prompt_embed_cache_routing_affinity=False,
    )
    agent_loop.extra_tokenizer_map = {}
    agent_loop.mm_processor_kwargs = {"fps": 24}
    agent_loop.processor = None
    agent_loop.process_multi_modal_info = AsyncMock(
        return_value={"images": ["image"], "videos": ["video"], "audios": ["audio"]}
    )
    agent_loop.ct_build_initial_tokens = AsyncMock(return_value=[1, 2, 3])
    agent_loop._assert_mm_supported = lambda _: None
    agent_loop.server_manager = SimpleNamespace(
        generate=AsyncMock(
            return_value=SimpleNamespace(
                diffusion_output=torch.zeros(1),
                log_probs=None,
                num_preempted=None,
                extra_fields={},
            )
        )
    )

    await agent_loop.run({}, raw_prompt=[{"role": "user", "content": "prompt"}])

    call = agent_loop.server_manager.generate.await_args.kwargs
    assert call["image_data"] == ["image"]
    assert call["video_data"] == ["video"]
    assert call["audio_data"] == ["audio"]
    assert call["mm_processor_kwargs"] == {"fps": 24}


@pytest.mark.asyncio
async def test_single_turn_agent_accepts_numpy_negative_prompt():
    agent_loop = object.__new__(DiffusionSingleTurnAgentLoop)
    agent_loop.rollout_config = SimpleNamespace(
        enable_prompt_embed_cache=False,
        enable_prompt_embed_cache_routing_affinity=False,
    )
    agent_loop.extra_tokenizer_map = {"text_encoder": object()}
    agent_loop.mm_processor_kwargs = {}
    agent_loop.processor = None
    agent_loop.process_multi_modal_info = AsyncMock(return_value={})
    agent_loop.ct_build_initial_tokens = AsyncMock(return_value=[1, 2, 3])
    agent_loop._tokenize_per_encoder = AsyncMock(return_value={"text_encoder": [1, 2, 3]})
    agent_loop._assert_mm_supported = lambda _: None
    agent_loop.server_manager = SimpleNamespace(
        generate=AsyncMock(
            return_value=SimpleNamespace(
                diffusion_output=torch.zeros(1),
                log_probs=None,
                num_preempted=None,
                extra_fields={},
            )
        )
    )
    raw_prompt = [{"role": "user", "content": "prompt"}]
    raw_negative_prompt = np.array(
        [
            {"role": "system", "content": "negative"},
            {"role": "user", "content": "blurry"},
        ]
    )

    await agent_loop.run({}, raw_prompt=raw_prompt, raw_negative_prompt=raw_negative_prompt)

    assert agent_loop.ct_build_initial_tokens.await_count == 2
    assert agent_loop.ct_build_initial_tokens.await_args_list[1].args[0] is raw_negative_prompt
    assert agent_loop._tokenize_per_encoder.await_count == 2
    assert agent_loop._tokenize_per_encoder.await_args_list[1].args[0] is raw_negative_prompt


@pytest.mark.parametrize(
    ("key", "value", "expected_shape"),
    [
        ("prompt_embeds", torch.ones(2, 4), (3, 4)),
        ("negative_prompt_embeds", torch.ones(2, 4), (3, 4)),
        ("prompt_embeds_mask", torch.ones(2), (3,)),
        ("negative_prompt_embeds_mask", torch.ones(2), (3,)),
    ],
)
def test_pad_prompt_extra_field_pads_without_truncation(key, value, expected_shape):
    padded = _pad_prompt_extra_field(key, value, target_length=3)

    assert padded.shape == expected_shape
    assert torch.equal(padded[:2], value)
    assert torch.count_nonzero(padded[2:]) == 0


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("prompt_embeds", torch.ones(4, 2)),
        ("prompt_embeds_mask", torch.ones(4)),
    ],
)
def test_pad_prompt_extra_field_rejects_truncation(key, value):
    with pytest.raises(ValueError, match="exceeds max_prompt_embed_length=3"):
        _pad_prompt_extra_field(key, value, target_length=3)


def test_reference_rows_are_padded_to_the_global_limit():
    values = [torch.ones(1, 2, 96), torch.ones(1, 5, 96)]
    counts = [torch.tensor([[2]]), torch.tensor([[5]])]

    padded, masks = _pad_reference_rows("condition_video_rows", values, counts, target_length=7)

    assert [value.shape for value in padded] == [(1, 7, 96), (1, 7, 96)]
    assert [mask.sum().item() for mask in masks] == [2, 5]
    assert torch.count_nonzero(padded[0][:, 2:]) == 0
    assert torch.count_nonzero(padded[1][:, 5:]) == 0

    outputs = [
        DataProto(
            batch=TensorDict(
                {
                    "condition_video_rows": value,
                    "condition_video_rows_mask": mask,
                    "condition_video_row_count": count,
                },
                batch_size=1,
            )
        )
        for value, mask, count in zip(padded, masks, counts, strict=True)
    ]
    combined = DataProto.concat(outputs)
    assert combined.batch["condition_video_rows"].shape == (2, 7, 96)
    assert combined.batch["condition_video_rows_mask"].sum(dim=1).tolist() == [2, 5]


@pytest.mark.parametrize(
    ("value", "count", "target_length", "message"),
    [
        (torch.ones(1, 2, 96), torch.tensor([[1]]), 4, "row count 1 does not match tensor rows 2"),
        (torch.ones(1, 5, 96), torch.tensor([[5]]), 4, "exceeding max_prompt_embed_length=4"),
        (torch.ones(2, 2, 96), torch.tensor([[2]]), 4, "must have shape \\[1, rows, width\\]"),
    ],
)
def test_reference_row_padding_rejects_invalid_inputs(value, count, target_length, message):
    with pytest.raises(ValueError, match=message):
        _pad_reference_rows("condition_video_rows", [value], [count], target_length)


@pytest.mark.asyncio
async def test_tq_writer_preserves_allowlisted_non_tensor_trajectory_metadata(monkeypatch):
    worker_cls = DiffusionAgentLoopWorkerTQ.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    captured = {}
    img_shapes = [(1, 32, 32), (1, 64, 64)]
    internal = SimpleNamespace(
        prompt_ids=torch.tensor([[1, 2]]),
        response_diffusion_output=torch.zeros(1, 3, 2, 2),
        response_logprobs=None,
        reward_score=None,
        num_turns=2,
        extra_fields={
            "condition_image_latents": torch.zeros(1, 4096, 64),
            "audio": torch.zeros(1, 1, 16),
            "media_kind": "video",
            "img_shapes": img_shapes,
            "audio_sample_rate": 32_000,
            "unrelated_metadata": "do-not-forward",
        },
    )

    monkeypatch.setattr(diffusion_agent_loop_tq, "list_of_dict_to_tensordict", lambda rows: rows)

    async def fake_kv_batch_put(*, keys, fields, tags, partition_id):
        captured.update(keys=keys, fields=fields, tags=tags, partition_id=partition_id)

    monkeypatch.setattr(diffusion_agent_loop_tq.tq, "async_kv_batch_put", fake_kv_batch_put)

    await worker._write_trajectories_to_tq(
        [(0, internal)],
        uid="sample",
        trajectory={"step": 3},
        validate=False,
    )

    field = captured["fields"][0]
    assert field["extra_fields"] == {
        "img_shapes": img_shapes,
        "media_kind": "video",
        "audio_sample_rate": 32_000,
        "min_global_steps": 3,
        "max_global_steps": 3,
    }
    assert "unrelated_metadata" not in field["extra_fields"]
    assert field["condition_image_latents"].shape == (4096, 64)
    assert field["audio"].shape == (1, 16)
    assert captured["tags"][0]["response_shape"] == (3, 2, 2)


@pytest.mark.asyncio
async def test_tq_writer_batches_group_sessions_and_partitions_field_sets(monkeypatch):
    worker_cls = DiffusionAgentLoopWorkerTQ.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    puts = []

    def make_internal(with_reward: bool):
        internal = SimpleNamespace(
            prompt_ids=torch.tensor([[1, 2]]),
            response_diffusion_output=torch.zeros(1, 3, 2, 2),
            response_logprobs=None,
            reward_score=0.5 if with_reward else None,
            num_turns=1,
            extra_fields={},
        )
        return internal

    monkeypatch.setattr(diffusion_agent_loop_tq, "list_of_dict_to_tensordict", lambda rows: rows)

    async def fake_kv_batch_put(*, keys, fields, tags, partition_id):
        puts.append({"keys": keys, "fields": fields, "tags": tags, "partition_id": partition_id})

    monkeypatch.setattr(diffusion_agent_loop_tq.tq, "async_kv_batch_put", fake_kv_batch_put)

    # Session 1 carries rm_scores, sessions 0/2 do not: rows split by field
    # signature instead of crashing on missing keys.
    await worker._write_trajectories_to_tq(
        [(0, make_internal(False)), (1, make_internal(True)), (2, make_internal(False))],
        uid="sample",
        trajectory={"step": 7},
        validate=False,
        index=3,
        gen_batch_seq=5,
    )

    assert [put["keys"] for put in puts] == [["sample_0_0", "sample_2_0"], ["sample_1_0"]]
    assert "rm_scores" in puts[1]["fields"][0]
    assert all("rm_scores" not in put["fields"][0] for put in puts[:1])
    assert all(tag["global_steps"] == 7 for put in puts for tag in put["tags"])
    # The prompt identity rides in the tags so the trainer can restore the
    # v0 prompt-major row order: batch-local position plus the per-run
    # generation-batch number.
    assert all(tag["prompt_index"] == 3 for put in puts for tag in put["tags"])
    assert all(tag["gen_batch_seq"] == 5 for put in puts for tag in put["tags"])


def test_tq_batch_restores_non_tensor_trajectory_metadata(monkeypatch):
    img_shapes = [
        [(1, 32, 32), (1, 64, 64)],
        [(1, 32, 32), (1, 64, 64)],
    ]

    monkeypatch.setattr(
        tq_utils.tq,
        "kv_batch_get",
        lambda **kwargs: {
            "all_latents": torch.zeros(2, 4, 1024, 64),
            "extra_fields": [
                {"img_shapes": img_shapes[0], "media_kind": "video", "audio_sample_rate": 32_000},
                {"img_shapes": img_shapes[1]},
            ],
        },
    )

    data = tq_utils.diffusion_tq_batch_to_dataproto(
        SimpleNamespace(keys=["sample_0", "sample_1"], partition_id="train")
    )

    assert data.non_tensor_batch["img_shapes"].tolist() == img_shapes
    assert data.non_tensor_batch["media_kind"].tolist() == ["video", None]
    assert data.non_tensor_batch["audio_sample_rate"].tolist() == [32_000, None]
    assert tu.get(data.to_tensordict(), "img_shapes") == img_shapes


@pytest.mark.asyncio
async def test_run_prompt_publishes_failure_after_siblings_settle(monkeypatch):
    worker_cls = DiffusionAgentLoopWorkerTQ.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    worker.rollout_config = SimpleNamespace(n=2, val_kwargs=SimpleNamespace(n=2))
    lifecycle = []

    async def fake_kv_put(*, key, partition_id, tag):
        assert key == "sample"
        assert partition_id == "train"
        lifecycle.append(tag["status"])

    async def fake_run_agent_loop(self, sampling_params, *, session_id, **kwargs):
        del self, sampling_params, kwargs
        if session_id == 0:
            raise RuntimeError("session failed")
        await asyncio.sleep(0.01)
        lifecycle.append("sibling_settled")

    monkeypatch.setattr(diffusion_agent_loop_tq.tq, "async_kv_put", fake_kv_put)
    worker._run_agent_loop = MethodType(fake_run_agent_loop, worker)

    written = []

    async def fake_write(outputs, **kwargs):
        del kwargs
        lifecycle.append("written")
        written.append(outputs)

    worker._write_trajectories_to_tq = fake_write

    await worker._run_prompt(
        prompt={"uid": "sample", "agent_name": "diffusion_single_turn_agent"},
        sampling_params={},
        trajectory={"validate": False},
        prompt_index=0,
    )

    assert lifecycle == ["running", "sibling_settled", "written", "failure"]
    assert [session_id for session_id, _output in written[0]] == [1]


@pytest.mark.asyncio
async def test_generate_sequences_seeds_from_global_prompt_index(monkeypatch):
    """Per-request rollout seeds must derive from the global batch index (#561).

    Each agent worker only sees a chunk of the batch, so seeding from the
    chunk-local position makes every worker reuse the same seed offsets and
    roll out duplicated noise. Two chunks carrying global indices [0, 1] and
    [2, 3] must yield distinct seeds for all prompt x session combinations.
    """
    worker_cls = DiffusionAgentLoopWorkerTQ.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    worker.background_tasks = set()
    worker.rollout_config = SimpleNamespace(
        n=2,
        pipeline={},
        algo={},
        calculate_log_probs=False,
        agent=SimpleNamespace(default_agent_loop="diffusion_single_turn_agent"),
        val_kwargs=SimpleNamespace(n=2, seed=0, pipeline={}, algo={}),
    )

    async def fake_kv_put(*, key, partition_id, tag):
        del key, partition_id, tag

    captured_seeds: list[int] = []

    async def fake_run_agent_loop(self, sampling_params, *, session_id, **kwargs):
        del self, session_id, kwargs
        captured_seeds.append(sampling_params["seed"])

    monkeypatch.setattr(diffusion_agent_loop_tq, "_config_to_sampling_dict", lambda cfg: {})
    monkeypatch.setattr(diffusion_agent_loop_tq.tq, "async_kv_put", fake_kv_put)
    worker._run_agent_loop = MethodType(fake_run_agent_loop, worker)

    def make_chunk(global_indices: list[int], uids: list[str]) -> TensorDict:
        return TensorDict(
            {
                "index": torch.tensor(global_indices),
                "uid": NonTensorStack(*uids),
                "rollout_seed": NonTensorData(42),
                "global_steps": NonTensorData(1),
            },
            batch_size=[len(global_indices)],
        )

    # Two chunks as AgentLoopManagerTQ would split a 4-prompt batch across two
    # workers: chunk-local positions restart at 0, global indices do not.
    await worker.generate_sequences(make_chunk([0, 1], ["a", "b"]))
    await worker.generate_sequences(make_chunk([2, 3], ["c", "d"]))
    await asyncio.gather(*worker.background_tasks)

    assert len(captured_seeds) == 4 * 2  # 4 prompts x rollout.n=2
    assert len(set(captured_seeds)) == 4 * 2  # no duplicated noise across workers


@pytest.mark.asyncio
@pytest.mark.parametrize("validate", [False, True])
async def test_async_reward_data_proto_preserves_validate_meta_info(validate: bool):
    reward_loop_worker_handle = _FakeRewardLoopWorkerHandle()
    worker = _DummyDiffusionAgentLoopWorker(reward_loop_worker_handle)
    output = DiffusionAgentLoopOutput(
        prompt_ids=[1, 2],
        response_diffusion_output=torch.zeros(3, 2, 2, dtype=torch.uint8),
        metrics=AgentLoopMetrics(),
    )

    await worker._compute_score(
        output,
        prompts=torch.tensor([[1, 2]]),
        responses=torch.zeros(1, 3, 2, 2, dtype=torch.uint8),
        kwargs={},
        validate=validate,
    )

    received_data = reward_loop_worker_handle.compute_score.received_data
    assert received_data is not None
    assert received_data.meta_info == {"validate": validate}


@pytest.mark.asyncio
async def test_named_reward_admission_bounds_generation_and_overlaps_scoring(monkeypatch):
    slow_started = asyncio.Event()
    release_slow = asyncio.Event()
    fast_scoring = asyncio.Event()
    release_fast_score = asyncio.Event()
    generated = []
    scored = []

    async def generate(sampling_params, uid, **kwargs):
        generated.append(uid)
        if uid == "slow":
            slow_started.set()
            await release_slow.wait()
        else:
            await slow_started.wait()
        return DiffusionAgentLoopOutput(
            prompt_ids=[1, 2],
            response_diffusion_output=torch.zeros(3, 2, 2, dtype=torch.uint8),
            metrics=AgentLoopMetrics(),
        )

    async def score(data):
        uid = data.non_tensor_batch["uid"].item()
        scored.append(uid)
        if uid == "fast":
            fast_scoring.set()
            await release_fast_score.wait()
        return {"reward_score": 1.0, "reward_extra_info": {"reward/quality": 1.0}}

    handles = {
        "quality": (SimpleNamespace(compute_score=SimpleNamespace(remote=lambda d: asyncio.create_task(score(d)))),)
    }
    worker = object.__new__(DiffusionAgentLoopWorker)
    worker.config = SimpleNamespace(data={})
    worker.server_manager = worker.processor = worker.dataset_cls = worker.hf_model_type = None
    worker.model_config = SimpleNamespace(extra_tokenizer_map={})
    worker.rollout_config = SimpleNamespace(prompt_length=2)
    worker.tokenizer = SimpleNamespace(
        pad=lambda *args, **kwargs: {"input_ids": torch.tensor([1, 2]), "attention_mask": torch.ones(2)}
    )
    worker.reward_loop_worker_handles = handles
    worker.streaming_reward_client = StreamingRewardClient(handles, StreamingRewardConfig(max_inflight=2))
    monkeypatch.setattr(
        "verl_omni.agent_loop.diffusion_agent_loop.hydra.utils.instantiate",
        lambda **kwargs: SimpleNamespace(run=generate),
    )
    fast, slow, queued = [
        asyncio.create_task(
            worker._run_agent_loop({}, agent_name="diffusion_single_turn_agent", uid=uid, raw_prompt="prompt")
        )
        for uid in ("fast", "slow", "queued")
    ]
    try:
        async with asyncio.timeout(5):
            await fast_scoring.wait()
            assert generated == ["fast", "slow"]
            assert not queued.done()
            release_fast_score.set()
            assert (await fast).reward_score == 1.0
            assert slow_started.is_set() and not slow.done()
            assert (await queued).reward_score == 1.0
            assert generated == ["fast", "slow", "queued"]
            assert scored == ["fast", "queued"]
            release_slow.set()
            assert (await slow).reward_score == 1.0
            assert scored == ["fast", "queued", "slow"]
    finally:
        release_fast_score.set()
        release_slow.set()
        for task in (fast, slow, queued):
            task.cancel()
        await asyncio.gather(fast, slow, queued, return_exceptions=True)
        await worker.close_reward_streaming()


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_kind", ["success", "cancel", "timeout"])
async def test_named_reward_agent_publishes_only_live_results(exit_kind):
    entered = asyncio.Event()
    release = asyncio.Event()
    received = []

    async def score(data):
        received.append(data)
        entered.set()
        await release.wait()
        return {"reward_score": 2.0, "reward_extra_info": {"reward/quality": 2.0}}

    handles = {
        "quality": (SimpleNamespace(compute_score=SimpleNamespace(remote=lambda d: asyncio.create_task(score(d)))),)
    }
    worker = object.__new__(DiffusionAgentLoopWorker)
    worker.reward_loop_worker_handles = handles
    worker.streaming_reward_client = StreamingRewardClient(
        handles, StreamingRewardConfig(timeout=0.02 if exit_kind == "timeout" else None)
    )
    output = DiffusionAgentLoopOutput(
        prompt_ids=[1, 2],
        response_diffusion_output=torch.zeros(3, 2, 2, dtype=torch.uint8),
        metrics=AgentLoopMetrics(),
    )

    async def run():
        async with worker.streaming_reward_client.sample_slot():
            await worker._compute_score(
                output,
                prompts=torch.tensor([[1, 2]]),
                responses=torch.zeros(1, 3, 2, 2, dtype=torch.uint8),
                kwargs={"uid": "sample-a"},
                validate=True,
            )

    task = asyncio.create_task(run())
    await entered.wait()
    if exit_kind == "cancel":
        task.cancel()
    elif exit_kind == "timeout":
        await asyncio.sleep(0.04)
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    if exit_kind == "success":
        await task
        assert output.reward_score == 2.0
        assert output.extra_fields["reward_extra_info"]["reward/combined"] == 2.0
    else:
        expected = asyncio.CancelledError if exit_kind == "cancel" else TimeoutError
        with pytest.raises(expected):
            await task
        assert output.reward_score is None
        assert "reward_extra_info" not in output.extra_fields
    assert received[0].meta_info == {"validate": True}
    assert received[0].non_tensor_batch["uid"].tolist() == ["sample-a"]


@pytest.mark.asyncio
async def test_tq_close_cancels_prompts_and_drains_accepted_reward():
    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    published = []

    async def score(data):
        entered.set()
        await release.wait()
        finished.set()
        return {"reward_score": 1.0, "reward_extra_info": {}}

    handles = {
        "quality": (SimpleNamespace(compute_score=SimpleNamespace(remote=lambda d: asyncio.create_task(score(d)))),)
    }
    worker = object.__new__(DiffusionAgentLoopWorkerTQ.__ray_metadata__.modified_class)
    worker.streaming_reward_client = StreamingRewardClient(handles, StreamingRewardConfig(max_inflight=1))

    async def prompt():
        data = DataProto.from_dict(tensors={"responses": torch.zeros(1, 1)})
        published.append(await worker.streaming_reward_client.compute_score(data))

    first = asyncio.create_task(prompt())
    await entered.wait()
    queued = asyncio.create_task(prompt())
    worker.background_tasks = {first, queued}
    close = asyncio.create_task(worker.close_reward_streaming())
    await asyncio.sleep(0)
    assert not close.done()
    release.set()
    await close
    assert finished.is_set()
    assert first.cancelled() and queued.cancelled()
    assert not published
