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
"""CPU contracts for reward-replica dispatch."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch
from verl.protocol import DataProto

from verl_omni.reward_loop.replica_dispatch import dispatch_reward_groups


def _data(size: int) -> DataProto:
    return DataProto.from_dict(tensors={"sample_id": torch.arange(size, dtype=torch.int64)})


def _sample_ids(data: DataProto) -> list[int]:
    return data.batch["sample_id"].tolist()


def _worker(compute):
    return SimpleNamespace(compute_score_batch=SimpleNamespace(remote=compute))


def _recording_worker(calls: list[list[int]]):
    async def compute(data):
        sample_ids = _sample_ids(data)
        calls.append(sample_ids)
        return sample_ids

    return _worker(compute)


@pytest.mark.asyncio
@pytest.mark.parametrize(("size", "replicas"), [(0, 3), (1, 3), (3, 5), (7, 4)])
async def test_static_dispatch_preserves_padded_one_chunk_per_replica(size, replicas):
    calls = [[] for _ in range(replicas)]
    workers = [_recording_worker(worker_calls) for worker_calls in calls]

    result = await dispatch_reward_groups(_data(size), {"static": workers}, {})

    assert result == {"static": list(range(size))}
    if size == 0:
        assert calls == [[] for _ in range(replicas)]
        return

    assert all(len(worker_calls) == 1 for worker_calls in calls)
    chunk_sizes = {len(worker_calls[0]) for worker_calls in calls}
    assert len(chunk_sizes) == 1
    assert sum(len(worker_calls[0]) for worker_calls in calls) >= size
    if size % replicas:
        assert sum(len(worker_calls[0]) for worker_calls in calls) > size


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [0, 1, 3, 7])
async def test_dynamic_dispatch_scores_each_sample_exactly_once_without_padding(size):
    calls = [[] for _ in range(4)]
    workers = [_recording_worker(worker_calls) for worker_calls in calls]

    result = await dispatch_reward_groups(_data(size), {"native": workers}, {"native": 2})

    dispatched = [sample_id for worker_calls in calls for batch in worker_calls for sample_id in batch]
    assert result == {"native": list(range(size))}
    assert sorted(dispatched) == list(range(size))
    assert len(dispatched) == size
    assert all(0 < len(batch) <= 2 for worker_calls in calls for batch in worker_calls)


@pytest.mark.asyncio
async def test_dynamic_dispatch_replenishes_the_first_free_replica_with_next_contiguous_slice():
    class GatedWorker:
        def __init__(self):
            self.calls = []
            self.entered = asyncio.Event()
            self.second_entered = asyncio.Event()
            self.release = asyncio.Event()
            self.active = 0

            async def compute(data):
                assert self.active == 0
                self.active += 1
                sample_ids = _sample_ids(data)
                self.calls.append(sample_ids)
                if len(self.calls) == 1:
                    self.entered.set()
                else:
                    self.second_entered.set()
                await self.release.wait()
                self.active -= 1
                return sample_ids

            self.compute_score_batch = SimpleNamespace(remote=compute)

    workers = [GatedWorker() for _ in range(3)]
    task = asyncio.create_task(dispatch_reward_groups(_data(7), {"native": workers}, {"native": 2}))
    await asyncio.gather(*(worker.entered.wait() for worker in workers))

    assert [worker.calls for worker in workers] == [[[0, 1]], [[2, 3]], [[4, 5]]]
    workers[1].release.set()
    await workers[1].second_entered.wait()
    assert workers[1].calls == [[2, 3], [6]]
    assert workers[0].calls == [[0, 1]]
    assert workers[2].calls == [[4, 5]]

    workers[0].release.set()
    workers[2].release.set()
    assert await task == {"native": list(range(7))}
    assert all(worker.active == 0 for worker in workers)


@pytest.mark.asyncio
async def test_mixed_static_and_dynamic_groups_keep_independent_dispatch_contracts():
    static_calls = [[], []]
    dynamic_calls = [[], []]
    groups = {
        "shared": [_recording_worker(calls) for calls in static_calls],
        "native": [_recording_worker(calls) for calls in dynamic_calls],
    }

    result = await dispatch_reward_groups(_data(3), groups, {"native": 1})

    assert result == {"shared": [0, 1, 2], "native": [0, 1, 2]}
    assert all(len(calls) == 1 for calls in static_calls)
    assert sum(len(calls[0]) for calls in static_calls) == 4
    dynamic_ids = [sample_id for calls in dynamic_calls for batch in calls for sample_id in batch]
    assert sorted(dynamic_ids) == [0, 1, 2]
    assert all(len(batch) == 1 for calls in dynamic_calls for batch in calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, "2"])
async def test_invalid_dynamic_batch_size_fails_before_any_remote_send(batch_size):
    calls = []
    groups = {
        "valid": [_recording_worker(calls)],
        "invalid": [_recording_worker(calls)],
    }

    with pytest.raises(ValueError, match="dispatch_batch_size must be a positive integer"):
        await dispatch_reward_groups(_data(2), groups, {"invalid": batch_size})

    assert calls == []


@pytest.mark.asyncio
async def test_empty_group_fails_before_any_remote_send():
    calls = []

    with pytest.raises(ValueError, match="Reward worker group 'empty' has no replicas"):
        await dispatch_reward_groups(
            _data(2),
            {"valid": [_recording_worker(calls)], "empty": []},
            {},
        )

    assert calls == []


@pytest.mark.asyncio
async def test_result_cardinality_mismatch_is_not_retried():
    calls = []

    async def compute(data):
        sample_ids = _sample_ids(data)
        calls.append(sample_ids)
        return sample_ids[:-1]

    with pytest.raises(ValueError, match="returned 1 results for 2 samples"):
        await dispatch_reward_groups(_data(3), {"native": [_worker(compute)]}, {"native": 2})

    assert calls == [[0, 1]]


@pytest.mark.asyncio
async def test_worker_failure_stops_all_groups_and_drains_accepted_remote_calls():
    peer_entered = asyncio.Event()
    release_peer = asyncio.Event()
    failure_calls = []
    peer_calls = []

    async def fail(data):
        failure_calls.append(_sample_ids(data))
        await peer_entered.wait()
        raise RuntimeError("reward failed")

    async def peer(data):
        peer_calls.append(_sample_ids(data))
        peer_entered.set()
        await release_peer.wait()
        return _sample_ids(data)

    task = asyncio.create_task(
        dispatch_reward_groups(
            _data(4),
            {"failing": [_worker(fail)], "peer": [_worker(peer)]},
            {"failing": 1, "peer": 1},
        )
    )
    await peer_entered.wait()
    await asyncio.sleep(0)

    assert not task.done()
    release_peer.set()
    with pytest.raises(RuntimeError, match="reward failed"):
        await task

    assert failure_calls == [[0]]
    assert peer_calls == [[0]]


@pytest.mark.asyncio
async def test_synchronous_remote_failure_drains_an_active_peer_without_new_dispatch():
    peer_entered = asyncio.Event()
    release_peer = asyncio.Event()
    peer_calls = []
    failure_calls = []

    async def peer(data):
        peer_calls.append(_sample_ids(data))
        peer_entered.set()
        await release_peer.wait()
        return _sample_ids(data)

    class FailingRemote:
        def remote(self, data):
            assert peer_entered.is_set()
            failure_calls.append(_sample_ids(data))
            raise RuntimeError("remote submission failed")

    task = asyncio.create_task(
        dispatch_reward_groups(
            _data(4),
            {
                "peer": [_worker(peer)],
                "failing": [SimpleNamespace(compute_score_batch=FailingRemote())],
            },
            {"peer": 1, "failing": 1},
        )
    )
    await peer_entered.wait()
    await asyncio.sleep(0)
    assert not task.done()

    release_peer.set()
    with pytest.raises(RuntimeError, match="remote submission failed"):
        await task
    assert peer_calls == [[0]]
    assert failure_calls == [[0]]


def test_dispatch_state_is_local_to_each_asyncio_run_call():
    calls = []
    worker = _recording_worker(calls)

    first = asyncio.run(dispatch_reward_groups(_data(3), {"native": [worker]}, {"native": 2}))
    second = asyncio.run(dispatch_reward_groups(_data(1), {"native": [worker]}, {"native": 2}))

    assert first == {"native": [0, 1, 2]}
    assert second == {"native": [0]}
    assert calls == [[0, 1], [2], [0]]
