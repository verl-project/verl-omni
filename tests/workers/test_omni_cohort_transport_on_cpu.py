# Copyright 2026 verl-omni contributors
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

"""Native metadata chunking and registered bridge with an in-memory storage client."""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("transfer_queue", reason="Native metadata tests require the optional TransferQueue runtime")

from verl.protocol import BatchData
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import tensordict_utils as tu
from verl.utils import transferqueue_utils as tqu
from verl.utils.transferqueue_utils import BatchMeta, KVBatchMeta

from verl_omni.workers.utils.cohort_transport import cohort_transport

IDENTITY = "_proximal_transport_identity"


@pytest.fixture
def storage(monkeypatch):
    keys = [f"row-{i}" for i in range(8)]
    indexes = [41, 13, 87, 25, 64, 92, 30, 18]
    key_to_index = dict(zip(keys, indexes, strict=True))
    index_to_key = dict(zip(indexes, keys, strict=True))

    class Client:
        async def async_kv_retrieve_meta(self, keys, partition_id, create):
            assert create is False
            return BatchMeta(
                global_indexes=[key_to_index[key] for key in keys],
                partition_ids=[partition_id] * len(keys),
                field_schema={"payload": {"dtype": torch.int64}},
                production_status=np.ones(len(keys), dtype=np.int8),
            )

        async def async_kv_retrieve_keys(self, global_indexes, partition_id):
            assert partition_id == "train"
            return [index_to_key[index] for index in global_indexes]

        async def async_get_data(self, meta):
            return TensorDict({"payload": torch.tensor(meta.global_indexes)}, batch_size=[meta.size])

    monkeypatch.setattr(tqu, "TQ_INITIALIZED", True)
    monkeypatch.setattr(tqu.tq, "get_client", lambda: Client())
    return keys, indexes


class Receiver:
    def __init__(self):
        self.calls = 0

    @cohort_transport
    @register(dispatch_mode=Dispatch.ALL_TO_ALL)
    def receive(self, data):
        self.calls += 1
        return {
            "identity": tu.get_non_tensor_data(data, IDENTITY, default=None),
            "payload": data["payload"].tolist(),
            "request": tu.get_non_tensor_data(data, "request", default=None),
        }


@pytest.mark.parametrize("chunks", [1, 2, 4])
def test_native_dispatch_binds_actual_storage_keys_and_preserves_bridge(storage, chunks):
    keys, indexes = storage
    original = KVBatchMeta(keys=keys, tags=[{} for _ in keys], partition_id="train", extra_info={"request": "nonce"})
    shards = BatchData(original).chunk(chunks=chunks)
    assert all(isinstance(shard, BatchMeta) for shard in shards)
    observed = []
    for shard in shards:
        receiver = Receiver()
        result = receiver.receive(shard)
        assert receiver.calls == 1
        assert result["identity"]["partition_id"] == "train"
        assert result["request"] == "nonce"
        assert result["payload"] == shard.global_indexes
        assert IDENTITY not in shard.extra_info
        observed.extend(result["identity"]["keys"])
    assert observed == keys
    assert [index for shard in shards for index in shard.global_indexes] == indexes
    assert IDENTITY not in original.extra_info


@pytest.mark.parametrize("carrier", ["kv", "native"])
def test_caller_cannot_supply_transport_identity(storage, carrier):
    keys, _ = storage
    meta = KVBatchMeta(keys=keys, tags=[{} for _ in keys], partition_id="train", extra_info={IDENTITY: "forged"})
    if carrier == "native":
        meta = BatchData(meta).chunk(chunks=1)[0]
    receiver = Receiver()
    with pytest.raises(ValueError, match="Caller-supplied"):
        receiver.receive(meta)
    assert receiver.calls == 0


def test_stored_transport_identity_is_rejected(storage):
    keys, _ = storage
    meta = KVBatchMeta(keys=keys, tags=[{} for _ in keys], partition_id="train", fields=[IDENTITY])
    receiver = Receiver()
    with pytest.raises(ValueError, match="Caller-supplied"):
        receiver.receive(meta)
    assert receiver.calls == 0


@pytest.mark.parametrize("partitions,indexes", [([], []), (["train", "other"], [41, 13])])
def test_empty_or_mixed_native_partition_is_rejected(storage, partitions, indexes):
    meta = BatchMeta(global_indexes=indexes, partition_ids=partitions)
    receiver = Receiver()
    with pytest.raises(ValueError, match="single-partition"):
        receiver.receive(meta)
    assert receiver.calls == 0


def test_materialized_tensor_cannot_spoof_dispatched_metadata(storage):
    receiver = Receiver()
    with pytest.raises(TypeError, match="dispatched batch metadata"):
        receiver.receive(TensorDict({"payload": torch.ones(8)}, batch_size=[8]))
    assert receiver.calls == 0
