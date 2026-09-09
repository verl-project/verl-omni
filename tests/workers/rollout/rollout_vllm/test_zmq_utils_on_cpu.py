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

import os
from types import SimpleNamespace
from unittest.mock import patch

from verl.utils.device import get_visible_devices_keyword
from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

from verl_omni.workers.rollout.vllm_rollout.utils import vLLMOmniColocateWorkerExtension
from verl_omni.workers.rollout.vllm_rollout.zmq_utils import make_update_zmq_handle, make_update_zmq_id

_VISIBLE_DEVICES = get_visible_devices_keyword()


def test_update_id_is_unique_when_the_same_step_is_retried():
    assert make_update_zmq_id(7, 2) == "step-7-seq-2"
    assert make_update_zmq_id(7, 2) != make_update_zmq_id(7, 3)


def test_shared_update_id_preserves_rank_specific_routes():
    update_id = "step-7-seq-2"
    rank_0 = "ipc:///tmp/rl-colocate-zmq-job-replica-0-rank-0.sock"
    rank_1 = "ipc:///tmp/rl-colocate-zmq-job-replica-0-rank-1.sock"

    rank_0_handle = make_update_zmq_handle(rank_0, update_id)
    rank_1_handle = make_update_zmq_handle(rank_1, update_id)

    assert rank_0_handle == "ipc:///tmp/rl-colocate-zmq-job-replica-0-rank-0-update-step-7-seq-2.sock"
    assert rank_1_handle == "ipc:///tmp/rl-colocate-zmq-job-replica-0-rank-1-update-step-7-seq-2.sock"
    assert rank_0_handle != rank_1_handle
    assert rank_0_handle != make_update_zmq_handle(rank_0, "step-7-seq-3")


def test_non_ipc_handle_is_unchanged():
    handle = "tcp://127.0.0.1:5555"

    assert make_update_zmq_handle(handle, "step-7-seq-2") == handle


def test_worker_receivers_resolve_shared_update_id_from_their_rank_local_handles():
    received_handles = []

    class FakeReceiver:
        def __init__(self, zmq_handle, device, use_shm):
            received_handles.append(zmq_handle)

        def receive_weights(self, on_bucket_received):
            on_bucket_received([])

    update_id = "step-7-seq-2"
    base_handles = [
        "ipc:///tmp/rl-colocate-zmq-job-replica-0-rank-0.sock",
        "ipc:///tmp/rl-colocate-zmq-job-replica-0-rank-1.sock",
    ]
    with patch.object(bucketed_weight_transfer, "BucketedWeightReceiver", FakeReceiver):
        for base_handle in base_handles:
            worker = SimpleNamespace(
                device="npu",
                _pending_lora_peft_config=None,
                _get_zmq_handle=lambda base_handle=base_handle: base_handle,
                _get_standard_weight_model_and_config=lambda: None,
                model_runner=SimpleNamespace(pipeline=SimpleNamespace(load_weights=lambda weights: None)),
            )
            vLLMOmniColocateWorkerExtension.update_weights_from_ipc(
                worker,
                use_shm=True,
                zmq_update_id=update_id,
            )

    assert received_handles == [make_update_zmq_handle(handle, update_id) for handle in base_handles]
    assert len(set(received_handles)) == len(base_handles)


def _get_handle(local_rank, env_extra, clear=False):
    worker = SimpleNamespace(local_rank=local_rank)
    env = {"VERL_RAY_JOB_ID": "job", "VERL_REPLICA_RANK": "0", **env_extra}
    with patch.dict(os.environ, env, clear=clear):
        return vLLMOmniColocateWorkerExtension._get_zmq_handle(worker)


def test_stage_local_rank_is_remapped_to_node_local_index():
    # DiT stage on GPUs 4-7 of an 8-GPU replica: stage-local rank 2 maps to
    # node-local rank 6.
    handle = _get_handle(
        2,
        {_VISIBLE_DEVICES: "4, 5, 6, 7", "VERL_ZMQ_BASE_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"},
    )

    assert handle == "ipc:///tmp/rl-colocate-zmq-job-replica-0-rank-6.sock"


def test_offset_device_ids_map_back_to_zero_based_local_rank():
    # Offset physical GPUs: stage-local rank remaps to the replica index.
    handle = _get_handle(0, {_VISIBLE_DEVICES: "4,5,6,7", "VERL_ZMQ_BASE_VISIBLE_DEVICES": "4,5,6,7"})

    assert handle == "ipc:///tmp/rl-colocate-zmq-job-replica-0-rank-0.sock"


def test_offset_replica_stage_uses_replica_index():
    # Replica on physical GPUs 8-15, stage on the second half: node-local rank
    # is the replica index (4).
    handle = _get_handle(0, {_VISIBLE_DEVICES: "12,13,14,15", "VERL_ZMQ_BASE_VISIBLE_DEVICES": "8,9,10,11,12,13,14,15"})

    assert handle == "ipc:///tmp/rl-colocate-zmq-job-replica-0-rank-4.sock"


def test_full_replica_visible_mapping_is_identity():
    handle = _get_handle(
        5,
        {_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7", "VERL_ZMQ_BASE_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"},
    )

    assert handle == "ipc:///tmp/rl-colocate-zmq-job-replica-0-rank-5.sock"


def test_raw_rank_is_kept_when_device_lists_are_absent():
    handle = _get_handle(3, {}, clear=True)

    assert handle == "ipc:///tmp/rl-colocate-zmq-job-replica-0-rank-3.sock"


def test_raw_rank_is_kept_when_only_replica_list_is_absent():
    handle = _get_handle(3, {_VISIBLE_DEVICES: "0,1,2,3"})

    assert handle == "ipc:///tmp/rl-colocate-zmq-job-replica-0-rank-3.sock"


def test_raw_rank_is_kept_when_rank_is_outside_stage_devices():
    handle = _get_handle(3, {_VISIBLE_DEVICES: "0,1", "VERL_ZMQ_BASE_VISIBLE_DEVICES": "0,1,2,3"})

    assert handle == "ipc:///tmp/rl-colocate-zmq-job-replica-0-rank-3.sock"


def test_raw_rank_is_kept_when_device_is_not_in_replica_list():
    # ROCm device UUIDs remap by string position; an entry absent from the
    # replica list falls back to the raw stage-local rank.
    handle = _get_handle(
        1,
        {_VISIBLE_DEVICES: "GPU-1234,GPU-5678", "VERL_ZMQ_BASE_VISIBLE_DEVICES": "GPU-1111,GPU-2222"},
    )

    assert handle == "ipc:///tmp/rl-colocate-zmq-job-replica-0-rank-1.sock"


def test_device_uuid_is_remapped_by_position_in_replica_list():
    handle = _get_handle(
        1,
        {
            _VISIBLE_DEVICES: "GPU-1234,GPU-5678",
            "VERL_ZMQ_BASE_VISIBLE_DEVICES": "GPU-1111,GPU-1234,GPU-5678,GPU-9999",
        },
    )

    assert handle == "ipc:///tmp/rl-colocate-zmq-job-replica-0-rank-2.sock"
