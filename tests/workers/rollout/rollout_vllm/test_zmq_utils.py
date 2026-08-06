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

from types import SimpleNamespace
from unittest.mock import patch

from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

from verl_omni.workers.rollout.vllm_rollout.utils import vLLMOmniColocateWorkerExtension
from verl_omni.workers.rollout.vllm_rollout.zmq_utils import make_update_zmq_handle, make_update_zmq_id


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
                load_weights=lambda weights: None,
            )
            vLLMOmniColocateWorkerExtension.update_weights_from_ipc(
                worker,
                use_shm=True,
                zmq_update_id=update_id,
            )

    assert received_handles == [make_update_zmq_handle(handle, update_id) for handle in base_handles]
    assert len(set(received_handles)) == len(base_handles)
