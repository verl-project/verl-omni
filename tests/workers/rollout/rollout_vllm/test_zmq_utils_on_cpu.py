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
import socket
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from verl.utils.device import get_visible_devices_keyword
from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

from verl_omni.workers.rollout.vllm_rollout.utils import vLLMOmniColocateWorkerExtension
from verl_omni.workers.rollout.vllm_rollout.zmq_utils import (
    make_base_zmq_handle,
    make_update_zmq_handle,
    make_update_zmq_id,
)

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


@pytest.fixture(autouse=True)
def _clear_zmq_directory(monkeypatch):
    monkeypatch.delenv("VERL_ZMQ_DIR", raising=False)


@pytest.fixture
def private_zmq_directory():
    with tempfile.TemporaryDirectory(prefix="vnp-", dir="/tmp") as directory:
        yield Path(directory).resolve()


@pytest.mark.parametrize("override", [None, ""])
def test_base_zmq_handle_keeps_legacy_default(monkeypatch, override):
    if override is not None:
        monkeypatch.setenv("VERL_ZMQ_DIR", override)
    assert make_base_zmq_handle("job", 2, 3) == "ipc:///tmp/rl-colocate-zmq-job-replica-2-rank-3.sock"


def test_base_zmq_handle_accepts_root_directory(monkeypatch):
    monkeypatch.setenv("VERL_ZMQ_DIR", "/")
    assert make_base_zmq_handle("job", 0, 0) == "ipc:///rl-colocate-zmq-job-replica-0-rank-0.sock"


def test_independent_directories_allow_identical_job_ids(monkeypatch, private_zmq_directory):
    with tempfile.TemporaryDirectory(prefix="vnp-", dir="/tmp") as second:
        sockets, routes = [], []
        try:
            for directory in (private_zmq_directory, Path(second).resolve()):
                monkeypatch.setenv("VERL_ZMQ_DIR", str(directory))
                base = make_base_zmq_handle("00000001", 0, 0)
                route = make_update_zmq_handle(base, make_update_zmq_id(1, 1))
                listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sockets.append(listener)
                listener.bind(route.removeprefix("ipc://"))
                routes.append(route)
            assert routes[0] != routes[1]
        finally:
            for listener in sockets:
                listener.close()


@pytest.mark.parametrize(
    "kind,suffix",
    [(kind, "") for kind in ("relative", "missing", "file")]
    + [("symlink", suffix) for suffix in ("", "/", "///", "/.", "/..", "/child")],
)
def test_base_zmq_handle_rejects_invalid_directory(monkeypatch, tmp_path, private_zmq_directory, kind, suffix):
    if kind == "relative":
        directory = "relative"
    elif kind == "missing":
        directory = str(tmp_path / "missing")
    elif kind == "file":
        path = tmp_path / "file"
        path.write_text("not a directory")
        directory = str(path)
    else:
        (private_zmq_directory / "child").mkdir()
        path = private_zmq_directory / "link"
        path.symlink_to(private_zmq_directory, target_is_directory=True)
        directory = str(path) + suffix
    monkeypatch.setenv("VERL_ZMQ_DIR", directory)
    with pytest.raises(ValueError, match="VERL_ZMQ_DIR"):
        make_base_zmq_handle("job", 0, 0)


def test_base_zmq_handle_checks_socket_path_bytes(monkeypatch, private_zmq_directory):
    monkeypatch.setenv("VERL_ZMQ_DIR", str(private_zmq_directory))
    fixed_size = len(os.fsencode(make_base_zmq_handle("", 0, 0).removeprefix("ipc://")))
    job_length = 107 - fixed_size
    assert job_length > 0
    handle = make_base_zmq_handle("j" * job_length, 0, 0)
    assert len(os.fsencode(handle.removeprefix("ipc://"))) == 107
    with pytest.raises(ValueError, match="shorter than 108 bytes"):
        make_base_zmq_handle("j" * (job_length + 1), 0, 0)
    with pytest.raises(ValueError, match="shorter than 108 bytes"):
        make_base_zmq_handle("é" * job_length, 0, 0)
    with pytest.raises(ValueError, match="no NUL"):
        make_base_zmq_handle("job\0", 0, 0)


def test_update_zmq_handle_validates_final_socket_path(monkeypatch, private_zmq_directory):
    monkeypatch.setenv("VERL_ZMQ_DIR", str(private_zmq_directory))
    fixed_size = len(os.fsencode(make_base_zmq_handle("", 0, 0).removeprefix("ipc://")))
    base = make_base_zmq_handle("j" * (107 - fixed_size), 0, 0)
    with pytest.raises(ValueError, match="shorter than 108 bytes"):
        make_update_zmq_handle(base, make_update_zmq_id(1, 1))


@pytest.mark.parametrize("sp", [1, 2])
@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("explicit_replica", [-1, 5])
def test_sender_and_receiver_share_base_route(monkeypatch, private_zmq_directory, sp, private, explicit_replica):
    import verl.workers.rollout.base as base
    import verl.workers.rollout.vllm_rollout.vllm_rollout as upstream

    from verl_omni.workers.config import DiffusionRolloutConfig
    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniServerAdapter

    monkeypatch.setattr(base, "omega_conf_to_dataclass", lambda config: config)
    monkeypatch.setattr(upstream.ray, "get_runtime_context", lambda: SimpleNamespace(get_job_id=lambda: "job"))
    monkeypatch.setattr(upstream, "is_support_ipc", lambda: True)
    monkeypatch.setenv("RAY_LOCAL_WORLD_SIZE", "4")
    width = 2 * sp
    monkeypatch.setenv("RANK", str(2 * width - 1))
    if private:
        monkeypatch.setenv("VERL_ZMQ_DIR", str(private_zmq_directory))
    config = DiffusionRolloutConfig(name="vllm_omni", tensor_model_parallel_size=2, ulysses_degree=sp)
    sender = vLLMOmniServerAdapter(config, None, None, replica_rank=explicit_replica)
    replica = 1 if explicit_replica == -1 else explicit_replica
    stage_devices = ",".join(str(device) for device in range(8 + width - 2, 8 + width))
    replica_devices = ",".join(str(device) for device in range(8, 8 + width))
    receiver = _get_handle(
        1,
        {
            "VERL_REPLICA_RANK": str(replica),
            _VISIBLE_DEVICES: stage_devices,
            "VERL_ZMQ_BASE_VISIBLE_DEVICES": replica_devices,
        },
    )
    assert sender.zmq_handle == receiver
    assert sender.rollout_rank == width - 1
    assert sender.replica_rank == replica
    assert sender.node_rank == 0
    assert sender._has_server is False
    expected_dir = private_zmq_directory if private else Path("/tmp")
    assert receiver == f"ipc://{expected_dir}/rl-colocate-zmq-job-replica-{replica}-rank-{width - 1}.sock"
    update_id = make_update_zmq_id(1, 1)
    assert make_update_zmq_handle(sender.zmq_handle, update_id) == make_update_zmq_handle(receiver, update_id)


@pytest.mark.parametrize(
    "private,extras", [(False, " KEEP,OTHER "), (True, ""), (True, " KEEP,OTHER "), (True, "KEEP,VERL_ZMQ_DIR")]
)
def test_private_zmq_directory_updates_vllm_ray_copy_list(monkeypatch, private_zmq_directory, private, extras):
    from verl_omni.workers.rollout.vllm_rollout import vllm_omni_async_server as module

    monkeypatch.setattr(module.vLLMHttpServer, "_post_init", lambda self, devices: None)
    monkeypatch.setenv("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", extras)
    if private:
        monkeypatch.setenv("VERL_ZMQ_DIR", str(private_zmq_directory))
    server = object.__new__(module.vLLMOmniHttpServer)
    server.config = SimpleNamespace(ulysses_degree=1, ring_degree=1, full_determinism=False)
    server._generate_strategy = SimpleNamespace(post_init=lambda devices: None)
    server._post_init("0,1")
    forwarded = os.environ["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"]
    if private:
        assert set(forwarded.split(",")) == {key.strip() for key in extras.split(",") if key.strip()} | {"VERL_ZMQ_DIR"}
        assert forwarded.split(",").count("VERL_ZMQ_DIR") == 1
    else:
        assert forwarded == extras
