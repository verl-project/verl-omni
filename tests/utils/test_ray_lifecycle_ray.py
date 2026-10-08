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
"""Real-Ray cleanup of owned Python shared memory and semaphores; no GPUs."""

import json
import os
import subprocess
import sys
import time
from multiprocessing import shared_memory

import psutil
import pytest
import ray

from verl_omni.utils.ray_lifecycle import actor_process_identity, terminate_actor_and_wait


@ray.remote(num_cpus=1)
class Consumer:
    def start(self, resource_owner):
        self.memory = None
        if resource_owner == "actor":
            self.memory = shared_memory.SharedMemory(create=True, size=1024 * 1024)
        code = """
import json
import multiprocessing as mp
import time
from multiprocessing import shared_memory
memory = shared_memory.SharedMemory(create=True, size=1024 * 1024)
lock = mp.get_context('spawn').Lock()
print(json.dumps({'memory': memory.name, 'semaphore': lock._semlock.name}), flush=True)
time.sleep(3600)
"""
        if resource_owner == "orphan":
            code = (
                "import subprocess, sys, time\n"
                f"child = subprocess.Popen([sys.executable, '-c', {code!r}], "
                "stdout=subprocess.PIPE, text=True, start_new_session=True)\n"
                "print(child.stdout.readline().strip(), flush=True)\n"
                "time.sleep(3600)\n"
            )
        self.child = subprocess.Popen(
            [sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, start_new_session=True
        )
        resources = json.loads(self.child.stdout.readline())
        names = [resources["memory"], "sem." + resources["semaphore"].lstrip("/")]
        if self.memory is not None:
            names.append(self.memory.name)
        children = psutil.Process().children(recursive=True)
        assert any("multiprocessing.resource_tracker" in " ".join(p.cmdline()) for p in children)
        metadata = {
            "identity": actor_process_identity(self),
            "processes": [(p.pid, p.create_time()) for p in children],
            "resources": names,
            "commands": {p.pid: p.cmdline() for p in children},
        }
        if resource_owner == "orphan":
            self.child.terminate()
            self.child.wait(timeout=3)
        return metadata

    def block(self):
        time.sleep(3600)


@pytest.fixture(scope="module")
def cluster(tmp_path_factory):
    ray.init(
        num_cpus=2,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=128 * 1024**2,
        _temp_dir=str(tmp_path_factory.mktemp("rt")),
    )
    yield
    ray.shutdown()


def is_running(pid, created):
    try:
        process = psutil.Process(pid)
        return process.create_time() == created and process.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)
    except psutil.NoSuchProcess:
        return False


@pytest.mark.parametrize("resource_owner", ["actor", "child", "orphan"])
@pytest.mark.parametrize("max_restarts", [0, 1])
def test_forced_consumer_exit_allows_owned_tracker_to_unlink(cluster, resource_owner, max_restarts):
    unrelated = shared_memory.SharedMemory(create=True, size=17)
    unrelated.buf[:] = b"still-owned-here!"
    actor = Consumer.options(max_restarts=max_restarts).remote()
    metadata = ray.get(actor.start.remote(resource_owner), timeout=30)
    print(metadata, flush=True)
    assert all(os.path.exists("/dev/shm/" + name) for name in metadata["resources"])
    actor.block.remote()
    started = time.monotonic()
    try:
        terminate_actor_and_wait(actor, metadata["identity"], timeout=30)
        assert time.monotonic() - started < 35
        assert not is_running(metadata["identity"]["pid"], metadata["identity"]["created"])
        assert all(not is_running(pid, created) for pid, created in metadata["processes"])
        assert not [name for name in metadata["resources"] if os.path.exists("/dev/shm/" + name)]
        assert os.path.exists("/dev/shm/" + unrelated.name)
        assert bytes(unrelated.buf) == b"still-owned-here!"
        deadline = time.monotonic() + 5
        while ray._private.state.actors(actor._actor_id.hex())["State"] != "DEAD":
            assert time.monotonic() < deadline, "Ray actor did not reach terminal state"
            time.sleep(0.01)
        assert ray._private.state.actors(actor._actor_id.hex())["NumRestarts"] == 0
        print(ray._private.state.actors(actor._actor_id.hex()), flush=True)
    finally:
        unrelated.close()
        unrelated.unlink()
        ray.kill(actor, no_restart=True)
