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
"""Actual accelerator fault injection, ordered drain, and repeated residency checks."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ray
import torch
from PIL import Image
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayResourcePool

from tests.reward_loop.validate_native_fsdp import finish_validation
from verl_omni.reward_loop import native_fsdp


class FaultRank(native_fsdp.PickScoreFSDPRank):
    """Test-only faults; the production rank has no fault-injection configuration."""

    def __init__(self, model_path, collective_timeout=30):
        self.fault = json.loads((Path(model_path) / "fault.json").read_text())
        self.fault_rank = int(os.environ["RANK"])
        if self.fault["mode"] == "init" and self.fault_rank == self.fault["rank"]:
            raise RuntimeError("injected rank initialization failure")
        super().__init__(model_path, collective_timeout)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def wake_up(self):
        """Install a hook after the root FSDP all-gather has completed."""
        status = super().wake_up()

        def hook(module, args):
            marker = Path(self.fault["markers"])
            (marker / f"rank{self.fault_rank}").write_text("after root FSDP pre-forward/all-gather")
            if self.fault["mode"] == "hold" or self.fault_rank == self.fault["rank"]:
                if self.fault["mode"] == "forward_error":
                    raise RuntimeError("injected rank forward failure")
                if self.fault["mode"] in {"hold", "kill", "hang", "nccl_timeout"}:
                    deadline = time.monotonic() + 60
                    while not (marker / "release").exists():
                        if time.monotonic() > deadline:
                            raise TimeoutError("test hook was not released")
                        time.sleep(0.01)

        self.model.register_forward_pre_hook(hook)
        return status

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def status(self):
        """Record Ray's actual bundle resource assignment alongside the GPU UUID."""
        # The public runtime API strips placement-group suffixes from resource names.
        resources = ray._private.worker.global_worker.core_worker.resource_ids()
        return {
            **super().status(),
            "assigned_resources": {name: sum(amount for _, amount in values) for name, values in resources.items()},
        }


def wait_markers(directory, timeout=10):
    """Require both ranks to have entered a real sharded forward."""
    deadline = time.monotonic() + timeout
    while not all((directory / f"rank{rank}").exists() for rank in (0, 1)):
        if time.monotonic() > deadline:
            raise TimeoutError("both rank forward markers were not observed")
        time.sleep(0.01)


def case_checkpoint(source, output, mode, rank):
    """Make an immutable per-case checkpoint view without changing source weights."""
    output.mkdir(parents=True)
    shutil.copy2(source / "config.json", output / "config.json")
    if mode == "corrupt":
        (output / "model.safetensors").write_bytes(b"invalid safetensors checkpoint")
    elif mode != "missing":
        (output / "model.safetensors").symlink_to((source / "model.safetensors").resolve())
    markers = output / "markers"
    markers.mkdir()
    (output / "fault.json").write_text(json.dumps({"mode": mode, "rank": rank, "markers": str(markers)}))
    return markers


def main():
    """Run real rank faults on a tiny checkpoint and confirm all owned ranks exit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True, help="tiny checkpoint from --tiny validation")
    parser.add_argument("--processor-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundles", default="0,1", help="rank-ordered parent-pool indices")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    indices = [int(index) for index in args.bundles.split(",")]
    count = torch.cuda.device_count()
    if count <= max(indices):
        raise ValueError("Expose all GPUs addressed by --bundles")
    ray.init(
        num_cpus=12,
        num_gpus=count,
        include_dashboard=False,
        _temp_dir=tempfile.mkdtemp(prefix="nfsdp-f-"),
        object_store_memory=256 * 1024**2,
        runtime_env={
            "env_vars": {
                "OMP_NUM_THREADS": "4",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC": "1000",
            }
        },
    )
    pool = RayResourcePool([count], max_colocate_count=1, name_prefix="fsdp_faults")
    report = {
        "cases": [],
        "bundles": indices,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve().relative_to(Path.cwd()), Path("verl_omni/reward_loop/native_fsdp.py"))
        },
    }
    replica = None
    error = None
    image = Image.new("RGB", (224, 224), color="blue")
    try:
        cases = [
            ("missing", 0),
            ("corrupt", 0),
            ("init", 0),
            ("init", 1),
            ("forward_error", 0),
            ("forward_error", 1),
            ("kill", 0),
            ("kill", 1),
            ("hang", 1),
            ("nccl_timeout", 1),
        ]
        for number, (mode, rank) in enumerate(cases):
            checkpoint = args.output / f"case{number}-{mode}-{rank}"
            markers = case_checkpoint(args.model_path, checkpoint, mode, rank)
            replica = native_fsdp.build_replica(
                pool,
                str(checkpoint),
                args.processor_path,
                timeout=120,
                collective_timeout=5 if mode == "nccl_timeout" else 20,
                bundle_indices=indices,
                _rank_class=FaultRank,
            )
            started = time.monotonic()
            phase = "wake_up"
            try:
                replica.wake_up()
                phase = "infer"
                if mode == "hang":
                    replica.timeout = 5
                elif mode == "nccl_timeout":
                    replica.timeout = 40
                with ThreadPoolExecutor(1) as executor:
                    future = executor.submit(replica.infer, ["a blue image"], [image])
                    if mode in {"kill", "hang", "nccl_timeout"}:
                        wait_markers(markers)
                        assert not future.done()
                        if mode == "kill":
                            ray.kill(replica.worker_group.workers[rank], no_restart=True)
                    future.result(timeout=40)
            except Exception as exc:
                elapsed = time.monotonic() - started
                assert replica._closed and replica._termination_error is None
                assert elapsed < 130
                if mode in {"forward_error", "kill", "hang", "nccl_timeout"}:
                    assert phase == "infer"
                    assert all((markers / f"rank{index}").exists() for index in (0, 1))
                if mode == "hang":
                    assert isinstance(exc.__cause__, ray.exceptions.GetTimeoutError)
                elif mode == "nccl_timeout":
                    assert isinstance(exc.__cause__, ray.exceptions.RayActorError | ray.exceptions.RayTaskError)
                for actor in replica.worker_group.workers:
                    try:
                        ray.get(actor.status.remote(), timeout=5)
                    except ray.exceptions.RayActorError:
                        continue
                    raise AssertionError("rank survived a group failure") from exc
                report["cases"].append(
                    {
                        "mode": mode,
                        "rank": rank,
                        "error": type(exc).__name__,
                        "phase": phase,
                        "cause_type": type(exc.__cause__).__name__,
                        "seconds": elapsed,
                        "forward_markers": [p.name for p in markers.glob("rank*")],
                        "all_ranks_dead": True,
                    }
                )
            else:
                raise AssertionError(f"fault {mode} returned success")
            replica.close()
            replica = None

        # Twenty real load/score/unload cycles must reach a stable memory plateau.
        healthy_checkpoint = args.output / "healthy"
        case_checkpoint(args.model_path, healthy_checkpoint, "none", 0)
        replica = native_fsdp.build_replica(
            pool, str(healthy_checkpoint), args.processor_path, bundle_indices=indices, _rank_class=FaultRank
        )
        asleep = []
        previous = None
        for _ in range(20):
            status = replica.wake_up()
            if previous is None:
                report["placement_shards"] = status
                assert [row["rank"] for row in status] == [0, 1]
                assert len({row["gpu_uuid"] for row in status}) == 2
                for row, index in zip(status, indices, strict=True):
                    assert any(
                        key.startswith(f"GPU_group_{index}_") and value == 1
                        for key, value in row["assigned_resources"].items()
                    )
            result = replica.infer(["a blue image"], [image])
            if previous is not None:
                for key in previous:
                    torch.testing.assert_close(result[key], previous[key], rtol=0, atol=0)
            previous = result
            status = replica.sleep()
            assert all(row["local_numel"] == 0 for row in status)
            asleep.append([row["allocated_bytes"] for row in status])
        for rank in (0, 1):
            values = [row[rank] for row in asleep[2:]]
            assert max(values) - min(values) < 2 * 1024**2
        report["residency_cycles"] = asleep
        replica.close()
        replica = None

        for method in ("sleep", "close"):
            checkpoint = args.output / f"drain-{method}"
            markers = case_checkpoint(args.model_path, checkpoint, "hold", 0)
            replica = native_fsdp.build_replica(
                pool, str(checkpoint), args.processor_path, bundle_indices=indices, _rank_class=FaultRank
            )
            replica.wake_up()
            lock, attempted = threading.Lock(), threading.Event()

            class ObservedLock:
                def __init__(self, lock, attempted):
                    self.lock, self.attempted = lock, attempted

                def __enter__(self):
                    if self.lock.locked():
                        self.attempted.set()
                    self.lock.acquire()

                def __exit__(self, *args):
                    self.lock.release()

            replica._lock = ObservedLock(lock, attempted)
            with ThreadPoolExecutor(2) as executor:
                forward = executor.submit(replica.infer, ["a blue image"], [image])
                wait_markers(markers)
                lifecycle = executor.submit(getattr(replica, method))
                assert attempted.wait(5) and not lifecycle.done()
                (markers / "release").touch()
                assert forward.result(15)["image_embeddings"].shape[0] == 1
                lifecycle.result(15)
            replica.close()
            replica = None
            report[f"concurrent_{method}_drained"] = True
        report["passed"] = True
    except BaseException as exc:
        error = exc
    finally:
        finish_validation(replica, pool, args.output, report, error)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
