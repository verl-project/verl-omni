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
"""Inspect real trainer cleanup before shutdown after success or missing reward weights."""

import argparse
import json
import os
import runpy
import sys
import time
from pathlib import Path

import ray

from tests.reward_loop.validate_native_fsdp import owned_actor_states


def main():
    """Require the expected fit outcome, checkpoint and dead owned actors."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-success", action="store_true")
    args = parser.parse_args()
    report = {"passed": False, "expected_success": args.expect_success, "executable": sys.executable}
    failure = None
    try:
        os.environ["HYDRA_FULL_ERROR"] = "1"
        sys.argv = ["verl_omni.trainer.main_diffusion_v1", *json.loads(args.overrides.read_text())]
        try:
            runpy.run_module("verl_omni.trainer.main_diffusion_v1", run_name="__main__")
        except Exception as exc:
            if args.expect_success:
                raise
            message = str(exc)
            assert "FSDP wake_up failed" in message and "fit" in message
            assert "pytorch_model.bin" in message or "model.safetensors" in message
            report["expected_error"] = {"type": type(exc).__name__, "message": message}
        else:
            assert args.expect_success, "missing reward weights returned training success"
        assert ray.is_initialized(), "cleanup must be inspected before driver/cluster shutdown"
        job_id = ray.get_runtime_context().get_job_id().hex()
        expected_counts = {
            "PickScoreFSDPRank": 2,
            "OmniFSDPRewardLoopWorker": 1,
            "TransferQueueController": 1,
            "SimpleStorageUnit": 1,
        }
        actor_ids = [
            actor_id
            for actor_id, row in ray._private.state.actors().items()
            if row["JobID"] == job_id and row["ActorClassName"] in expected_counts
        ]
        deadline = time.monotonic() + 15
        while True:
            rows = owned_actor_states(actor_ids, expected_counts)
            if all(row["State"] == "DEAD" for row in rows):
                break
            if time.monotonic() > deadline:
                raise AssertionError(f"owned reward or queue actors survived cleanup: {rows}")
            time.sleep(0.1)
        report["actors_before_ray_shutdown"] = rows
        latest = int((args.checkpoint_root / "latest_checkpointed_iteration.txt").read_text())
        assert latest == args.expected_step
        assert not (args.checkpoint_root / f"global_step_{latest + 1}").exists()
        report.update(passed=True, latest_checkpoint=latest)
    except BaseException as exc:
        failure = exc
        report["verification_error"] = str(exc)
    finally:
        ray.shutdown()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if failure is not None:
        raise failure
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
