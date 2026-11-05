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
"""Subprocess fixture with configurable invalid results and no ML dependency."""

import argparse
import json
import os
import signal
import time
from pathlib import Path

PROCESS_CALLS = 0


def main() -> None:
    """Write a controller-shaped fixture result for one subprocess request."""
    global PROCESS_CALLS
    PROCESS_CALLS += 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    mode = request["workload"].get("mode", "ok").split(",")
    chosen = mode[min(request["trial_id"], len(mode) - 1)]
    if chosen == "exit":
        raise SystemExit(2)
    if chosen == "malformed":
        args.result.write_text("{")
        return
    if chosen == "isolation":
        print(json.dumps({"pid": os.getpid(), "process_calls": PROCESS_CALLS}), flush=True)
        if PROCESS_CALLS != 1:
            chosen = "mismatch"
    if chosen == "ignore_term":

        def mark_shutdown(_signum: int, _frame: object) -> None:
            print("shutdown-observed", flush=True)

        signal.signal(signal.SIGTERM, mark_shutdown)
        print(f"ready {os.getpid()}", flush=True)
        while True:
            time.sleep(0.1)
    if chosen == "missing":
        return
    witnesses = [{"loss": [1.0 + step], "grads": [3.0, 4.0], "params": [5.0, 6.0]} for step in range(request["steps"])]
    if chosen == "signed":
        witnesses = [
            {"loss": [-1.0 - step], "grads": [-3.0, 4.0], "params": [-5.0, 6.0]} for step in range(request["steps"])
        ]
    if chosen == "mismatch":
        witnesses[0]["grads"] = [-3.0, 4.0]
    if chosen == "param_mismatch":
        witnesses[0]["params"] = [-5.0, 6.0]
    if chosen == "shape":
        witnesses[0]["params"] = [5.0]
    if chosen == "grad_shape":
        witnesses[0]["grads"] = [3.0]
    if chosen == "nonfinite":
        witnesses[0]["loss"] = [float("nan")]
    result = {
        key: request[key] for key in ("trial_id", "microbatch", "accumulation", "steps", "workload", "workload_digest")
    }
    result.update(
        status="ok",
        reason="",
        initial_state_digest="same",
        update_seconds=[1.0] * request["steps"],
        witnesses=witnesses,
    )
    if chosen in ("oom", "failure"):
        result.update(status=chosen, reason=chosen, initial_state_digest=None, update_seconds=[], witnesses=[])
    if chosen == "identity":
        result["accumulation"] += 1
    if chosen == "wrong_trial":
        result["trial_id"] += 1
    if chosen == "wrong_workload":
        result["workload"] = {"other": "identity"}
    if chosen == "wrong_steps":
        result["steps"] += 1
    if chosen == "partial":
        del result["witnesses"]
    if chosen == "zero_time":
        result["update_seconds"][0] = 0
    if chosen == "inf_time":
        result["update_seconds"][0] = float("inf")
    if chosen == "overflowing_times":
        result["update_seconds"] = [1e308] * request["steps"]
    if chosen == "huge_time_integer":
        result["update_seconds"][0] = 10**1000
    if chosen == "huge_witness_integer":
        result["witnesses"][0]["grads"][0] = 10**1000
    if chosen == "fast":
        result["update_seconds"] = [0.1] * request["steps"]
    if chosen == "slow":
        result["update_seconds"] = [2.0] * request["steps"]
    if chosen == "state":
        result["initial_state_digest"] = "different"
    args.result.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
