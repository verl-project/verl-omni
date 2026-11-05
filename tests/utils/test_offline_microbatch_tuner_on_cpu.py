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
"""Finite subprocess and schema contract tests for the offline controller."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parents[1] / "scripts" / "tune_microbatch.py"
WORKER = HERE / "fixtures" / "offline_microbatch_worker.py"
spec = importlib.util.spec_from_file_location("offline_microbatch_tuner", SCRIPT)
assert spec is not None and spec.loader is not None
tuner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = tuner
spec.loader.exec_module(tuner)
SearchPlan = tuner.SearchPlan
WorkloadIdentity = tuner.WorkloadIdentity
run = tuner.run


class PlanTests(unittest.TestCase):
    """Validate canonical plan and workload admission."""

    def setUp(self) -> None:
        self.data = dict(
            effective_batch=8, candidates=[1, 2, 4], baseline=2, max_trials=3, steps=3, atol=1e-10, rtol=1e-10
        )

    def test_legal_and_order(self) -> None:
        self.assertEqual(SearchPlan.parse(self.data).ordered(), (2, 1, 4))

    def test_illegal_batches_and_schema(self) -> None:
        for change in (
            {"candidates": [1, 1]},
            {"candidates": [3]},
            {"baseline": 8},
            {"max_trials": 0},
            {"steps": True},
            {"rtol": float("nan")},
            {"extra": 1},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                SearchPlan.parse({**self.data, **change})

    def test_identity_canonical(self) -> None:
        self.assertEqual(
            WorkloadIdentity.parse({"b": "2", "a": "1"}).digest(), WorkloadIdentity.parse({"a": "1", "b": "2"}).digest()
        )
        with self.assertRaises(ValueError):
            WorkloadIdentity.parse({"seed": 1})

    def test_direct_constructor_cannot_bypass_admission(self) -> None:
        valid = SearchPlan.parse(self.data)
        invalid = [
            SearchPlan(8, (1, 3), 1, 2, 2, 0, 0),
            SearchPlan(8, (1, 1), 1, 2, 2, 0, 0),
            SearchPlan(8, (1,), 1, 0, 2, 0, 0),
            SearchPlan(8, (1,), 1, -1, 2, 0, 0),
            SearchPlan(8, (1,), 1, 1, 0, 0, 0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            for index, plan in enumerate(invalid):
                output = Path(directory) / f"bad-{index}"
                with self.subTest(index=index), self.assertRaises(ValueError):
                    run(plan, WorkloadIdentity.parse({"mode": "ok"}), [sys.executable], output)
                self.assertFalse(output.exists())
            output = Path(directory) / "bad-identity"
            with self.assertRaises(ValueError):
                run(valid, WorkloadIdentity((("mode", "ok"), ("mode", "bad"))), [sys.executable], output)
            self.assertFalse(output.exists())


class WorkerTests(unittest.TestCase):
    """Validate fresh workers, numerical alignment and failure exclusion."""

    def execute(
        self, modes: str, *, max_trials: int = 3, candidates: list[int] | None = None, probe_marker: str | None = None
    ) -> dict:
        plan = SearchPlan.parse(
            dict(
                effective_batch=8,
                candidates=candidates or [1, 2, 4],
                baseline=1,
                max_trials=max_trials,
                steps=2,
                atol=1e-10,
                rtol=1e-10,
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            if probe_marker is not None:
                print("\n" + probe_marker, flush=True)
            result = run(plan, WorkloadIdentity.parse({"mode": modes}), [sys.executable, str(WORKER)], output)
            self.assertEqual(json.loads((output / "aggregate.json").read_text()), result)
            self.assertEqual(len(list(output.glob("trial-*/request.json"))), min(max_trials, len(plan.candidates)))
            self.assertEqual(
                json.loads((output / "source.json").read_text())["workload_digest"], result["source"]["workload_digest"]
            )
            return result

    def test_baseline_first_budget_and_tie(self) -> None:
        result = self.execute("ok,ok,ok", max_trials=2)
        self.assertEqual([trial["microbatch"] for trial in result["trials"]], [1, 2])
        self.assertEqual(result["budget_excluded"], [4])
        self.assertEqual(result["recommendation"]["microbatch"], 1)

    def test_faster_valid_candidate(self) -> None:
        result = self.execute("ok,fast,slow")
        self.assertEqual(result["recommendation"]["microbatch"], 2)

    def test_signed_loss_gradient_and_parameter_witnesses_are_valid(self) -> None:
        result = self.execute("signed,signed,signed")
        self.assertEqual([row["status"] for row in result["trials"]], ["valid"] * 3)
        self.assertEqual(result["recommendation"]["microbatch"], 1)

    def test_baseline_failure_and_all_invalid(self) -> None:
        result = self.execute("oom,fast,fast")
        self.assertIsNone(result["recommendation"])
        self.assertTrue(all(item["status"] == "excluded" for item in result["trials"]))

    def test_failure_modes_cannot_win(self) -> None:
        for mode in (
            "oom",
            "failure",
            "malformed",
            "missing",
            "partial",
            "exit",
            "nonfinite",
            "identity",
            "wrong_trial",
            "wrong_workload",
            "wrong_steps",
            "zero_time",
            "inf_time",
            "state",
            "shape",
            "grad_shape",
            "param_mismatch",
        ):
            with self.subTest(mode=mode):
                result = self.execute(f"ok,{mode},ok")
                self.assertEqual(result["trials"][1]["status"], "excluded")
                self.assertEqual(result["recommendation"]["microbatch"], 1)

    def test_element_mismatch_with_same_norm_is_rejected(self) -> None:
        result = self.execute("ok,mismatch")
        self.assertIn("mismatch", result["trials"][1]["reason"])

    def test_overflow_results_do_not_abort(self) -> None:
        result = self.execute("ok,overflowing_times,ok", probe_marker="MICRO_TUNER_OVERFLOW_PROBE_ENTERED")
        self.assertEqual(result["trials"][1]["status"], "excluded")
        self.assertIn("derived", result["trials"][1]["reason"])
        self.assertEqual(result["recommendation"]["microbatch"], 1)
        invalid_baseline = self.execute("overflowing_times,ok,ok")
        self.assertIsNone(invalid_baseline["recommendation"])
        self.assertTrue(all(item["status"] == "excluded" for item in invalid_baseline["trials"]))

    def test_huge_integer_results_do_not_abort(self) -> None:
        for mode in ("huge_time_integer", "huge_witness_integer"):
            with self.subTest(mode=mode):
                result = self.execute(f"ok,{mode},ok", probe_marker="MICRO_TUNER_INTEGER_PROBE_ENTERED")
                self.assertEqual(result["trials"][1]["status"], "excluded")
                self.assertEqual(result["recommendation"]["microbatch"], 1)

    def test_fresh_process_boundary(self) -> None:
        plan = SearchPlan.parse(
            dict(effective_batch=8, candidates=[1, 2, 4], baseline=1, max_trials=3, steps=1, atol=0, rtol=0)
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            result = run(
                plan,
                WorkloadIdentity.parse({"mode": "isolation"}),
                [sys.executable, str(WORKER)],
                output,
            )
            reports = [json.loads((output / f"trial-{index:03d}" / "stdout.txt").read_text()) for index in range(3)]
            self.assertEqual([item["status"] for item in result["trials"]], ["valid"] * 3)
            self.assertEqual([item["process_calls"] for item in reports], [1, 1, 1])
            self.assertEqual(len({item["pid"] for item in reports}), 3)

    def test_launch_failure_is_recorded(self) -> None:
        plan = SearchPlan.parse(
            dict(effective_batch=8, candidates=[1, 2], baseline=1, max_trials=2, steps=1, atol=0, rtol=0)
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            result = run(plan, WorkloadIdentity.parse({"mode": "ok"}), [str(Path(directory) / "absent-worker")], output)
            self.assertIsNone(result["recommendation"])
            self.assertEqual(len(result["trials"]), 2)
            self.assertTrue(all("launch failed" in item["reason"] for item in result["trials"]))
            self.assertTrue((output / "aggregate.json").exists())

    def test_nonmain_thread_rejected_before_output_creation(self) -> None:
        plan = SearchPlan.parse(
            dict(effective_batch=8, candidates=[1], baseline=1, max_trials=1, steps=1, atol=0, rtol=0)
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            failures = []

            def attempt() -> None:
                try:
                    run(plan, WorkloadIdentity.parse({"mode": "ok"}), [sys.executable], output)
                except ValueError as error:
                    failures.append(str(error))

            worker = threading.Thread(target=attempt)
            worker.start()
            worker.join(timeout=5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(len(failures), 1)
            self.assertIn("main thread", failures[0])
            self.assertFalse(output.exists())

    def test_interrupt_settles_sigterm_ignoring_owned_child(self) -> None:
        plan = dict(effective_batch=8, candidates=[1], baseline=1, max_trials=1, steps=1, atol=0, rtol=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plan.json").write_text(json.dumps(plan))
            (root / "workload.json").write_text(json.dumps({"mode": "ignore_term"}))
            output = root / "output"
            command = [
                sys.executable,
                str(SCRIPT),
                "--plan",
                str(root / "plan.json"),
                "--workload",
                str(root / "workload.json"),
                "--output",
                str(output),
                "--",
                sys.executable,
                str(WORKER),
            ]
            controller = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            child_pid = None
            try:
                stream = output / "trial-000" / "stdout.txt"
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if stream.exists() and stream.read_text().startswith("ready "):
                        child_pid = int(stream.read_text().split()[1])
                        break
                    time.sleep(0.02)
                self.assertIsNotNone(child_pid, "fixture never entered its gated blocking phase")
                controller.send_signal(signal.SIGINT)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if "shutdown-observed" in stream.read_text():
                        break
                    self.assertIsNone(controller.poll(), "controller exited before the shutdown phase")
                    time.sleep(0.02)
                self.assertIn("shutdown-observed", stream.read_text())
                self.assertIsNone(controller.poll(), "controller did not remain live for repeated interrupt")
                controller.send_signal(signal.SIGINT)
                controller.communicate(timeout=5)
                self.assertNotEqual(controller.returncode, 0)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                if controller.poll() is None:
                    controller.kill()
                    controller.communicate(timeout=5)
                if child_pid is not None:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_output_not_overwritten(self) -> None:
        plan = SearchPlan.parse(
            dict(effective_batch=8, candidates=[1], baseline=1, max_trials=1, steps=1, atol=0, rtol=0)
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                run(
                    plan,
                    WorkloadIdentity.parse({"mode": "ok"}),
                    [sys.executable, str(WORKER)],
                    Path(directory),
                )


class CLITests(unittest.TestCase):
    """Exercise the public file CLI and persisted artifacts."""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.plan = self.root / "plan.json"
        self.workload = self.root / "workload.json"
        self.output = self.root / "search"
        self.plan.write_text(
            json.dumps(
                {
                    "effective_batch": 8,
                    "candidates": [1, 2, 4],
                    "baseline": 1,
                    "max_trials": 2,
                    "steps": 2,
                    "atol": 1e-10,
                    "rtol": 1e-10,
                }
            )
        )
        self.workload.write_text(json.dumps({"mode": "ok,ok,ok"}))

    def invoke(self, *worker: str) -> subprocess.CompletedProcess[str]:
        """Run the public CLI with a fixture plan in a fresh test directory."""
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--plan",
                str(self.plan),
                "--workload",
                str(self.workload),
                "--output",
                str(self.output),
                "--",
                *worker,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_missing_worker_and_bad_plan_do_not_create_output(self) -> None:
        missing = self.invoke()
        self.assertNotEqual(missing.returncode, 0)
        self.assertFalse(self.output.exists())
        self.plan.write_text(json.dumps({"effective_batch": 8, "candidates": [3]}))
        bad = self.invoke(sys.executable, str(WORKER))
        self.assertNotEqual(bad.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_cli_budget_tie_stdout_and_output_exclusivity(self) -> None:
        self.workload.write_text(json.dumps({"mode": "isolation"}))
        result = self.invoke(sys.executable, str(WORKER))
        self.assertEqual(result.returncode, 0, result.stderr)
        aggregate = json.loads((self.output / "aggregate.json").read_text())
        self.assertEqual([row["microbatch"] for row in aggregate["trials"]], [1, 2])
        self.assertEqual(aggregate["budget_excluded"], [4])
        self.assertEqual(aggregate["recommendation"]["microbatch"], 1)
        self.assertEqual(json.loads(result.stdout), aggregate)
        self.assertTrue((self.output / "trial-000" / "stdout.txt").is_file())
        self.assertEqual(json.loads((self.output / "trial-000" / "stdout.txt").read_text())["process_calls"], 1)
        self.assertTrue((self.output / "trial-000" / "stderr.txt").is_file())
        repeated = self.invoke(sys.executable, str(WORKER))
        self.assertNotEqual(repeated.returncode, 0)
        self.assertIn("FileExistsError", repeated.stderr)

    def test_cli_baseline_failure_cannot_recommend_candidate(self) -> None:
        self.workload.write_text(json.dumps({"mode": "oom,fast,fast"}))
        result = self.invoke(sys.executable, str(WORKER))
        self.assertEqual(result.returncode, 0, result.stderr)
        aggregate = json.loads((self.output / "aggregate.json").read_text())
        self.assertEqual([row["status"] for row in aggregate["trials"]], ["excluded", "excluded"])
        self.assertIsNone(aggregate["recommendation"])


if __name__ == "__main__":
    unittest.main()
