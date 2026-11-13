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

"""CPU lifecycle fixtures only; these are not paired rollout/actor evidence."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

RECIPE = Path(__file__).resolve().parents[2] / "examples/flowgrpo_trainer/sd35/consistency"
spec = importlib.util.spec_from_file_location("sd35_capture_scope", RECIPE / "capture_scope.py")
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
capture_forward = module.capture_forward


class FakeDebugger:
    def __init__(self, *, config_path, events, dump_count=1, stop_error=None):
        self.config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.events = events
        self.dump_count = dump_count
        self.stop_error = stop_error

    def start(self, *, model):
        self.events.append(("start", model))

    def stop(self):
        self.events.append("stop")
        if self.stop_error is not None:
            raise self.stop_error

    def step(self):
        self.events.append("step")
        root = Path(self.config["dump_path"])
        for index in range(self.dump_count):
            proc = root / "step0" / f"rank{index}"
            data = proc / "dump_tensor_data"
            data.mkdir(parents=True)
            (data / "Module.block.output.0.pt").write_bytes(b"fixture tensor artifact")
            (proc / "dump.json").write_text(
                json.dumps(
                    {
                        "task": "tensor",
                        "level": "L0",
                        "dump_data_dir": str(data),
                        "data": {"Module.block.forward.0": {"output": [{"data_name": "Module.block.output.0.pt"}]}},
                    }
                ),
                encoding="utf-8",
            )


class CaptureScopeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.module = torch.nn.Linear(2, 2)
        self.inputs = {
            "hidden_states": torch.tensor([[1.0, 2.0]]),
            "encoder_hidden_states": torch.tensor([[3.0, 4.0]]),
            "pooled_projections": torch.tensor([[5.0, 6.0]]),
            "timestep": torch.tensor([7.0]),
        }
        self.context = {
            "run_id": "run",
            "policy_id": "a" * 64,
            "global_step": "12",
            "sample_id": "sample",
            "trajectory_index": 0,
            "branch": "cond",
            "timestep_unit": "raw_scheduler",
            "timestep": 7.0,
            "noise_level": 0.5,
            "sde_type": "flow",
        }
        self.events = []

    def factory(self, **kwargs):
        return FakeDebugger(events=self.events, **kwargs)

    def call(self, forward, **overrides):
        args = {
            "enabled": True,
            "role": "actor",
            "module": self.module,
            "forward": forward,
            "inputs": self.inputs,
            "context": self.context,
            "output_root": self.root,
            "debugger_factory": self.factory,
        }
        args.update(overrides)
        return capture_forward(**args)

    def test_disabled_invokes_only_forward_once(self):
        value = object()
        calls = []
        with patch.dict("sys.modules", {"msprobe": None}):
            result = capture_forward(
                enabled=False,
                role="invalid",
                module=None,
                forward=lambda: calls.append(1) or value,
                inputs=None,
                context=None,
                output_root=None,
            )
        self.assertIs(result, value)
        self.assertEqual(calls, [1])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_success_records_real_dump_and_exact_result(self):
        value = object()
        calls = []
        result = self.call(lambda: calls.append("forward") or value)
        self.assertIs(result, value)
        self.assertEqual(calls, ["forward"])
        self.assertEqual(
            [event if isinstance(event, str) else event[0] for event in self.events], ["start", "stop", "step"]
        )
        (record_path,) = self.root.rglob("record.json")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        self.assertEqual(record["global_step"], 12)
        self.assertEqual(record["phase"], "actorlog_prob")
        self.assertEqual(record["timestep_input"]["shape"], [1])
        self.assertEqual(record["inputs"]["prompt_embeds"]["shape"], [1, 2])
        self.assertTrue(Path(record["dump_path"]).is_file())

    def test_invalid_context_and_inputs_never_call_forward(self):
        calls = []
        for changes in (
            {"global_step": "01"},
            {"branch": "uncond"},
            {"trajectory_index": True},
            {"timestep": float("nan")},
            {"policy_id": "bad"},
            {"timestep_unit": "msprobe_step"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.call(lambda: calls.append(1), context={**self.context, **changes})
        with self.assertRaises(ValueError):
            self.call(lambda: calls.append(1), role="other")
        with self.assertRaises(ValueError):
            self.call(lambda: calls.append(1), inputs={**self.inputs, "hidden_states": torch.ones(2, 2)})
        self.assertEqual(calls, [])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_missing_or_duplicate_dump_is_incomplete(self):
        for count in (0, 2):
            with self.subTest(count=count):
                subroot = self.root / f"case{count}"
                subroot.mkdir()

                def factory(dump_count=count, **kwargs):
                    return FakeDebugger(events=[], dump_count=dump_count, **kwargs)

                with self.assertRaisesRegex(ValueError, "exactly one"):
                    self.call(lambda: "forward ran", output_root=subroot, debugger_factory=factory)
                self.assertEqual(len(list(subroot.rglob("failure.json"))), 1)
                self.assertEqual(list(subroot.rglob("record.json")), [])

    def test_forward_failure_stops_and_preserves_original_if_stop_fails(self):
        original = RuntimeError("forward failure")

        def failing_forward():
            raise original

        def factory(**kwargs):
            return FakeDebugger(events=self.events, stop_error=RuntimeError("stop failure"), **kwargs)

        with self.assertRaises(RuntimeError) as caught:
            self.call(failing_forward, debugger_factory=factory)
        self.assertIs(caught.exception, original)
        self.assertEqual(self.events[-1], "stop")
        (failure,) = self.root.rglob("failure.json")
        self.assertIn("stop failure", failure.read_text(encoding="utf-8"))
        self.assertEqual(list(self.root.rglob("record.json")), [])

    def test_fresh_path_rejects_second_capture_before_forward(self):
        self.call(lambda: 1)
        calls = []
        with self.assertRaises(FileExistsError):
            self.call(lambda: calls.append(1))
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
