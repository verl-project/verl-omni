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
"""CPU contract checks; real engine admission and GPU updates require the target runtime."""

import copy
import importlib.util
import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("sd35_worker", HERE / "scripts/sd35_microbatch_worker.py")
worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker
SPEC.loader.exec_module(worker)


class WorkerContractTest(unittest.TestCase):
    def test_request_is_fixed_and_identity_bound(self):
        identity = {"data_sha256": "a", "source_sha256": "b", "model_sha256": "c", "precision": "fp32"}
        request = {
            "trial_id": 0,
            "microbatch": 1,
            "accumulation": 2,
            "steps": 2,
            "effective_batch": 2,
            "workload": identity,
            "workload_digest": worker.hashlib.sha256(worker.canonical(identity)).hexdigest(),
        }
        self.assertEqual(worker.validate_request(request, identity), request)
        for change in ({"steps": 3}, {"microbatch": 4}, {"accumulation": 1}, {"workload_digest": "bad"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                worker.validate_request({**request, **change}, identity)
        with self.assertRaises(ValueError):
            worker.validate_request(request, {**identity, "data_sha256": "changed"})

    def test_identity_changes_with_input_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch.pt"
            path.write_bytes(b"one")
            first = worker.file_hash(path)
            path.write_bytes(b"two")
            self.assertNotEqual(first, worker.file_hash(path))
            with self.assertRaises(ValueError):
                worker.read_json(self._json(Path(directory), '{"a":1,"a":2}'))

    @staticmethod
    def _json(directory, body):
        path = directory / "input.json"
        path.write_text(body)
        return path

    def test_real_cpu_tensor_validation(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        data = {
            "all_latents": torch.zeros(2, 3, 16, 32, 32),
            "all_timesteps": torch.zeros(2, 2),
            "prompt_embeds": torch.zeros(2, 1, 4096),
            "pooled_prompt_embeds": torch.zeros(2, 2048),
            "old_log_probs": torch.zeros(2, 2),
            "advantages": torch.ones(2, 2),
        }
        self.assertIs(worker.validate_batch(data, torch), data)
        with self.assertRaises(ValueError):
            worker.validate_batch({**data, "advantages": torch.zeros(2, 2)}, torch)
        with self.assertRaises(ValueError):
            worker.validate_batch({**data, "old_log_probs": torch.full((2, 2), float("nan"))}, torch)
        with self.assertRaises(ValueError):
            worker.validate_batch({**data, "extra": torch.ones(1)}, torch)

    def test_warmup_restore_rewinds_optimizer_scheduler_and_rng(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        import numpy as np

        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        optimizer = torch.optim.AdamW([parameter], lr=0.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        engine = SimpleNamespace(optimizer=optimizer, lr_scheduler=scheduler)
        params = {"lora_weight": parameter}
        initial = {"lora_weight": parameter.detach().clone()}
        initial_optimizer = copy.deepcopy(optimizer.state_dict())
        initial_scheduler = copy.deepcopy(scheduler.state_dict())
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        with (
            patch.object(torch.cuda, "get_rng_state_all", return_value=[torch.tensor([7], dtype=torch.uint8)]),
            patch.object(torch.cuda, "set_rng_state_all") as set_cuda,
        ):
            rng = worker.rng_snapshot(torch, np)
            expected = (random.random(), float(np.random.random()), float(torch.rand(())))
            parameter.grad = torch.tensor([0.5, -0.25])
            optimizer.step()
            scheduler.step()
            random.random()
            np.random.random()
            torch.rand(())
            self.assertFalse(torch.equal(parameter.detach(), initial["lora_weight"]))
            self.assertTrue(optimizer.state)
            worker.restore(engine, params, initial, initial_optimizer, initial_scheduler, rng, torch, np)
            self.assertTrue(torch.equal(parameter.detach(), initial["lora_weight"]))
            self.assertEqual(optimizer.state_dict(), initial_optimizer)
            self.assertEqual(scheduler.state_dict(), initial_scheduler)
            self.assertIsNone(parameter.grad)
            self.assertEqual((random.random(), float(np.random.random()), float(torch.rand(()))), expected)
            set_cuda.assert_called_once()
            self.assertTrue(torch.equal(set_cuda.call_args.args[0][0], rng[1][0]))

    def test_run_error_never_publishes_ok_result(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        workload = {"data_sha256": "a", "source_sha256": "b", "model_sha256": "c", "precision": "fp32"}
        request = {
            "trial_id": 0,
            "microbatch": 1,
            "accumulation": 2,
            "steps": 2,
            "effective_batch": 2,
            "workload": workload,
            "workload_digest": worker.hashlib.sha256(worker.canonical(workload)).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, batch, result = root / "model", root / "batch.pt", root / "result.json"
            model.mkdir()
            batch.write_bytes(b"fixture")
            request_path = self._json(root, json.dumps(request))
            argv = [
                "worker",
                "--mode",
                "run",
                "--model",
                str(model),
                "--batch",
                str(batch),
                "--request",
                str(request_path),
                "--result",
                str(result),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(worker, "imports", return_value=(object(),)),
                patch.object(worker, "identity", return_value=workload),
                patch.object(torch, "load", return_value={}),
                patch.object(worker, "validate_batch", return_value={}),
                patch.object(worker, "admit", return_value=(None, None, None, "config")),
                patch.object(worker, "run", side_effect=RuntimeError("update failed")),
                self.assertRaisesRegex(RuntimeError, "update failed"),
            ):
                worker.main()
            recorded = worker.read_json(result)
            self.assertEqual(recorded["status"], "failure")
            self.assertIsNone(recorded["initial_state_digest"])
            self.assertEqual(recorded["witnesses"], [])
            self.assertFalse(result.with_suffix(".json.diagnostics.json").exists())


class RealCPUAdmissionTest(unittest.TestCase):
    """Opt-in admission against a local SD3.5 cache and captured actor batch."""

    def test_real_scheduler_adapter_and_tensordict_admission(self):
        """Both candidates admit; a changed captured timestep is rejected exactly."""
        model_name = os.environ.get("SD35_MICROBATCH_TEST_MODEL")
        batch_name = os.environ.get("SD35_MICROBATCH_TEST_BATCH")
        if not model_name or not batch_name:
            self.skipTest("set SD35_MICROBATCH_TEST_MODEL and SD35_MICROBATCH_TEST_BATCH for real CPU admission")
        import torch

        self.assertFalse(torch.cuda.is_initialized())
        model_path = Path(model_name).resolve(strict=True)
        batch_path = Path(batch_name).resolve(strict=True)
        payload = worker.validate_batch(torch.load(batch_path, map_location="cpu", weights_only=True), torch)
        classes = worker.imports()
        admitted_two = worker.admit(model_path, payload, 2, classes)
        admitted_one = worker.admit(model_path, payload, 1, classes)
        self.assertEqual(admitted_two[1].ppo_micro_batch_size_per_gpu, 2)
        self.assertEqual(admitted_one[1].ppo_micro_batch_size_per_gpu, 1)
        self.assertEqual(admitted_two[3], admitted_one[3])
        self.assertFalse(torch.cuda.is_initialized())
        changed = dict(payload)
        changed["all_timesteps"] = payload["all_timesteps"].clone()
        changed["all_timesteps"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "configured scheduler first two timesteps"):
            worker.admit(model_path, changed, 1, classes)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()
