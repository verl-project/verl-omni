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

"""CPU contract tests for saved diffusion trainer capture replay."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "replay_diffusion_trainer_capture.py"
BATCH = "a" * 32
SOURCE = "b" * 64


def save_event(root, name, kind, payload, **record_overrides):
    folder = root / name
    folder.mkdir()
    path = folder / "payload.pt"
    torch.save(payload, path)
    record = {
        "version": 1,
        "kind": kind,
        "run_id": "run-one",
        "batch_id": BATCH,
        "global_step": 2,
        "rank": 0,
        "payload_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "capture_source_sha256": SOURCE,
    }
    if kind == "engine_forward":
        record["sigma_timestep_contract"] = True
    record.update(record_overrides)
    (folder / "record.json").write_text(json.dumps(record) + "\n")
    return folder


def driver(n=3, t=4):
    base = torch.ones((n, t), dtype=torch.float32)
    steps = torch.arange(t, dtype=torch.float32).expand(n, t).clone() * 100
    return {
        "actor_log_probs": base.clone(),
        "rollout_log_probs": base.clone(),
        "timesteps": steps,
        "uids": ["duplicate"] * n,
    }


def engine():
    return {
        "actor_log_probs": torch.ones((2, 1), dtype=torch.float32),
        "rollout_log_probs": torch.ones((2, 1), dtype=torch.float32),
        "timesteps": torch.tensor([[400.0], [600.0]]),
        "sigma": torch.tensor([[0.4], [0.6]]),
        "sigma_next": torch.tensor([[0.3], [0.5]]),
        "next_timesteps": torch.tensor([[300.0], [500.0]]),
        "num_train_timesteps": 1000,
        "sigma_indices": torch.tensor([4, 6], dtype=torch.long),
    }


def run(root, output, *args):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), "--output", str(output), *args],
        text=True,
        capture_output=True,
    )
    report = json.loads(output.read_text()) if output.is_file() else None
    return result, report


def test_full_driver_and_repeated_engine_probes(tmp_path):
    root = tmp_path / "capture"
    root.mkdir()
    save_event(root, "driver", "driver_pair", driver(5, 7))
    save_event(root, "engine-one", "engine_forward", engine())
    save_event(root, "engine-two", "engine_forward", engine())
    result, report = run(root, tmp_path / "report.json")
    assert result.returncode == 0, result.stderr
    assert report["status"] == "admitted"
    assert report["driver_pair_count"] == 1 and report["engine_forward_count"] == 2
    assert report["sigma_contract_probe_count"] == 2 and report["sigma_diagnosis_available"] is True
    assert next(row for row in report["findings"] if row["kind"] == "driver_pair")["shape"] == [5, 7]
    assert all(row["comparison"]["allclose"] for row in report["findings"])


def test_offcell_driver_failure_covers_entire_matrix(tmp_path):
    root = tmp_path / "capture"
    root.mkdir()
    payload = driver(4, 5)
    payload["actor_log_probs"][3, 4] += 0.1
    save_event(root, "driver", "driver_pair", payload)
    save_event(root, "engine", "engine_forward", engine())
    result, report = run(root, tmp_path / "report.json")
    assert result.returncode == 3, result.stderr
    assert report["status"] == "numerical_mismatch"
    finding = report["findings"][0]
    assert finding["comparison"]["max_abs_logprob_difference"] == pytest.approx(0.1)
    assert finding["per_timestep"][-1]["count"] == 4
    assert finding["per_timestep"][-1]["rollout_corr/logprob_abs_diff_max"] == pytest.approx(0.1)


@pytest.mark.parametrize("field", ["sigma", "sigma_next"])
def test_consumed_sigma_fault_is_logic_mismatch(tmp_path, field):
    root = tmp_path / "capture"
    root.mkdir()
    save_event(root, "driver", "driver_pair", driver())
    payload = engine()
    payload[field][1, 0] += 0.02
    payload["actor_log_probs"][0, 0] += 0.1
    save_event(root, "engine", "engine_forward", payload)
    result, report = run(root, tmp_path / "report.json")
    assert result.returncode == 2, result.stderr
    assert report["status"] == "logic_mismatch"
    fault = report["findings"][1]["sigma_faults"][0]
    assert fault["sample_local_row"] == 1 and fault["field"] == field
    assert fault["raw_timestep"] == (600 if field == "sigma" else 500)
    assert fault["expected"] == pytest.approx(0.6 if field == "sigma" else 0.5)
    assert fault["actual"] != pytest.approx(fault["expected"])


def test_restored_sigma_and_numeric_only(tmp_path):
    root = tmp_path / "capture"
    root.mkdir()
    save_event(root, "driver", "driver_pair", driver())
    payload = engine()
    payload["actor_log_probs"][0, 0] += 0.1
    save_event(root, "engine", "engine_forward", payload)
    result, report = run(root, tmp_path / "numeric.json")
    assert result.returncode == 3, result.stderr
    assert report["status"] == "numerical_mismatch"
    assert report["sigma_contract_probe_count"] == 1 and report["sigma_diagnosis_available"] is True
    assert report["findings"][1]["sigma_faults"] == []
    folder = root / "engine"
    for path in folder.iterdir():
        path.unlink()
    folder.rmdir()
    save_event(root, "engine", "engine_forward", engine())
    result, report = run(root, tmp_path / "restored.json")
    assert result.returncode == 0, result.stderr
    assert report["status"] == "admitted"


def test_uncontracted_probe_does_not_claim_sigma_logic(tmp_path):
    root = tmp_path / "capture"
    root.mkdir()
    save_event(root, "driver", "driver_pair", driver())
    payload = engine()
    payload["actor_log_probs"][1, 0] += 0.1
    for field in ("sigma", "sigma_next", "next_timesteps", "num_train_timesteps", "sigma_indices"):
        del payload[field]
    save_event(root, "engine", "engine_forward", payload, sigma_timestep_contract=False)
    result, report = run(root, tmp_path / "report.json")
    assert result.returncode == 3, result.stderr
    assert report["status"] == "numerical_mismatch"
    assert report["sigma_contract_probe_count"] == 0 and report["sigma_diagnosis_available"] is False
    assert report["findings"][1]["sigma_faults"] == []


def test_driver_only_capture_is_rejected(tmp_path):
    root = tmp_path / "capture"
    root.mkdir()
    save_event(root, "driver", "driver_pair", driver())
    result, report = run(root, tmp_path / "report.json")
    assert result.returncode != 0 and "at least one engine_forward is required" in result.stderr
    assert report is None


def test_duplicate_driver_batch_is_rejected(tmp_path):
    root = tmp_path / "capture"
    root.mkdir()
    save_event(root, "driver-one", "driver_pair", driver())
    save_event(root, "driver-two", "driver_pair", driver())
    result, report = run(root, tmp_path / "report.json")
    assert result.returncode != 0 and "duplicate driver batch" in result.stderr
    assert report is None


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("missing", "partial event"),
        ("hash", "payload SHA256 mismatch"),
        ("orphan", "orphan engine event"),
        ("source", "capture source mismatch"),
        ("nonfinite", "finite CPU float32"),
        ("shape", "finite CPU float32"),
        ("symlink", "root has invalid events"),
        ("path", "root must be unsymlinked"),
    ],
)
def test_invalid_capture_fails_without_report(tmp_path, damage, message):
    root = tmp_path / "capture"
    root.mkdir()
    payload = driver()
    if damage == "nonfinite":
        payload["actor_log_probs"][0, 0] = float("nan")
    if damage == "shape":
        payload["timesteps"] = payload["timesteps"][:, :1]
    folder = save_event(root, "driver", "driver_pair", payload)
    if damage == "missing":
        (folder / "record.json").unlink()
    elif damage == "hash":
        (folder / "payload.pt").write_bytes(b"changed")
    elif damage in ("orphan", "source"):
        overrides = {"batch_id": "c" * 32} if damage == "orphan" else {"capture_source_sha256": "d" * 64}
        save_event(root, "engine", "engine_forward", engine(), **overrides)
    elif damage == "symlink":
        (root / "linked").symlink_to(folder, target_is_directory=True)
    elif damage == "path":
        linked_root = tmp_path / "linked-root"
        linked_root.symlink_to(root, target_is_directory=True)
        root = linked_root
    result, report = run(root, tmp_path / "report.json")
    assert result.returncode != 0 and message in result.stderr
    assert report is None


def test_output_exclusive_and_no_cuda_initialization(tmp_path):
    root = tmp_path / "capture"
    root.mkdir()
    save_event(root, "driver", "driver_pair", driver())
    save_event(root, "engine", "engine_forward", engine())
    output = tmp_path / "report.json"
    result, _ = run(root, output)
    assert result.returncode == 0, result.stderr
    result, _ = run(root, output)
    assert result.returncode != 0 and "output must be fresh" in result.stderr
    result, _ = run(root, root / "inside.json")
    assert result.returncode != 0 and "output must be outside root" in result.stderr
    assert not torch.cuda.is_initialized()
