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

"""L1 CPU contract tests for the SD3.5 two-run replay comparator."""

import json
import subprocess
import sys
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "consistency_checking" / "sd3" / "compare_runs.py"


def _save_batch(root: Path, batch: dict[str, torch.Tensor]) -> Path:
    root.mkdir(parents=True)
    path = root / "actor_batch.pt"
    torch.save(batch, path)
    return path


def _run(first: Path, second: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(first.parent), str(second.parent), *extra],
        capture_output=True,
        text=True,
        check=False,
    )


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "all_latents": torch.randn(2, 3, 4),
        "all_timesteps": torch.arange(2 * 3, dtype=torch.float32).reshape(2, 3),
        "old_log_probs": torch.randn(2, 3),
    }


def test_identical_runs_pass(tmp_path: Path) -> None:
    batch = _batch()
    first = _save_batch(tmp_path / "run-a", batch)
    second = _save_batch(tmp_path / "run-b", batch)

    result = _run(first, second)

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["equal"] is True
    assert set(report["tensors"]) == set(batch)


def test_differing_latents_fail(tmp_path: Path) -> None:
    batch = _batch()
    drifted = {key: value.clone() for key, value in batch.items()}
    drifted["all_latents"][0, 0, 0] += 1.0
    first = _save_batch(tmp_path / "run-a", batch)
    second = _save_batch(tmp_path / "run-b", drifted)

    result = _run(first, second)

    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["equal"] is False
    assert report["mismatched"] == ["all_latents"]


def test_declared_tolerance_accepts_float_drift(tmp_path: Path) -> None:
    batch = _batch()
    drifted = {key: value.clone() for key, value in batch.items()}
    drifted["old_log_probs"][0, 0] += 1e-7
    first = _save_batch(tmp_path / "run-a", batch)
    second = _save_batch(tmp_path / "run-b", drifted)

    strict = _run(first, second)
    tolerated = _run(first, second, "--atol", "old_log_probs=1e-6")

    assert strict.returncode == 1
    assert tolerated.returncode == 0, tolerated.stderr
    report = json.loads(tolerated.stdout)
    assert report["tensors"]["old_log_probs"]["atol"] == 1e-6


def test_unknown_tolerance_key_fails(tmp_path: Path) -> None:
    batch = _batch()
    first = _save_batch(tmp_path / "run-a", batch)
    second = _save_batch(tmp_path / "run-b", batch)

    result = _run(first, second, "--atol", "reward=1e-6")

    assert result.returncode == 1
    assert "reward" in result.stderr
