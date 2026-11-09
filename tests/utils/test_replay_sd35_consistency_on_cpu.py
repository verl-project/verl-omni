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

"""CPU contract tests for the existing SD3.5 producer replay bridge."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "replay_sd35_consistency.py"
SHA = "a" * 64
INPUTS = {
    name: {"sha256": SHA, "dtype": "torch.float32", "shape": [1]}
    for name in ("hidden_states", "prompt_embeds", "pooled_projections")
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def fixture(tmp_path, nested=True):
    root = tmp_path / "capture"
    root.mkdir()
    policy = "b" * 64
    records = {}
    for role in ("rollout", "actor"):
        folder = f"{role}/captures/{role}/capture-id"
        dump = f"{folder}/dump/step0/rank0/dump.json"
        put_json(root / dump, {"data": {"Module.layer": {"value": 1}}})
        row = {
            "run_id": root.name,
            "policy_id": policy,
            "global_step": 1,
            "sample_id": "sd35-prompt-0",
            "trajectory_index": 0,
            "branch": "cond",
            "timestep_unit": "raw_scheduler",
            "timestep": 812.5,
            "noise_level": 0.7,
            "sde_type": "sde",
            "role": role,
            "phase": "rolloutgenerate" if role == "rollout" else "actorlog_prob",
            "timestep_input": {"sha256": SHA, "dtype": "torch.float32", "shape": [1]},
            "dump_path": str(root / dump),
        }
        row.update({"inputs": INPUTS} if nested else INPUTS)
        # The producer's raw capture always has nested input metadata.
        raw = {**row, "inputs": INPUTS}
        put_json(root / folder / "record.json", raw)
        records[role] = row
    batch = {"old_log_probs": torch.ones(2, 2, dtype=torch.float32)}
    tensor_path = root / "rollout/actor_batch.pt"
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(batch, tensor_path)
    comparison_path = root / "actor/comparison.pt"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"actor_log_probs": batch["old_log_probs"].clone(), "rollout_log_probs": batch["old_log_probs"].clone()},
        comparison_path,
    )
    common = {
        "shared_run_root": str(root),
        "rollout_root": str(root / "rollout"),
        "identity": {"run": root.name},
        "provenance": {
            "source_head": "c" * 40,
            "source_manifest_sha256": "d" * 64,
            "probe_sha256": "e" * 64,
            "transformer_weight_sha256": policy,
            "transformer_weight_hash_verified": True,
        },
    }
    mapping = [
        {
            "request_id": f"{root.name}:sd35-prompt-{i}",
            "sample_id": f"sd35-prompt-{i}",
            "prompt": prompt,
            "seed": 1234 + i,
            "api_output_index": i,
        }
        for i, prompt in enumerate(("A red cube on a white table.", "A blue sphere on a white table."))
    ]
    put_json(
        root / "rollout/manifest.json",
        {
            **common,
            "role": "rollout",
            "real_prompt": True,
            "request_map": mapping,
            "actor_batch_sha256": digest(tensor_path.read_bytes()),
            "capture_record": records["rollout"],
        },
    )
    put_json(
        root / "actor/comparison.json",
        {
            **common,
            "role": "actor",
            "rollout_batch_sha256": digest(tensor_path.read_bytes()),
            "comparison_sha256": digest(comparison_path.read_bytes()),
            "allclose_atol": 1e-5,
            "allclose_rtol": 1e-3,
            "capture_record": records["actor"],
        },
    )
    return root


def pin(root):
    names = ["rollout/manifest.json", "rollout/actor_batch.pt", "actor/comparison.json", "actor/comparison.pt"]
    names += [f"{role}/captures/{role}/capture-id/record.json" for role in ("rollout", "actor")]
    names += [f"{role}/captures/{role}/capture-id/dump/step0/rank0/dump.json" for role in ("rollout", "actor")]
    data = "".join(f"{digest((root / name).read_bytes())}  {name}\n" for name in names).encode()
    (root.parent / "producer-pins.sha256").write_bytes(data)
    return digest(data)


def run(root, sha, output):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(root),
            "--pins",
            str(root.parent / "producer-pins.sha256"),
            "--pins-sha256",
            sha,
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize("nested", [True, False])
def test_cli_replays_captured_first_pair(tmp_path, nested):
    root = fixture(tmp_path, nested)
    original = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    sha = pin(root)
    result = run(root, sha, tmp_path / "report")
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "report/report.json").read_text())
    assert report["status"] == "admitted" and report["row_keys"] == [["sd35-prompt-0", 0, "cond"]]
    assert all((root / path).read_bytes() == data for path, data in original.items())


def test_cli_replays_unchanged_bytes_after_relocation(tmp_path):
    root = fixture(tmp_path)
    sha = pin(root)
    moved = tmp_path / "new-location"
    shutil.copytree(root, moved)
    result = run(moved, sha, tmp_path / "report")
    assert result.returncode == 0, result.stderr
    assert all(
        (moved / path.relative_to(root)).read_bytes() == path.read_bytes() for path in root.rglob("*") if path.is_file()
    )


@pytest.mark.parametrize(
    "change",
    [
        "corrupt",
        "sample",
        "policy",
        "ordering",
        "seed",
        "prompt",
        "source",
        "provenance",
        "tolerance",
        "pin_digest",
        "existing",
    ],
)
def test_cli_rejects_bad_source_or_destination(tmp_path, change):
    root = fixture(tmp_path)
    sha = pin(root)
    output = tmp_path / "report"
    if change == "corrupt":
        (root / "actor/comparison.pt").write_bytes(b"corrupt")
    elif change in ("sample", "policy"):
        path = root / "actor/captures/actor/capture-id/record.json"
        row = json.loads(path.read_text())
        row["sample_id" if change == "sample" else "policy_id"] = "wrong"
        put_json(path, row)
        sha = pin(root)
    elif change in ("ordering", "seed", "prompt", "source", "provenance", "tolerance"):
        path = root / "rollout/manifest.json"
        if change in ("source", "provenance", "tolerance"):
            path = root / "actor/comparison.json"
        row = json.loads(path.read_text())
        if change == "ordering":
            row["request_map"].reverse()
        elif change == "seed":
            row["request_map"][0]["seed"] = 999
        elif change == "prompt":
            row["request_map"][0]["prompt"] = "different"
        elif change == "source":
            row["provenance"]["source_manifest_sha256"] = "bad"
        elif change == "provenance":
            del row["provenance"]["probe_sha256"]
        else:
            row["allclose_atol"] = 100
        put_json(path, row)
        if change in ("source", "provenance"):
            rollout_path = root / "rollout/manifest.json"
            rollout_row = json.loads(rollout_path.read_text())
            rollout_row["provenance"] = row["provenance"]
            put_json(rollout_path, rollout_row)
        sha = pin(root)
    elif change == "pin_digest":
        sha = "0" * 64
    else:
        output.mkdir()
    result = run(root, sha, output)
    assert result.returncode != 0
    assert not (output / "report.json").exists()


def test_cli_reports_numerical_mismatch(tmp_path):
    root = fixture(tmp_path)
    path = root / "actor/comparison.pt"
    tensors = torch.load(path, map_location="cpu", weights_only=True)
    tensors["actor_log_probs"][0, 0] += 0.1
    torch.save(tensors, path)
    metadata_path = root / "actor/comparison.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["comparison_sha256"] = digest(path.read_bytes())
    put_json(metadata_path, metadata)
    sha = pin(root)
    output = tmp_path / "report"
    result = run(root, sha, output)
    assert result.returncode != 0
    assert json.loads((output / "report.json").read_text())["status"] == "numerical_mismatch"
    assert json.loads((output / "audit.json").read_text())["checker_report"]["status"] == "numerical_mismatch"


def test_help_has_no_capture_or_model_default(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], text=True, capture_output=True)
    assert result.returncode == 0
    assert "--root" in result.stdout and "--pins-sha256" in result.stdout
    assert "/share/" not in result.stdout
