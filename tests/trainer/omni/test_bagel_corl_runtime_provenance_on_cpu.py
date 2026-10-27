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

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load():
    spec = importlib.util.spec_from_file_location("manifest", ROOT / "scripts/bagel_corl/write_run_manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_secret_free_complete_source_and_checkpoint_hashes(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    (repo / "main.py").write_text("original")
    subprocess.run(["git", "-C", str(repo), "add", "main.py"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=t@t", "commit", "-m", "fixture"],
        check=True,
        capture_output=True,
    )
    (repo / "main.py").write_text("dirty secret-content")
    (repo / "helper.py").write_text("untracked helper")
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"id": 1}\n')
    checkpoint = tmp_path / "global_step_25"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"checkpoint")
    config = {
        "source_root": str(repo),
        "wandb_api_key": "SUPERSECRET",
        "data": {"train_files": str(dataset)},
        "actor_rollout_ref": {"actor": {"optim": {"lr": 3e-5, "password": "SUPERSECRET"}}, "model": {"lr_gen": 1e-5}},
        "scorer_version": "v1",
    }
    tool = load()
    path = tool.write_run_manifest(tmp_path / "out", config, str(checkpoint))
    first = path.read_text()
    manifest = json.loads(first)
    assert "SUPERSECRET" not in first and "secret-content" not in first
    assert manifest["source"]["dirty_patch_sha256"]
    assert manifest["source"]["untracked_source_sha256"]["helper.py"]
    assert manifest["datasets"]["train"][0]["sha256"]
    assert manifest["checkpoint"]["step"] == 25
    assert manifest["checkpoint"]["files_sha256"]["weights.bin"]
    assert manifest["optimizer_group_lrs"] == {"und": 3e-5, "gen": 1e-5}
    assert tool.write_run_manifest(tmp_path / "out", config, str(checkpoint)).read_text() == first
    (repo / "helper.py").write_text("changed helper")
    changed = json.loads(tool.write_run_manifest(tmp_path / "out", config, str(checkpoint)).read_text())
    assert changed["source"] != manifest["source"]


def test_allowlisted_leaf_cannot_smuggle_secret_subtree(tmp_path):
    with pytest.raises(ValueError, match="scalar"):
        load().write_run_manifest(
            tmp_path,
            {"source_root": str(ROOT), "actor_rollout_ref": {"actor": {"optim": {"lr": {"api_key": "SECRET"}}}}},
            None,
        )


def test_manifest_does_not_hash_its_own_output(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    (repo / "main.py").write_text("source")
    subprocess.run(["git", "-C", str(repo), "add", "main.py"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=t@t", "commit", "-m", "fixture"],
        check=True,
        capture_output=True,
    )
    tool = load()
    path = tool.write_run_manifest(repo / "unignored-output", {"source_root": str(repo)}, None)
    first = path.read_bytes()
    assert tool.write_run_manifest(repo / "unignored-output", {"source_root": str(repo)}, None).read_bytes() == first


@pytest.mark.parametrize(
    "mode,convention",
    [
        ("raw_logprobs", "raw_policy"),
        ("processed_logprobs", "processed_behavior_vs_raw_actor"),
    ],
)
def test_manifest_records_actual_logprob_mode_without_claiming_parity(tmp_path, mode, convention):
    config = {
        "source_root": str(ROOT),
        "actor_rollout_ref": {
            "rollout": {
                "logprobs_mode": mode,
                "temperature": 0.7,
                "repetition_penalty": 1.05,
                "top_k": 50,
                "top_p": 0.9,
            }
        },
    }
    manifest = json.loads(load().write_run_manifest(tmp_path, config, None).read_text())
    assert manifest["resolved_config_allowlist"]["actor_rollout_ref.rollout.logprobs_mode"] == mode
    assert manifest["logprob_convention"] == convention
    assert manifest["logprob_parity_verified"] is False
    assert manifest["behavior_sampling_convention"] == "transformed"
    assert manifest["behavior_importance_correction"] == "unverified"
