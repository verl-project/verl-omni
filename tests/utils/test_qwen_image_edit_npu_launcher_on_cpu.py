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
"""Exercise the NPU wrapper's checkpoint ownership and launch boundaries on CPU."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_LAUNCHER = Path(__file__).resolve().parents[2] / "tests/special_e2e/run_flowgrpo_qwen_image_edit_npu.sh"


def _checkpoint(directory, *, marker=False):
    """Create a caller-owned checkpoint, optionally marked as a builder fixture."""
    directory.mkdir(parents=True)
    (directory / "model_index.json").write_text('{"_class_name": "QwenImageEditPlusPipeline"}')
    (directory / "weights.bin").write_bytes(b"user checkpoint weights")
    if marker:
        (directory / "tiny_checkpoint_metadata.json").write_text(
            json.dumps({"format_version": 1, "chat_template": "previous tiny template"})
        )
    return directory


def _files(path):
    """Snapshot file contents so launcher tests can detect unintended writes."""
    if path.is_file():
        return {"": path.read_bytes()}
    return {str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()}


@pytest.fixture
def edit_launcher(tmp_path):
    """Isolate the real wrapper with recording builder and trainer subprocesses."""
    script_dir = tmp_path / "fixture repo" / "tests" / "special_e2e"
    script_dir.mkdir(parents=True)
    launcher = script_dir / _LAUNCHER.name
    shutil.copyfile(_LAUNCHER, launcher)
    (script_dir / "lib_npu_e2e.sh").write_text('NUM_NPUS=${NUM_NPUS:-2}\nNPU_RUNTIME_ARGS=("trainer.device=npu")\n')
    (script_dir / "build_qwen_image_edit_plus_tiny_random.py").write_text(
        """import json, os, sys
from pathlib import Path
output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
with Path(os.environ["LAUNCH_EVENTS"]).open("a") as stream:
    stream.write(json.dumps({"kind": "builder", "model_path": str(output)}) + "\\n")
output.mkdir(parents=True, exist_ok=True)
(output / "weights.bin").write_bytes(b"tiny random weights")
(output / "model_index.json").write_text("{}")
(output / "tiny_checkpoint_metadata.json").write_text(
    json.dumps({"format_version": 1, "chat_template": "stub tiny template"}))
raise SystemExit(int(os.environ.get("STUB_BUILDER_EXIT", "0")))
"""
    )
    (script_dir / "run_flowgrpo_qwen_image_edit.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail
python3 - "$@" <<'PY'
import json, os, sys
from pathlib import Path
with Path(os.environ["LAUNCH_EVENTS"]).open("a") as stream:
    stream.write(json.dumps({
        "kind": "trainer", "model_path": os.environ["MODEL_PATH"], "argv": sys.argv[1:]
    }) + "\\n")
raise SystemExit(int(os.environ.get("STUB_TRAINER_EXIT", "0")))
PY
"""
    )
    home = tmp_path / "home"
    home.mkdir()
    events = tmp_path / "events.jsonl"

    def run(overrides=None, *args):
        """Return the wrapper's raw exit and ordered builder/trainer events."""
        env = dict(os.environ)
        env.pop("MODEL_PATH", None)
        env.update(
            HOME=str(home),
            DATA_DIR=str(tmp_path / "data"),
            LAUNCH_EVENTS=str(events),
            PATH=str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", ""),
            STUB_BUILDER_EXIT="0",
            STUB_TRAINER_EXIT="0",
        )
        env.update(overrides or {})
        result = subprocess.run(["bash", str(launcher), *args], env=env, capture_output=True, text=True, timeout=20)
        recorded = [json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
        return result, recorded

    return SimpleNamespace(
        run=run,
        script_dir=script_dir,
        home=home,
        default=home / ".cache/verl-omni-npu-smoke/qwen-image-edit-plus",
    )


@pytest.mark.parametrize("symlink", [False, True])
def test_explicit_checkpoint_is_reused_without_a_tiny_marker(edit_launcher, tmp_path, symlink):
    """Reuse caller checkpoints, including symlinks, without rebuilding their files."""
    target = _checkpoint(tmp_path / "user checkpoint")
    model = target
    if symlink:
        model = tmp_path / "checkpoint link"
        model.symlink_to(target, target_is_directory=True)
    before = _files(target)

    result, events = edit_launcher.run({"MODEL_PATH": str(model)}, "trainer.experiment_name=caller-choice")

    assert result.returncode == 0, result.stderr
    assert _files(target) == before
    assert [event["kind"] for event in events] == ["trainer"]
    assert events[0]["model_path"] == str(model)
    assert events[0]["argv"][-1] == "trainer.experiment_name=caller-choice"
    assert "trainer.device=npu" in events[0]["argv"]


@pytest.mark.parametrize("kind", ["missing", "file", "missing-manifest"])
def test_invalid_explicit_checkpoint_fails_without_writing(edit_launcher, tmp_path, kind):
    """Reject invalid explicit paths before either builder or trainer can write."""
    model = tmp_path / "explicit checkpoint"
    if kind == "file":
        model.write_bytes(b"user data")
    elif kind == "missing-manifest":
        model.mkdir()
        (model / "weights.bin").write_bytes(b"user checkpoint weights")
    existed = model.exists()
    before = _files(model)

    result, events = edit_launcher.run({"MODEL_PATH": str(model)})

    assert result.returncode == 2
    assert "MODEL_PATH" in result.stderr
    assert events == []
    assert model.exists() == existed
    assert _files(model) == before


@pytest.mark.parametrize("empty_override", [False, True])
def test_only_the_default_fixture_is_built(edit_launcher, empty_override):
    """Build the managed default only when no nonempty checkpoint override is set."""
    result, events = edit_launcher.run({"MODEL_PATH": ""} if empty_override else {})

    assert result.returncode == 0, result.stderr
    assert [event["kind"] for event in events] == ["builder", "trainer"]
    assert all(event["model_path"] == str(edit_launcher.default) for event in events)
    assert (edit_launcher.default / "weights.bin").read_bytes() == b"tiny random weights"


def test_explicit_default_path_is_still_reused(edit_launcher):
    """Treat an explicit default-directory override as caller owned."""
    _checkpoint(edit_launcher.default)
    before = _files(edit_launcher.default)

    result, events = edit_launcher.run({"MODEL_PATH": str(edit_launcher.default)})

    assert result.returncode == 0, result.stderr
    assert [event["kind"] for event in events] == ["trainer"]
    assert _files(edit_launcher.default) == before


@pytest.mark.parametrize("populated", [False, True])
def test_default_empty_or_builder_owned_directory_is_allowed(edit_launcher, populated):
    """Permit rebuilding an empty cache or a fixture with the expected ownership marker."""
    if populated:
        _checkpoint(edit_launcher.default, marker=True)
    else:
        edit_launcher.default.mkdir(parents=True)

    result, events = edit_launcher.run()

    assert result.returncode == 0, result.stderr
    assert [event["kind"] for event in events] == ["builder", "trainer"]


@pytest.mark.parametrize("metadata", [None, "invalid-json", '{"format_version": 2}', "[]"])
def test_unowned_default_directory_is_not_modified(edit_launcher, metadata):
    """Reject populated caches whose ownership metadata is absent or invalid."""
    _checkpoint(edit_launcher.default)
    if metadata is not None:
        (edit_launcher.default / "tiny_checkpoint_metadata.json").write_text(metadata)
    before = _files(edit_launcher.default)

    result, events = edit_launcher.run()

    assert result.returncode != 0
    assert "unowned path" in result.stderr
    assert events == []
    assert _files(edit_launcher.default) == before


def test_default_leaf_symlink_is_not_modified(edit_launcher, tmp_path):
    """Do not rebuild the managed cache through a symlink to someone else's model."""
    target = _checkpoint(tmp_path / "user model", marker=True)
    edit_launcher.default.parent.mkdir(parents=True)
    edit_launcher.default.symlink_to(target, target_is_directory=True)
    before = _files(target)

    result, events = edit_launcher.run()

    assert result.returncode != 0
    assert edit_launcher.default.is_symlink()
    assert "unowned path" in result.stderr
    assert events == []
    assert _files(target) == before


def test_default_file_is_not_modified(edit_launcher):
    """Leave non-directory data at the default cache path untouched."""
    edit_launcher.default.parent.mkdir(parents=True)
    edit_launcher.default.write_bytes(b"user data")

    result, events = edit_launcher.run()

    assert result.returncode != 0
    assert "unowned path" in result.stderr
    assert events == []
    assert edit_launcher.default.read_bytes() == b"user data"


def test_default_cache_parent_may_be_a_mount_symlink(edit_launcher, tmp_path):
    """Allow a mounted cache parent while preserving ownership of the leaf directory."""
    cache = tmp_path / "mounted cache"
    cache.mkdir()
    (edit_launcher.home / ".cache").symlink_to(cache, target_is_directory=True)

    result, events = edit_launcher.run()

    assert result.returncode == 0, result.stderr
    assert [event["kind"] for event in events] == ["builder", "trainer"]


@pytest.mark.parametrize("code", [1, 23, 134])
def test_builder_nonzero_exit_stops_before_training(edit_launcher, code):
    """Propagate builder failures, including native aborts, without starting training."""
    result, events = edit_launcher.run({"STUB_BUILDER_EXIT": str(code)})

    assert result.returncode == code
    assert [event["kind"] for event in events] == ["builder"]


@pytest.mark.parametrize("code", [1, 23, 134])
def test_training_nonzero_exit_is_preserved(edit_launcher, tmp_path, code):
    """Propagate trainer failures without rewriting the caller checkpoint."""
    model = _checkpoint(tmp_path / "user checkpoint")
    before = _files(model)

    result, events = edit_launcher.run({"MODEL_PATH": str(model), "STUB_TRAINER_EXIT": str(code)})

    assert result.returncode == code
    assert [event["kind"] for event in events] == ["trainer"]
    assert _files(model) == before


@pytest.mark.parametrize("code", [0, 7, 134])
def test_real_edit_delegate_preserves_paths_and_exit(edit_launcher, tmp_path, code):
    """Exercise the shared GPU delegate's argv with spaces in model/data paths."""
    shared = _LAUNCHER.with_name("run_flowgrpo_qwen_image_edit.sh")
    shutil.copyfile(shared, edit_launcher.script_dir / shared.name)
    bins = tmp_path / "commands"
    bins.mkdir()
    capture = tmp_path / "trainer-argv.json"
    python = bins / "python3"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "if len(sys.argv) > 2 and sys.argv[1] == '-m':\n"
        "    Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[3:]))\n"
        "    raise SystemExit(int(os.environ['TRAIN_EXIT']))\n"
    )
    python.chmod(0o755)
    model = _checkpoint(tmp_path / "user model")
    data = tmp_path / "user data"
    result, _ = edit_launcher.run(
        {
            "MODEL_PATH": str(model),
            "DATA_DIR": str(data),
            "CAPTURE": str(capture),
            "TRAIN_EXIT": str(code),
            "PATH": str(bins) + os.pathsep + os.environ["PATH"],
        }
    )
    assert result.returncode == code, result.stdout + result.stderr
    argv = json.loads(capture.read_text())
    assert f"actor_rollout_ref.model.path={model}" in argv
    assert f"data.train_files={data / 'train.parquet'}" in argv
    assert f"data.val_files={data / 'test.parquet'}" in argv
