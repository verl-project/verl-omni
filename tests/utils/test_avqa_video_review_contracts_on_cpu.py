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
"""AVQA conversion publication and launcher entry contracts."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[2]
CONVERTER = SOURCE_ROOT / "examples/gspo_trainer/data_process/avqa_video.py"
LAUNCHER = SOURCE_ROOT / "examples/gspo_trainer/qwen3_omni/run_qwen3_omni_megatron_avqa_video_separate_async.sh"
FILENAMES = ("train.parquet", "validation.parquet", "manifest.json")


@pytest.fixture
def converter(monkeypatch):
    monkeypatch.syspath_prepend(str(CONVERTER.parent))
    spec = importlib.util.spec_from_file_location("avqa_review_contracts", CONVERTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def annotation(question_id):
    return {
        "id": question_id,
        "video_name": "clip",
        "question_text": "Which sound?",
        "multi_choice": ["a", "b", "c", "d"],
        "answer": 0,
    }


def publication_case(tmp_path, monkeypatch, converter, overwrite=True):
    """Prepare a publication transaction without invoking media decoders."""
    output = (tmp_path / "published").resolve()
    output.mkdir()
    unrelated = output / "notes.txt"
    unrelated.write_bytes(b"caller-owned note")
    if overwrite:
        for name in FILENAMES:
            (output / name).write_bytes(b"original " + name.encode())
    before = {name: (output / name).read_bytes() if (output / name).exists() else None for name in FILENAMES}
    train, val = tmp_path / "train.json", tmp_path / "val.json"
    train.write_text(json.dumps([annotation(1)]))
    val.write_text(json.dumps([annotation(2)]))
    arguments = [
        "avqa_video.py",
        "--train_json",
        str(train),
        "--validation_json",
        str(val),
        "--video_root",
        str(tmp_path),
        "--output_dir",
        str(output),
    ]
    if overwrite:
        arguments.append("--overwrite")
    monkeypatch.setattr(sys, "argv", arguments)

    # Media conversion is independent of publication and has its own contracts.
    def convert_split(source, video_root, destination, split, fps, max_frames):
        Path(destination).write_bytes(b"new " + split.encode())
        return {"input": 1, "kept": 1, "dropped": {}}

    monkeypatch.setattr(converter, "convert_split", convert_split)
    return output, before, unrelated


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("failed_name", FILENAMES)
def test_publication_failure_restores_original_outputs(tmp_path, monkeypatch, converter, overwrite, failed_name):
    output, before, unrelated = publication_case(tmp_path, monkeypatch, converter, overwrite)
    original_replace = Path.replace
    injected = []

    def replace(source, target):
        target = Path(target)
        if not injected and target == output / failed_name:
            injected.append(target)
            raise OSError("injected publication failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(OSError, match="injected publication failure"):
        converter.main()
    assert injected == [output / failed_name], "failure injection must reach publication"
    after = {name: (output / name).read_bytes() if (output / name).exists() else None for name in FILENAMES}
    assert after == before
    assert unrelated.read_bytes() == b"caller-owned note"
    # A recoverable publication failure must permit the same command to retry.
    converter.main()
    assert all((output / name).exists() for name in FILENAMES)


def test_backup_failure_does_not_change_published_files(tmp_path, monkeypatch, converter):
    output, before, _ = publication_case(tmp_path, monkeypatch, converter)
    copy = converter.copy2

    def fail_copy(source, target, **kwargs):
        if Path(source).name == "validation.parquet":
            raise OSError("injected backup failure")
        return copy(source, target, **kwargs)

    monkeypatch.setattr(converter, "copy2", fail_copy)
    with pytest.raises(OSError, match="injected backup failure"):
        converter.main()
    assert {name: (output / name).read_bytes() for name in FILENAMES} == before


def test_failed_rollback_retains_original_backup(tmp_path, monkeypatch, converter):
    output, before, unrelated = publication_case(tmp_path, monkeypatch, converter)
    replace = Path.replace

    def fail_replace(source, target):
        target = Path(target)
        if target == output / "manifest.json" and source.parent.name != "originals":
            raise OSError("injected publication failure")
        if source.parent.name == "originals" and target == output / "train.parquet":
            raise OSError("injected rollback failure")
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(RuntimeError, match="original files retained at") as caught:
        converter.main()
    [staging] = list(tmp_path.glob(".published-*"))
    assert str(staging / "originals") in str(caught.value)
    assert (staging / "originals/train.parquet").read_bytes() == before["train.parquet"]
    assert (output / "validation.parquet").read_bytes() == before["validation.parquet"]
    assert (output / "manifest.json").read_bytes() == before["manifest.json"]
    assert unrelated.read_bytes() == b"caller-owned note"


def test_keyboard_interrupt_rolls_back_publication(tmp_path, monkeypatch, converter):
    output, before, _ = publication_case(tmp_path, monkeypatch, converter)
    replace = Path.replace

    def interrupt(source, target):
        if Path(target) == output / "manifest.json" and source.parent.name != "originals":
            raise KeyboardInterrupt
        return replace(source, target)

    monkeypatch.setattr(Path, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        converter.main()
    assert {name: (output / name).read_bytes() for name in FILENAMES} == before


def test_interruption_after_replace_restores_changed_target(tmp_path, monkeypatch, converter):
    output, before, _ = publication_case(tmp_path, monkeypatch, converter)
    replace = Path.replace

    def interrupt_after_replace(source, target):
        result = replace(source, target)
        if source.parent.name != "originals" and Path(target) == output / "train.parquet":
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(Path, "replace", interrupt_after_replace)
    with pytest.raises(KeyboardInterrupt):
        converter.main()
    assert {name: (output / name).read_bytes() for name in FILENAMES} == before


@pytest.mark.parametrize("mode", ["missing", "identical", "symlink", "hardlink", "valid", "relative", "config-only"])
def test_invalid_dataset_paths_fail_before_trainer(tmp_path, mode):
    train, val = tmp_path / "train data.parquet", tmp_path / "val data.parquet"
    if mode not in ("missing", "config-only"):
        train.write_bytes(b"fixture")
        if mode == "identical":
            val = train
        elif mode == "symlink":
            val.symlink_to(train)
        elif mode == "hardlink":
            val.hardlink_to(train)
        else:
            val.write_bytes(b"validation fixture")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    git = binaries / "git"
    git.write_text("#!/bin/sh\nprintf '%s\\n' source-receipt\n")
    git.chmod(0o755)
    boundary = tmp_path / "trainer-entered"
    interpreter = tmp_path / "python-sensor"
    interpreter.write_text(
        '#!/bin/sh\ncase "$*" in *verl_omni.trainer.main_omni*) printf "%s\\n" "$@" > "$BOUNDARY";; esac\nexit 77\n'
    )
    interpreter.chmod(0o755)
    env = dict(
        os.environ,
        MODEL_PATH="/unused/model",
        TRAIN_FILE=train.name if mode == "relative" else str(train),
        VAL_FILE=val.name if mode == "relative" else str(val),
        PYTHON=str(interpreter),
        BOUNDARY=str(boundary),
        OUTPUT_DIR=str(tmp_path / "output"),
        PATH=str(binaries) + os.pathsep + os.environ["PATH"],
        AVQA_VIDEO_CONFIG_ONLY="1" if mode == "config-only" else "0",
        NUM_GPUS="8",
        ROLLOUT_GPUS="4",
        ROLLOUT_TP="4",
    )
    result = subprocess.run(["bash", str(LAUNCHER)], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=10)
    if mode in ("valid", "relative", "config-only"):
        assert boundary.exists(), result.stderr
        assert result.returncode == 77, result.stderr
        if mode != "config-only":
            arguments = boundary.read_text().splitlines()
            assert f"data.train_files={train.resolve()}" in arguments
            assert f"data.val_files={val.resolve()}" in arguments
        return
    assert not boundary.exists(), result.stderr
    assert result.returncode == 2, result.stderr


def test_invalid_container_is_rejected_at_conversion(tmp_path, monkeypatch, converter):
    (tmp_path / "clip.mp4").write_bytes(b"not an MP4 container")
    # A failing ffprobe isolates converter policy without depending on FFmpeg installation.
    binary = tmp_path / "ffprobe"
    binary.write_text("#!/bin/sh\nexit 1\n")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    source = tmp_path / "train.json"
    source.write_text(json.dumps([annotation(1)]))
    with pytest.raises(ValueError, match="invalid_media"):
        converter.convert_split(source, tmp_path, tmp_path / "train.parquet", "train")
    assert not (tmp_path / "train.parquet").exists()


@pytest.mark.parametrize("status", [None, "missing_audio_stream", "invalid_media"])
def test_conversion_caches_audio_probe_and_counts_drops(tmp_path, monkeypatch, converter, status):
    for clip in ("clip", "good"):
        (tmp_path / f"{clip}.mp4").touch()
    source = tmp_path / "train.json"
    source.write_text(json.dumps([annotation(1), annotation(2), {**annotation(3), "video_name": "good"}]))
    probes = []

    def probe(video):
        probes.append(Path(video).name)
        return status if Path(video).stem == "clip" else None

    monkeypatch.setattr(converter, "probe_audio_stream", probe)
    target = tmp_path / "train.parquet"
    summary = converter.convert_split(source, tmp_path, target, "train")
    assert probes == ["clip.mp4", "good.mp4"]
    assert summary == {
        "input": 3,
        "kept": 3 if status is None else 1,
        "dropped": {} if status is None else {status: 2},
    }
    import pandas as pd

    assert len(pd.read_parquet(target)) == summary["kept"]


def test_missing_decoder_is_actionable_and_does_not_publish(tmp_path, monkeypatch, converter):
    (tmp_path / "clip.mp4").touch()
    source = tmp_path / "train.json"
    source.write_text(json.dumps([annotation(1)]))

    def unavailable(video):
        raise RuntimeError("ffprobe is required")

    monkeypatch.setattr(converter, "probe_audio_stream", unavailable)
    target = tmp_path / "train.parquet"
    with pytest.raises(RuntimeError, match="ffprobe is required"):
        converter.convert_split(source, tmp_path, target, "train")
    assert not target.exists()


@pytest.mark.parametrize("as_package", [False, True])
def test_converter_help_from_script_or_package(as_package, tmp_path):
    command = [sys.executable, str(CONVERTER)]
    cwd = SOURCE_ROOT
    if as_package:
        # Check relative imports without assuming ownership of the generic
        # "examples" namespace, which dependencies may also define.
        package = tmp_path / "avqa_cli_fixture"
        package.mkdir()
        (package / "__init__.py").write_text("")
        for name in ("avqa_video.py", "nextqa.py"):
            (package / name).write_bytes((CONVERTER.parent / name).read_bytes())
        command = [sys.executable, "-m", "avqa_cli_fixture.avqa_video"]
        cwd = tmp_path
    result = subprocess.run(command + ["--help"], cwd=cwd, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "--train_json" in result.stdout
