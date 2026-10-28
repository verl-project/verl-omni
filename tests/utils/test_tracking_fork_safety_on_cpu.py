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

"""CPU regression tests: a fork during a background video export must not hang the export."""

import multiprocessing
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest
import torch

from verl_omni.utils import tracking

if "fork" not in multiprocessing.get_all_start_methods():
    pytest.skip("requires the fork start method", allow_module_level=True)


def _frames():
    # Larger than a pipe buffer, so a stdin-fed export would still be writing when the fork happens.
    return torch.zeros(4, 3, 128, 128, dtype=torch.uint8)


def test_fork_during_export_does_not_hang_it(tmp_path):
    started = tmp_path / "ffmpeg_started"
    release = tmp_path / "ffmpeg_release"
    # Stands in for ffmpeg: it finishes only once every holder of its stdin closes it.
    fake_ffmpeg = tmp_path / "fake_ffmpeg"
    fake_ffmpeg.write_text(
        f'#!/bin/sh\ntouch "{started}"\nwhile [ ! -e "{release}" ]; do sleep 0.01; done\ncat > /dev/null\n'
    )
    fake_ffmpeg.chmod(0o755)

    errors = []

    def export():
        try:
            tracking._export_video(_frames(), str(tmp_path / "sample.mp4"), fps=24, ffmpeg_exe=str(fake_ffmpeg))
        except Exception as error:
            errors.append(error)

    exporter = threading.Thread(target=export, daemon=True)
    exporter.start()
    deadline = time.monotonic() + 10
    while not started.exists():
        assert time.monotonic() < deadline, "fake ffmpeg did not start"
        time.sleep(0.01)

    # A long-lived fork taken mid-export, like a DataLoader worker.
    child = multiprocessing.get_context("fork").Process(target=time.sleep, args=(60,), daemon=True)
    child.start()
    try:
        release.touch()
        exporter.join(timeout=10)
        assert not exporter.is_alive(), "video export hung after a concurrent fork"
        assert errors == []
    finally:
        release.touch()
        child.terminate()
        child.join()
        exporter.join(timeout=10)


@pytest.mark.skipif(not getattr(subprocess, "_USE_POSIX_SPAWN", False), reason="CPython has no posix_spawn path here")
def test_export_spawns_ffmpeg_without_an_exec_status_pipe(monkeypatch, tmp_path):
    # The fork/exec fallback opens an exec-status pipe that a concurrent fork can inherit,
    # which blocks Popen itself; posix_spawn has no such pipe.
    fake_ffmpeg = tmp_path / "fake_ffmpeg"
    fake_ffmpeg.write_text("#!/bin/sh\n")
    fake_ffmpeg.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    def fail_fork_exec(*args, **kwargs):
        raise AssertionError("ffmpeg was started through fork/exec")

    monkeypatch.setattr(subprocess, "_fork_exec", fail_fork_exec)

    for ffmpeg_exe in (str(fake_ffmpeg), "fake_ffmpeg"):
        tracking._export_video(_frames(), str(tmp_path / "sample.mp4"), fps=24, ffmpeg_exe=ffmpeg_exe)


def test_export_gives_ffmpeg_no_pipe_and_removes_spooled_frames(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        frames_path = Path(command[command.index("-i") + 1])
        calls.append((kwargs, frames_path, frames_path.stat().st_size))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(tracking.subprocess, "run", fake_run)
    unrelated = tmp_path / "sample.rgb"
    unrelated.write_bytes(b"keep")

    tracking._export_video(_frames(), str(tmp_path / "sample.mp4"), fps=24, ffmpeg_exe="/usr/bin/ffmpeg")

    ((kwargs, frames_path, size),) = calls
    assert "input" not in kwargs
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["close_fds"] is False
    assert kwargs["timeout"] > 0
    assert size == 4 * 128 * 128 * 3
    assert frames_path != unrelated
    assert not frames_path.exists()
    assert unrelated.read_bytes() == b"keep"
