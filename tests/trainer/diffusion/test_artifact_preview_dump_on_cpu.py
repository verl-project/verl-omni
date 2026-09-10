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
"""A latent response can be scored while an independently named preview is dumped."""

import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from verl_omni.pipelines.rollout_artifacts import MediaArtifact
from verl_omni.pipelines.rollout_media import MediaSpec
from verl_omni.trainer.diffusion import ray_diffusion_trainer as v0
from verl_omni.trainer.diffusion.v1.trainer_base import PolicyGradientDiffusionTrainerV1 as V1
from verl_omni.utils import tracking


@pytest.mark.parametrize("version", [0, 1])
def test_latent_primary_dumps_declared_preview_without_axis_guessing(monkeypatch, tmp_path, version):
    preview = MediaArtifact(MediaSpec("video", "decoded", "TCHW", fps=12), torch.zeros(3, 4, 2, 5, dtype=torch.uint8))
    latents = torch.randn(2, 16, 2, 2, 2, dtype=torch.bfloat16)
    before = latents.clone()
    captured = []

    def export(output, path, **kwargs):
        captured.append(output)
        Path(path).write_bytes(b"video")

    monkeypatch.setattr(v0, "_export_video", export)
    args = dict(
        inputs=["one", "two"],
        outputs=latents,
        gts=[None, None],
        scores=[1, 2],
        reward_extra_infos_dict={},
        dump_path=str(tmp_path),
        max_samples=1,
        media_kind="video",
        previews=[preview, preview],
    )
    context = SimpleNamespace(global_steps=5)
    if version == 0:
        v0.BaseRayDiffusionTrainer._dump_generations(context, **args)
    else:
        with ThreadPoolExecutor(max_workers=1) as executor:
            context._dump_executor = executor
            context._dump_futures = []
            context._drain_dump_futures = lambda: None
            context._write_generations = V1._write_generations
            V1._dump_generations(context, **args)
            for future, step in context._dump_futures:
                future.result()
                assert step == 5
    assert len(captured) == 1
    assert captured[0].data.shape == (3, 4, 2, 5)  # T=3, C=4 is not CTHW
    assert captured[0].spec.fps == 12
    torch.testing.assert_close(latents, before)
    assert (tmp_path / "5.jsonl").is_file()


def test_invalid_preview_fails_before_background_submission(tmp_path):
    context = SimpleNamespace(_dump_executor=Mock())
    latent = MediaArtifact(MediaSpec("video", "latent", "CTHW"), torch.zeros(16, 2, 2, 2))
    with pytest.raises(ValueError, match="decoded visual"):
        V1._dump_generations(
            context, ["one"], torch.zeros(1, 16, 2, 2, 2), [None], [0], {}, str(tmp_path), previews=[latent]
        )
    context._dump_executor.submit.assert_not_called()


@pytest.mark.parametrize("export_fails", [False, True])
def test_wandb_uses_named_preview_and_retains_best_effort_io(monkeypatch, tmp_path, export_fails):
    preview = MediaArtifact(MediaSpec("video", "decoded", "TCHW", fps=12), torch.zeros(3, 3, 2, 5, dtype=torch.uint8))
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Video=lambda path, **kwargs: path))

    def export(output, path, **kwargs):
        assert output is preview
        if export_fails:
            raise OSError("simulated encoder failure")
        Path(path).write_bytes(b"video")

    monkeypatch.setattr(tracking, "_export_video", export)
    wrapped, _, media = tracking.wrap_val_samples_for_wandb([("prompt", preview, 1.0)], output_dir=str(tmp_path))
    assert bool(media) is not export_fails
    if export_fails:
        assert "simulated encoder failure" in wrapped[0][1]
    else:
        assert wrapped[0][1] == "val/videos/sample_1"


def test_artifact_export_preserves_fractional_fps(monkeypatch, tmp_path):
    preview = MediaArtifact(
        MediaSpec("video", "decoded", "TCHW", fps=29.97), torch.zeros(3, 3, 2, 4, dtype=torch.uint8)
    )
    commands = []
    monkeypatch.setattr(tracking.subprocess, "run", lambda command, **kwargs: commands.append(command))
    tracking._export_video(preview, str(tmp_path / "preview.mp4"), fps=24, ffmpeg_exe="/fake/ffmpeg")
    command = commands[0]
    assert command[command.index("-r") + 1] == "29.97"


def test_tracking_explicit_layout_bypasses_legacy_normalizer(monkeypatch):
    monkeypatch.setattr(tracking, "normalize_video_tensor", lambda data: pytest.fail("inferred artifact layout"))
    canonical = torch.arange(3 * 3 * 2 * 5, dtype=torch.uint8).reshape(3, 3, 2, 5)
    preview = MediaArtifact(MediaSpec("video", "decoded", "TCHW", fps=24), canonical)
    frames, width, height = tracking._video_tensor_to_rgb24(preview)
    assert frames.shape == (3, 2, 5, 3) and (width, height) == (5, 2)
    torch.testing.assert_close(torch.from_numpy(frames).permute(0, 3, 1, 2), canonical)


@pytest.mark.parametrize("version", [0, 1])
def test_intentionally_absent_preview_skips_dump_without_using_latents(tmp_path, version):
    context = SimpleNamespace(global_steps=1, _dump_executor=Mock())
    dump = v0.BaseRayDiffusionTrainer._dump_generations if version == 0 else V1._dump_generations
    dump(context, ["one"], torch.zeros(1, 16, 2, 2), [None], [1], {}, str(tmp_path), previews=[None])
    context._dump_executor.submit.assert_not_called()
    assert not (tmp_path / "1.jsonl").exists()


@pytest.mark.parametrize("version", [0, 1])
def test_invalid_audio_fails_before_dump_io(tmp_path, version):
    context = SimpleNamespace(global_steps=1, _dump_executor=Mock())
    dump = v0.BaseRayDiffusionTrainer._dump_generations if version == 0 else V1._dump_generations
    with pytest.raises(ValueError, match="expected layout=CT"):
        dump(
            context,
            ["one"],
            torch.zeros(1, 4, 3, 2, 2, dtype=torch.uint8),
            [None],
            [1],
            {},
            str(tmp_path),
            media_kind="video",
            fps=24,
            audios=[torch.zeros(1, 2, 8)],
            audio_sample_rates=[32000],
        )
    context._dump_executor.submit.assert_not_called()
    assert not (tmp_path / "1").exists()


@pytest.mark.parametrize("data", [torch.zeros(3, 2, 2), torch.zeros(1, 3, 2, 2, dtype=torch.uint8)])
def test_wandb_schema_errors_are_not_observability_failures(monkeypatch, data):
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Image=Mock()))
    with pytest.raises(ValueError, match="artifact='preview'"):
        tracking.wrap_val_samples_for_wandb([("one", data, 1)], media_kinds=["image"])
    sys.modules["wandb"].Image.assert_not_called()


def test_v1_dump_queue_is_bounded_and_does_not_retain_full_batch_storage(tmp_path, caplog):
    future = Future()
    context = SimpleNamespace(global_steps=1, _dump_executor=Mock(), _dump_futures=[])
    context._dump_executor.submit.return_value = future
    context._write_generations = V1._write_generations
    context._drain_dump_futures = lambda: V1._drain_dump_futures(context)
    context._report_dump_failure = V1._report_dump_failure
    pixels = torch.zeros(4, 3, 8, 8, dtype=torch.uint8)
    previews = [MediaArtifact(MediaSpec("image", "decoded", "CHW"), row) for row in pixels]
    args = dict(
        inputs=[str(i) for i in range(4)],
        outputs=torch.zeros(4, 16, 2, 2),
        gts=[None] * 4,
        scores=[0] * 4,
        reward_extra_infos_dict={"uid": [str(i) for i in range(4)]},
        dump_path=str(tmp_path),
        previews=previews,
        max_samples=1,
    )
    V1._dump_generations(context, **args)
    queued = context._dump_executor.submit.call_args.args[1:]
    assert queued[1] is None  # No unused primary/training tensor is retained.
    assert queued[4]["uid"] == ["0"]
    assert len(queued[12]) == 1
    copy = queued[12][0].data
    assert copy.data_ptr() != pixels[0].data_ptr()
    assert copy.untyped_storage().nbytes() == pixels[0].numel()
    V1._dump_generations(context, **args)
    assert context._dump_executor.submit.call_count == 1
    assert "previous dump is still running" in caplog.text
    future.set_result(None)
    context._drain_dump_futures()
    assert context._dump_futures == []
