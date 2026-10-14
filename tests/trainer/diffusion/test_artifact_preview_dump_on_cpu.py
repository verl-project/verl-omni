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

import json
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from verl_omni.pipelines.rollout_media import MediaArtifact, MediaSpec
from verl_omni.trainer.diffusion import ray_diffusion_trainer as v0
from verl_omni.trainer.diffusion.v1.trainer_base import PolicyGradientDiffusionTrainerV1 as V1
from verl_omni.utils import tracking


def _dump_image_previews(tmp_path, version, latents, previews, max_samples=None):
    context = SimpleNamespace(global_steps=5)
    dump = v0.BaseRayDiffusionTrainer._dump_generations if version == 0 else V1._dump_generations
    with ThreadPoolExecutor(max_workers=1) as executor:
        context._dump_executor = executor
        context._dump_futures = []
        context._drain_dump_futures = lambda: None
        dump(
            context,
            inputs=["one", "two"],
            outputs=latents,
            gts=[None, None],
            scores=[1, 2],
            reward_extra_infos_dict={},
            dump_path=str(tmp_path),
            max_samples=max_samples,
            previews=previews,
        )
        for future, _ in context._dump_futures:
            future.result()

    return [json.loads(line) for line in (tmp_path / "5.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("version", [0, 1])
@pytest.mark.parametrize("latent_shape", [(2, 16), (2, 16, 2, 2)])
@pytest.mark.parametrize("max_samples", [None, 1])
@pytest.mark.parametrize("channels", [1, 3, 4])
def test_latent_primary_dumps_decoded_image_previews(tmp_path, version, latent_shape, max_samples, channels):
    latents = torch.randn(latent_shape, dtype=torch.bfloat16)
    before = latents.clone()
    previews = [
        MediaArtifact(MediaSpec("image", "decoded", "CHW"), torch.full((channels, 4, 5), 37, dtype=torch.uint8)),
        MediaArtifact(MediaSpec("image", "decoded", "HWC"), torch.full((2, 3, channels), 211, dtype=torch.uint8)),
    ]
    rows = _dump_image_previews(tmp_path, version, latents, previews, max_samples)
    assert len(rows) == (2 if max_samples is None else max_samples)
    for row, size, value in zip(rows, [(5, 4), (3, 2)], [37, 211], strict=False):
        with Image.open(row["output"]) as image:
            assert image.size == size
            assert image.getpixel((0, 0)) == (value, value, value)
    torch.testing.assert_close(latents, before)


@pytest.mark.parametrize("version", [0, 1])
def test_image_conversion_failure_is_recorded_per_sample(monkeypatch, tmp_path, version):
    original_numpy = torch.Tensor.numpy
    calls = 0

    def numpy(tensor):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("simulated image conversion failure")
        return original_numpy(tensor)

    monkeypatch.setattr(torch.Tensor, "numpy", numpy)
    previews = [
        MediaArtifact(MediaSpec("image", "decoded", "CHW"), torch.zeros(3, 2, 2, dtype=torch.uint8)) for _ in range(2)
    ]
    rows = _dump_image_previews(tmp_path, version, None, previews)
    assert rows[0]["output"] is None
    assert rows[0]["image_export_error"] == "RuntimeError: simulated image conversion failure"
    assert Path(rows[1]["output"]).is_file()
    assert rows[1]["image_export_error"] is None


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
def test_intentionally_absent_previews_skip_validation_logging(monkeypatch, version):
    context = SimpleNamespace(
        global_steps=1,
        config=OmegaConf.create({"trainer": {"logger": ["wandb"], "log_val_generations": 2}}),
        validation_generations_logger=Mock(),
    )
    encoder = Mock()
    wrapper = Mock()
    media_logger = Mock()
    validate_outputs = Mock()
    monkeypatch.setattr(tracking, "_export_video", encoder)
    monkeypatch.setattr(v0, "wrap_val_samples_for_wandb", wrapper)
    monkeypatch.setattr(v0, "log_wandb_media", media_logger)
    monkeypatch.setattr(v0, "_validate_generation_outputs", validate_outputs)
    log = v0.BaseRayDiffusionTrainer._maybe_log_val_generations if version == 0 else V1._maybe_log_val_generations
    latents = torch.zeros(2, 16, 2, 2, dtype=torch.bfloat16)

    log(context, ["one", "two"], latents, [1, 2], previews=[None, None])

    encoder.assert_not_called()
    wrapper.assert_not_called()
    media_logger.assert_not_called()
    context.validation_generations_logger.log.assert_not_called()
    validate_outputs.assert_not_called()


@pytest.mark.parametrize("version", [0, 1])
def test_partially_absent_previews_still_fail_validation_logging(monkeypatch, version):
    context = SimpleNamespace(
        global_steps=1,
        config=OmegaConf.create({"trainer": {"logger": ["wandb"], "log_val_generations": 2}}),
        validation_generations_logger=Mock(),
    )
    encoder = Mock()
    monkeypatch.setattr(tracking, "_export_video", encoder)
    preview = MediaArtifact(MediaSpec("image", "decoded", "CHW"), torch.zeros(3, 2, 2, dtype=torch.uint8))
    log = v0.BaseRayDiffusionTrainer._maybe_log_val_generations if version == 0 else V1._maybe_log_val_generations

    with pytest.raises(ValueError, match="Preview sample=1 is missing"):
        log(context, ["one", "two"], torch.zeros(2, 16, 2, 2), [1, 2], previews=[preview, None])

    encoder.assert_not_called()
    context.validation_generations_logger.log.assert_not_called()


@pytest.mark.parametrize("version", [0, 1])
@pytest.mark.parametrize("invalid_audio", [torch.zeros(1, 2, 8), torch.zeros(1, 8)])
def test_capped_dump_validates_audio_for_unretained_rows(monkeypatch, tmp_path, version, invalid_audio):
    context = SimpleNamespace(global_steps=1, _dump_executor=Mock())
    encoder = Mock()
    monkeypatch.setattr(v0, "_export_video", encoder)
    dump = v0.BaseRayDiffusionTrainer._dump_generations if version == 0 else V1._dump_generations
    sample_rates = [32000, 32000 if invalid_audio.ndim == 3 else 0]
    with pytest.raises(ValueError, match="audio"):
        dump(
            context,
            ["one", "two"],
            torch.zeros(2, 4, 3, 2, 2, dtype=torch.uint8),
            [None, None],
            [1, 2],
            {},
            str(tmp_path),
            max_samples=1,
            media_kind="video",
            audios=[torch.zeros(1, 8), invalid_audio],
            audio_sample_rates=sample_rates,
        )
    encoder.assert_not_called()
    context._dump_executor.submit.assert_not_called()
    assert not (tmp_path / "1").exists()


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
    queued = context._dump_executor.submit.call_args.args
    assert queued[3] is None  # No unused primary/training tensor is retained.
    assert queued[6]["uid"] == ["0"]
    assert len(queued[13]) == 1
    copy = queued[13][0].data
    assert copy.data_ptr() != pixels[0].data_ptr()
    assert copy.untyped_storage().nbytes() == pixels[0].numel()
    V1._dump_generations(context, **args)
    assert context._dump_executor.submit.call_count == 1
    assert "previous dump is still running" in caplog.text
    future.set_result(None)
    context._drain_dump_futures()
    assert context._dump_futures == []


@pytest.mark.parametrize("version", [0, 1])
def test_mixed_preview_shapes_do_not_abort_the_dump(monkeypatch, tmp_path, version):
    """Media export is best-effort; differing preview shapes must be recorded, not raised."""
    captured = []

    def export(output, path, **kwargs):
        captured.append(output)
        Path(path).write_bytes(b"video")

    monkeypatch.setattr(v0, "_export_video", export)
    previews = [
        MediaArtifact(MediaSpec("video", "decoded", "TCHW", fps=12), torch.zeros(3, 3, 4, 5, dtype=torch.uint8)),
        MediaArtifact(MediaSpec("video", "decoded", "TCHW", fps=12), torch.zeros(2, 3, 2, 2, dtype=torch.uint8)),
    ]
    args = dict(
        inputs=["one", "two"],
        outputs=torch.randn(2, 16, 2, 2, 2, dtype=torch.bfloat16),
        gts=[None, None],
        scores=[1, 2],
        reward_extra_infos_dict={},
        dump_path=str(tmp_path),
        media_kind="video",
        previews=previews,
    )
    context = SimpleNamespace(global_steps=5)
    if version == 0:
        v0.BaseRayDiffusionTrainer._dump_generations(context, **args)
    else:
        with ThreadPoolExecutor(max_workers=1) as executor:
            context._dump_executor = executor
            context._dump_futures = []
            context._drain_dump_futures = lambda: None
            V1._dump_generations(context, **args)
            for future, _ in context._dump_futures:
                future.result()

    assert len(captured) == 2
    assert [item.data.shape for item in captured] == [(3, 3, 4, 5), (2, 3, 2, 2)]
    assert (tmp_path / "5.jsonl").is_file()
