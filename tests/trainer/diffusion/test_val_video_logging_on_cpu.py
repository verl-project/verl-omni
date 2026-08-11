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
"""CPU tests for the wandb branch of ``BaseRayDiffusionTrainer._maybe_log_val_generations``."""

import os
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("diffusers")
wandb = pytest.importorskip("wandb")
imageio = pytest.importorskip("imageio")
from omegaconf import OmegaConf  # noqa: E402  (after importorskip guards)

from verl_omni.trainer.diffusion.ray_diffusion_trainer import BaseRayDiffusionTrainer  # noqa: E402


class _RecordingValLogger:
    """Records wrapped samples in place of a live wandb run."""

    def __init__(self):
        self.calls = []

    def log(self, loggers, samples, step):
        self.calls.append((list(loggers), samples, step))


def _run_val_logging(
    outputs, *, log_val_generations, video_fps=8, global_steps=0, validation_data_dir=None, default_local_dir=None
):
    """Invoke the unbound ``_maybe_log_val_generations`` with a minimal stub ``self``."""
    val_logger = _RecordingValLogger()
    stub = SimpleNamespace(
        config=OmegaConf.create(
            {
                "trainer": {
                    "log_val_generations": log_val_generations,
                    "logger": ["wandb"],
                    "video_fps": video_fps,
                    "validation_data_dir": validation_data_dir,
                    "default_local_dir": default_local_dir,
                }
            }
        ),
        global_steps=global_steps,
        validation_generations_logger=val_logger,
    )
    n = outputs.shape[0] if hasattr(outputs, "shape") else len(outputs)
    inputs = [f"prompt {i}" for i in range(n)]
    scores = [float(i) for i in range(n)]
    BaseRayDiffusionTrainer._maybe_log_val_generations(stub, inputs, outputs, scores)
    return val_logger


def _warm_clips(n, t=8, h=32, w=32):
    """``n`` solid warm clips ``[N, T, C, H, W]`` with R > G > B, so an inverted encode is detectable."""
    clips = torch.zeros(n, t, 3, h, w)
    clips[:, :, 0] = 0.75  # R
    clips[:, :, 1] = 0.35  # G
    clips[:, :, 2] = 0.15  # B
    return clips


class TestMaybeLogValGenerationsWandb:
    def test_wandb_video_gets_real_decodable_mp4_in_persistent_dir(self, monkeypatch, tmp_path):
        """wandb.Video gets a real mp4 that remains available after log() returns."""
        captured = []
        logged_media = []

        class _FakeVideo:
            def __init__(self, data_or_path, *args, **kwargs):
                assert isinstance(data_or_path, str), f"expected a file path, got {type(data_or_path)}"
                assert os.path.isfile(data_or_path), f"wandb.Video got a non-existent path: {data_or_path}"
                reader = imageio.get_reader(data_or_path)
                frame = reader.get_data(4)
                reader.close()
                rgb = tuple(float(frame[..., c].mean()) for c in range(3))
                captured.append(SimpleNamespace(path=data_or_path, kwargs=dict(kwargs), rgb=rgb))
                self.data_or_path = data_or_path

        monkeypatch.setattr(wandb, "Video", _FakeVideo)
        monkeypatch.setattr(wandb, "run", object())
        monkeypatch.setattr(
            wandb, "log", lambda payload, step, commit=False: logged_media.append((payload, step, commit))
        )

        outputs = _warm_clips(2)  # [2, T, C, H, W]
        validation_data_dir = tmp_path / "val"
        val_logger = _run_val_logging(
            outputs, log_val_generations=2, global_steps=7, validation_data_dir=str(validation_data_dir)
        )

        # One wandb.Video per retained video sample, each a real mp4 tagged format=mp4.
        assert len(captured) == 2
        for c in captured:
            assert c.path.endswith(".mp4")
            assert c.kwargs.get("format") == "mp4"
            r, g, b = c.rgb
            assert r > g > b, f"channel order not preserved in wandb mp4: R={r:.1f} G={g:.1f} B={b:.1f}"
            assert r > 128 and b < 128, f"warm frame inverted in wandb mp4: R={r:.1f} B={b:.1f}"

        # Files live under validation_data_dir so wandb can read them during async artifact upload.
        media_dirs = {os.path.dirname(c.path) for c in captured}
        expected_dir = validation_data_dir / "wandb_val_media" / "global_step_7"
        assert media_dirs == {str(expected_dir)}
        assert expected_dir.is_dir()
        assert all(os.path.isfile(c.path) for c in captured)

        # Videos are top-level media because wandb 0.27 serializes table videos as the string "Video" offline.
        assert len(logged_media) == 1
        media_payload, media_step, commit = logged_media[0]
        assert media_step == 7 and commit is False
        assert list(media_payload) == ["val/videos/sample_1", "val/videos/sample_2"]
        assert all(isinstance(media, _FakeVideo) for media in media_payload.values())

        # The table contains stable keys pointing to the top-level media, not unsupported nested Video objects.
        assert len(val_logger.calls) == 1
        loggers, samples, step = val_logger.calls[0]
        assert loggers == ["wandb"] and step == 7
        assert len(samples) == 2
        assert [media_key for _inp, media_key, _score in samples] == list(media_payload)

    def test_wandb_video_uses_default_local_dir_when_validation_dir_is_unset(self, monkeypatch, tmp_path):
        captured = []

        class _FakeVideo:
            def __init__(self, data_or_path, *args, **kwargs):
                captured.append(data_or_path)

        monkeypatch.setattr(wandb, "Video", _FakeVideo)

        default_local_dir = tmp_path / "run"
        _run_val_logging(
            _warm_clips(1), log_val_generations=1, global_steps=11, default_local_dir=str(default_local_dir)
        )

        expected_path = default_local_dir / "wandb_val_media" / "global_step_11" / "0.mp4"
        assert captured == [str(expected_path)]
        assert expected_path.is_file()

    def test_real_wandb_video_ingests_our_production_mp4(self, tmp_path):
        """The REAL wandb.Video ingests a production-path mp4 from disk (size + sha256) with no moviepy."""
        from diffusers.utils import export_to_video

        from verl_omni.utils.reward_score.reward_utils import video_tensor_to_pil_frames

        clip = _warm_clips(1)[0]  # [T, C, H, W]
        path = str(tmp_path / "clip.mp4")
        export_to_video(video_tensor_to_pil_frames(clip), path, fps=8)
        assert os.path.getsize(path) > 0

        vid = wandb.Video(path, format="mp4")  # no fps= (ignored + warned for file paths)

        assert vid._path == path
        assert vid._format == "mp4"
        # wandb read our real file: the size + hash it will upload match the mp4 on disk.
        assert vid._size == os.path.getsize(path)
        assert isinstance(vid._sha256, str) and len(vid._sha256) == 64
