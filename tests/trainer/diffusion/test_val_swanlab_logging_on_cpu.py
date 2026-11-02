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

"""CPU tests for the SwanLab branch of ``BaseRayDiffusionTrainer._maybe_log_val_generations``."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

pytest.importorskip("diffusers")
wandb = pytest.importorskip("wandb")
from omegaconf import OmegaConf  # noqa: E402  (after importorskip guards)
from verl.utils.tracking import Tracking, ValidationGenerationsLogger  # noqa: E402

from tests.utils.swanlab_stub import use_swanlab  # noqa: E402
from verl_omni.pipelines.rollout_media import MediaArtifact, MediaSpec  # noqa: E402
from verl_omni.trainer.diffusion.ray_diffusion_trainer import BaseRayDiffusionTrainer  # noqa: E402


@pytest.fixture
def swanlab(monkeypatch):
    return use_swanlab(monkeypatch)


class _RecordingValLogger:
    def __init__(self):
        self.calls = []

    def log(self, loggers, samples, step):
        self.calls.append((list(loggers), samples, step))


def _stub(loggers, *, val_logger, global_steps=7, log_val_generations=2):
    return SimpleNamespace(
        config=OmegaConf.create(
            {
                "trainer": {
                    "log_val_generations": log_val_generations,
                    "logger": loggers,
                    "video_fps": 8,
                    "validation_data_dir": None,
                    "default_local_dir": None,
                }
            }
        ),
        global_steps=global_steps,
        validation_generations_logger=val_logger,
    )


def _clips(n=2, t=4, h=32, w=32):
    clips = torch.zeros(n, t, 3, h, w, dtype=torch.uint8)
    # PIL merges identical GIF frames, so each frame gets its own red level.
    clips[:, :, 0] = (191 + torch.arange(t, dtype=torch.uint8) * 4).view(t, 1, 1)
    return clips


def _log(stub, outputs, **kwargs):
    n = outputs.shape[0] if hasattr(outputs, "shape") else len(outputs)
    BaseRayDiffusionTrainer._maybe_log_val_generations(
        stub, [f"prompt {i}" for i in range(n)], outputs, [float(i) for i in range(n)], **kwargs
    )


def test_swanlab_only_logs_media_and_gives_the_table_text_rows(monkeypatch, swanlab):
    logged = []
    monkeypatch.setattr(swanlab, "log", lambda data, step=None: logged.append((data, step)))
    val_logger = _RecordingValLogger()

    _log(_stub(["console", "swanlab"], val_logger=val_logger), _clips(), media_kinds=["video"] * 2)

    assert len(logged) == 1
    media, step = logged[0]
    assert step == 7
    assert set(media) == {"val/videos/sample_1", "val/videos/sample_2"}
    assert [(call[0], call[2]) for call in val_logger.calls] == [(["console"], 7), (["swanlab"], 7)]
    rows = val_logger.calls[1][1]
    assert sorted(row[1] for row in rows) == ["val/videos/sample_1", "val/videos/sample_2"]


def test_wandb_and_swanlab_each_get_their_own_rows(monkeypatch, tmp_path, swanlab):
    monkeypatch.setattr(wandb, "run", object())
    monkeypatch.setattr(wandb, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(wandb, "Video", lambda path, **kwargs: SimpleNamespace(path=path))
    swanlab_logged = []
    monkeypatch.setattr(swanlab, "log", lambda data, step=None: swanlab_logged.append(data))
    val_logger = _RecordingValLogger()
    stub = _stub(["wandb", "swanlab"], val_logger=val_logger)
    stub.config.trainer.validation_data_dir = str(tmp_path)

    _log(stub, _clips(), media_kinds=["video"] * 2)

    assert [call[0] for call in val_logger.calls] == [["wandb"], ["swanlab"]]
    assert set(swanlab_logged[0]) == {"val/videos/sample_1", "val/videos/sample_2"}
    assert list(tmp_path.glob("wandb_val_media/global_step_7/*.mp4"))


def test_config_logger_list_passes_through_without_swanlab():
    val_logger = _RecordingValLogger()
    stub = _stub(["console", "tensorboard"], val_logger=val_logger, log_val_generations=1)

    _log(stub, _clips(n=1), media_kinds=["video"])

    assert [call[0] for call in val_logger.calls] == [["console", "tensorboard"]]
    assert isinstance(val_logger.calls[0][1][0][1], torch.Tensor)


def test_swanlab_log_failure_does_not_stop_the_table(monkeypatch, swanlab):
    def fail(*args, **kwargs):
        raise RuntimeError("no active run")

    monkeypatch.setattr(swanlab, "log", fail)
    val_logger = _RecordingValLogger()

    _log(_stub(["swanlab"], val_logger=val_logger), _clips(), media_kinds=["video"] * 2)

    assert [call[0] for call in val_logger.calls] == [[], ["swanlab"]]


def test_swanlab_previews_replace_outputs(monkeypatch, swanlab):
    logged = []
    monkeypatch.setattr(swanlab, "log", lambda data, step=None: logged.append(data))
    val_logger = _RecordingValLogger()
    preview = torch.zeros(3, 16, 16, dtype=torch.uint8)
    previews = [MediaArtifact(MediaSpec("image", "decoded", "CHW"), preview.clone()) for _ in range(2)]

    _log(_stub(["swanlab"], val_logger=val_logger), [None, None], previews=previews)

    assert set(logged[0]) == {"val/images/sample_1", "val/images/sample_2"}


def test_a_real_offline_run_writes_scalars_and_media(monkeypatch, tmp_path):
    pytest.importorskip("swanlab")
    monkeypatch.setenv("SWANLAB_MODE", "offline")
    monkeypatch.setenv("SWANLAB_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("SWANLAB_API_KEY", raising=False)

    tracking = Tracking("verl_omni_test", "swanlab_local", default_backend=["swanlab"], config={"seed": 1})
    audio = [torch.zeros(1, 800), torch.zeros(1, 800)]
    try:
        tracking.log({"val/reward": 1.0}, step=7)
        _log(
            _stub(["swanlab"], val_logger=ValidationGenerationsLogger()),
            _clips(),
            media_kinds=["video"] * 2,
            audios=audio,
            audio_sample_rates=[8000, 8000],
        )
    finally:
        tracking.finish()

    run_dirs = list(Path(tmp_path).glob("run-*"))
    assert len(run_dirs) == 1
    written = {path.suffix for path in (run_dirs[0] / "media").rglob("*") if path.is_file()}
    assert {".gif", ".wav", ".json"} <= written
    gif = next((run_dirs[0] / "media").rglob("*.gif"))
    assert Image.open(gif).n_frames == 4
