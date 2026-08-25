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
"""CPU tests for ``BaseRayDiffusionTrainer._dump_generations`` media handling.

The dump branches on tensor rank: 5-D video batches ``[N, T, C, H, W]`` are
written as ``{i}.mp4`` and 4-D image batches ``[N, C, H, W]`` as ``{i}.jpg``,
while ``max_samples`` bounds how many samples (and JSONL rows) are emitted.
The method only reads ``self.global_steps``, so a lightweight stub stands in
for the trainer.
"""

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# ``export_to_video`` (and its ffmpeg backend) is only needed for the video
# branch; skip the whole module rather than hard-fail where it is absent.
pytest.importorskip("diffusers")

import verl_omni.trainer.diffusion.ray_diffusion_trainer as ray_diffusion_trainer
from verl_omni.trainer.diffusion.ray_diffusion_trainer import BaseRayDiffusionTrainer


def _dump(dump_path, outputs, *, max_samples=None, global_steps=0, audios=None, audio_sample_rates=None):
    """Invoke the unbound ``_dump_generations`` with a minimal stub ``self``."""
    n = outputs.shape[0]
    stub = SimpleNamespace(global_steps=global_steps)
    inputs = [f"prompt {i}" for i in range(n)]
    gts = ["" for _ in range(n)]
    scores = [float(i) for i in range(n)]
    # An extra whose length matches the full batch must be sliced alongside it.
    reward_extra_infos_dict = {"reward": [float(i) for i in range(n)]}
    kwargs = {}
    if audios is not None:
        kwargs = {"audios": audios, "audio_sample_rates": audio_sample_rates}
    BaseRayDiffusionTrainer._dump_generations(
        stub,
        inputs,
        outputs,
        gts,
        scores,
        reward_extra_infos_dict,
        str(dump_path),
        max_samples=max_samples,
        fps=8,
        **kwargs,
    )


def _read_jsonl(dump_path, global_steps=0):
    with open(os.path.join(str(dump_path), f"{global_steps}.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


class TestDumpGenerations:
    def test_rejects_non_uint8_outputs(self, tmp_path):
        outputs = torch.zeros(1, 3, 16, 16)

        with pytest.raises(
            ValueError,
            match=r"Expected generation outputs to be a uint8 tensor, got torch\.float32\.",
        ):
            _dump(tmp_path, outputs)

    def test_video_batch_writes_one_mp4_per_sample(self, tmp_path):
        outputs = torch.randint(256, (2, 8, 3, 16, 16), dtype=torch.uint8)  # [N, T, C, H, W]
        _dump(tmp_path, outputs)

        visual = os.path.join(str(tmp_path), "0")
        mp4s = sorted(f for f in os.listdir(visual) if f.endswith(".mp4"))
        assert mp4s == ["0.mp4", "1.mp4"]

        rows = _read_jsonl(tmp_path)
        assert len(rows) == 2
        assert all(row["output"].endswith(".mp4") for row in rows)
        assert all(row["step"] == 0 for row in rows)
        # Full-length extras are carried through, sliced to the dumped count.
        assert [row["reward"] for row in rows] == [0.0, 1.0]

    def test_video_batch_muxes_generated_audio(self, tmp_path):
        from imageio_ffmpeg import get_ffmpeg_exe

        outputs = torch.randint(256, (1, 8, 3, 16, 16), dtype=torch.uint8)
        audios = torch.sin(torch.linspace(0, 100, 48_000)).reshape(1, 1, -1)
        _dump(tmp_path, outputs, audios=audios, audio_sample_rates=[48_000])

        path = os.path.join(str(tmp_path), "0", "0.mp4")
        probe = subprocess.run([get_ffmpeg_exe(), "-i", path], capture_output=True, text=True)
        assert "Video:" in probe.stderr and "Audio: aac" in probe.stderr

    def test_video_export_failure_preserves_raw_fallback_and_writes_jsonl(self, monkeypatch, tmp_path):
        outputs = torch.randint(256, (2, 8, 3, 16, 16), dtype=torch.uint8)

        def fake_export(output, output_path, **kwargs):
            if output_path.endswith("1.mp4"):
                raise subprocess.CalledProcessError(1, ["ffmpeg"])
            Path(output_path).write_bytes(b"video")

        monkeypatch.setattr(ray_diffusion_trainer, "_export_video", fake_export)
        _dump(tmp_path, outputs)

        rows = _read_jsonl(tmp_path)
        assert rows[0]["output"].endswith("0.mp4")
        assert rows[1]["output"] is None
        assert rows[1]["video_export_error"].startswith("CalledProcessError:")
        fallback_path = rows[1]["output_fallback"]
        assert fallback_path.endswith("1.pt")
        fallback = torch.load(fallback_path, weights_only=True)
        torch.testing.assert_close(fallback["video"], outputs[1])
        assert fallback["audio"] is None
        assert fallback["audio_sample_rate"] is None

    def test_image_batch_writes_one_jpg_per_sample(self, tmp_path):
        # Image regression: the 4-D path must stay byte-for-byte behaviour.
        outputs = torch.randint(256, (2, 3, 16, 16), dtype=torch.uint8)  # [N, C, H, W]
        _dump(tmp_path, outputs)

        visual = os.path.join(str(tmp_path), "0")
        jpgs = sorted(f for f in os.listdir(visual) if f.endswith(".jpg"))
        assert jpgs == ["0.jpg", "1.jpg"]
        assert not any(f.endswith(".mp4") for f in os.listdir(visual))

        rows = _read_jsonl(tmp_path)
        assert len(rows) == 2
        assert all(row["output"].endswith(".jpg") for row in rows)

    def test_max_samples_bounds_video_dump(self, tmp_path):
        outputs = torch.randint(256, (3, 8, 3, 16, 16), dtype=torch.uint8)
        _dump(tmp_path, outputs, max_samples=1)

        visual = os.path.join(str(tmp_path), "0")
        mp4s = [f for f in os.listdir(visual) if f.endswith(".mp4")]
        assert mp4s == ["0.mp4"]  # only the first sample encoded

        rows = _read_jsonl(tmp_path)
        assert len(rows) == 1
        assert rows[0]["reward"] == 0.0
