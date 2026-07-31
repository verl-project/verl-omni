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
from types import SimpleNamespace

import pytest
import torch

# ``export_to_video`` (and its ffmpeg backend) is only needed for the video
# branch; skip the whole module rather than hard-fail where it is absent.
pytest.importorskip("diffusers")

from verl_omni.trainer.diffusion.ray_diffusion_trainer import BaseRayDiffusionTrainer


def _dump(dump_path, outputs, *, max_samples=None, global_steps=0):
    """Invoke the unbound ``_dump_generations`` with a minimal stub ``self``."""
    n = outputs.shape[0]
    stub = SimpleNamespace(global_steps=global_steps)
    inputs = [f"prompt {i}" for i in range(n)]
    gts = ["" for _ in range(n)]
    scores = [float(i) for i in range(n)]
    # An extra whose length matches the full batch must be sliced alongside it.
    reward_extra_infos_dict = {"reward": [float(i) for i in range(n)]}
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
    )


def _read_jsonl(dump_path, global_steps=0):
    with open(os.path.join(str(dump_path), f"{global_steps}.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


class TestDumpGenerations:
    def test_video_batch_writes_one_mp4_per_sample(self, tmp_path):
        outputs = torch.rand(2, 8, 3, 16, 16)  # [N, T, C, H, W]
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

    def test_video_colors_not_inverted(self, tmp_path):
        """``export_to_video`` rescales NumPy input by 255 even when it is already
        ``uint8`` [0, 255], overflowing to a photographic negative (255 - x) that
        flips warm scenes to a cold "X-ray" look. The dump sidesteps this by
        handing it PIL frames (via ``video_tensor_to_pil_frames``), which are not
        rescaled; decode a solid warm clip and assert the channel order survives."""
        imageio = pytest.importorskip("imageio")

        outputs = torch.zeros(1, 8, 3, 32, 32)  # solid warm frame: R > G > B
        outputs[:, :, 0] = 0.75  # R
        outputs[:, :, 1] = 0.35  # G
        outputs[:, :, 2] = 0.15  # B
        _dump(tmp_path, outputs)

        path = os.path.join(str(tmp_path), "0", "0.mp4")
        reader = imageio.get_reader(path)  # ffmpeg backend (imageio-ffmpeg)
        frame = reader.get_data(4)  # a mid frame, [H, W, C] uint8 RGB
        reader.close()
        r, g, b = (float(frame[..., c].mean()) for c in range(3))
        # Warm source must stay warm. Under the old uint8 overflow these invert
        # (b > r, r < 128), so the ordering + range pins the fix.
        assert r > g > b, f"channel order not preserved: R={r:.1f} G={g:.1f} B={b:.1f}"
        assert r > 128 and b < 128, f"expected a warm frame, got R={r:.1f} B={b:.1f}"

    def test_image_batch_writes_one_jpg_per_sample(self, tmp_path):
        # Image regression: the 4-D path must stay byte-for-byte behaviour.
        outputs = torch.rand(2, 3, 16, 16)  # [N, C, H, W]
        _dump(tmp_path, outputs)

        visual = os.path.join(str(tmp_path), "0")
        jpgs = sorted(f for f in os.listdir(visual) if f.endswith(".jpg"))
        assert jpgs == ["0.jpg", "1.jpg"]
        assert not any(f.endswith(".mp4") for f in os.listdir(visual))

        rows = _read_jsonl(tmp_path)
        assert len(rows) == 2
        assert all(row["output"].endswith(".jpg") for row in rows)

    def test_max_samples_bounds_video_dump(self, tmp_path):
        outputs = torch.rand(3, 8, 3, 16, 16)
        _dump(tmp_path, outputs, max_samples=1)

        visual = os.path.join(str(tmp_path), "0")
        mp4s = [f for f in os.listdir(visual) if f.endswith(".mp4")]
        assert mp4s == ["0.mp4"]  # only the first sample encoded

        rows = _read_jsonl(tmp_path)
        assert len(rows) == 1
        assert rows[0]["reward"] == 0.0
