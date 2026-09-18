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
from pathlib import Path

import pyarrow.parquet as pq
import pytest


def _load_module():
    path = Path(__file__).parents[2] / "examples/diffusionnft_trainer/minimax_h3/prepare_ref2va_data.py"
    spec = importlib.util.spec_from_file_location("minimax_h3_prepare_ref2va_data", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare_ref2va_data = _load_module()


def _write_split(input_dir: Path, record: dict) -> None:
    (input_dir / "train.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_convert_split_serializes_mixed_references(tmp_path):
    for relative_path, payload in (
        ("images/character.png", b"image"),
        ("videos/motion.mp4", b"video-1"),
        ("videos/detail.mp4", b"video-2"),
        ("audios/voice.wav", b"audio"),
    ):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    _write_split(
        tmp_path,
        {
            "prompt": "Keep the same character and voice.",
            "image": "images/character.png",
            "videos": ["videos/motion.mp4", {"path": "videos/detail.mp4", "start_time_seconds": 1.5}],
            "audio": "audios/voice.wav",
        },
    )

    frame = prepare_ref2va_data._convert_split(tmp_path, "train", max_samples=-1)
    output = tmp_path / "train.parquet"
    frame.to_parquet(output, row_group_size=500)
    row = pq.read_table(output).to_pylist()[0]

    image_path = str((tmp_path / "images/character.png").resolve())
    video_paths = [
        str((tmp_path / "videos/motion.mp4").resolve()),
        str((tmp_path / "videos/detail.mp4").resolve()),
    ]
    audio_path = str((tmp_path / "audios/voice.wav").resolve())
    assert row["data_source"] == "minimax_h3_ref2va"
    assert row["ability"] == "reference_to_audio_video"
    assert row["prompt"] == [
        {"role": "user", "content": "<image><video><video><audio>Keep the same character and voice."}
    ]
    assert row["images"] == [{"bytes": b"image"}]
    assert row["videos"] == [
        {"start_time_seconds": None, "video": video_paths[0]},
        {"start_time_seconds": 1.5, "video": video_paths[1]},
    ]
    assert row["audios"] == [audio_path]
    assert row["extra_info"]["source_images"] == [image_path]
    assert row["extra_info"]["source_videos"] == video_paths
    assert row["extra_info"]["source_audios"] == [audio_path]


@pytest.mark.parametrize(
    ("references", "message"),
    [
        ({"images": ["missing.png"] * 10}, "exceeds reference limits: images=10"),
        ({"videos": ["missing.mp4"] * 4}, "exceeds reference limits: images=0, videos=4"),
        (
            {"image": "missing.png", "audios": ["missing.wav"] * 4},
            "exceeds reference limits: images=1, videos=0, audios=4",
        ),
        (
            {
                "images": ["missing.png"] * 9,
                "videos": ["missing.mp4"] * 3,
                "audio": "missing.wav",
            },
            "exceeds the 12-file reference limit",
        ),
    ],
)
def test_convert_split_rejects_reference_limits(tmp_path, references, message):
    _write_split(tmp_path, {"prompt": "A valid prompt.", **references})

    with pytest.raises(ValueError, match=message):
        prepare_ref2va_data._convert_split(tmp_path, "train", max_samples=-1)
