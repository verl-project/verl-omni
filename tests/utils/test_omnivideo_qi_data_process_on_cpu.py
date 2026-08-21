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

import pandas as pd


def _load_converter():
    path = Path(__file__).parents[2] / "examples/gspo_trainer/data_process/omnivideo_r1_qi.py"
    spec = importlib.util.spec_from_file_location("omnivideo_r1_qi_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(sample_id: str, video_name: str) -> dict:
    return {
        "id": sample_id,
        "Type": "0_30_s_nextqa_mc_qa_processed",
        "problem": "<video><audio>\nWhat happens?\nA. Nothing\nB. A door opens",
        "solution": "B. A door opens",
        "videos": [f"./data/videos/{video_name}"],
        "audios": [f"./data/audios/{video_name}.mp3"],
        "data_source": "0_30_s_nextqa",
    }


def test_build_row_can_use_the_video_audio_stream(tmp_path):
    converter = _load_converter()
    media_dir = tmp_path / "videos"
    media_dir.mkdir()
    video_path = media_dir / "sample.mp4"
    video_path.write_bytes(b"video")

    row, reason = converter.build_rl_row(
        _record("sample", "sample.mp4"),
        0,
        [("./data/videos", media_dir)],
        audio_from_video=True,
        fps=2.0,
        max_frames=64,
    )

    assert reason is None
    assert row is not None
    assert "audios" not in row
    assert row["videos"] == [{"video": str(video_path), "fps": 2.0, "min_frames": 4, "max_frames": 64}]
    assert row["prompt"][1]["content"].count("<video>") == 1
    assert "<audio>" not in row["prompt"][1]["content"]
    assert row["extra_info"]["video_path"] == str(video_path)


def test_build_row_adds_missing_videovista_media_placeholder(tmp_path):
    converter = _load_converter()
    media_dir = tmp_path / "videos"
    media_dir.mkdir()
    video_path = media_dir / "sample.mp4"
    video_path.write_bytes(b"video")
    record = _record("videovista", "sample.mp4")
    record["problem"] = "Describe the displayed video in detail."

    row, reason = converter.build_rl_row(
        record,
        0,
        [("./data/videos", media_dir)],
        audio_from_video=True,
        fps=2.0,
        max_frames=64,
    )

    assert reason is None
    assert row["prompt"][1]["content"].startswith("<video>\nDescribe the displayed video")


def test_convert_jsonl_splits_distinct_videos(tmp_path):
    converter = _load_converter()
    media_dir = tmp_path / "videos"
    media_dir.mkdir()
    for name in ("one.mp4", "two.mp4", "three.mp4"):
        (media_dir / name).write_bytes(b"video")

    input_path = tmp_path / "merged_train_all_qi.jsonl"
    records = [
        _record("one-a", "one.mp4"),
        _record("one-b", "one.mp4"),
        _record("two", "two.mp4"),
        _record("three", "three.mp4"),
    ]
    input_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    stats = converter.convert_jsonl(
        input_path,
        tmp_path / "output",
        [("./data/videos", media_dir)],
        val_size=1,
    )

    train = pd.read_parquet(stats["train"])
    validation = pd.read_parquet(stats["validation"])
    train_videos = {item["video"] for videos in train["videos"] for item in videos}
    validation_videos = {item["video"] for videos in validation["videos"] for item in videos}
    assert len(train) + len(validation) == 4
    assert train_videos.isdisjoint(validation_videos)
    assert list(train.columns) == ["data_source", "prompt", "videos", "ability", "reward_model", "extra_info"]
