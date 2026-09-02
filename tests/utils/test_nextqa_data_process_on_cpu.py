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
from types import SimpleNamespace

import pandas as pd
import pyarrow.parquet as pq
import pytest


def _load_module():
    path = Path(__file__).parents[2] / "examples/gspo_trainer/data_process/nextqa.py"
    spec = importlib.util.spec_from_file_location("nextqa_data_process", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


nextqa = _load_module()


def _record(video=3238737531, answer=3, qid=2, **overrides):
    record = {
        "video": video,
        "frame_count": 100,
        "width": 640,
        "height": 360,
        "question": "how many children are in the video",
        "answer": answer,
        "qid": qid,
        "type": "DC",
        "a0": "one",
        "a1": "three",
        "a2": "seven",
        "a3": "two",
        "a4": "five",
    }
    record.update(overrides)
    return record


def _write_dataset(root: Path, train_records, val_records, mapping) -> Path:
    repo_dir = root / "repo"
    repo_dir.mkdir(parents=True)
    pd.DataFrame(train_records).to_csv(repo_dir / "train.csv", index=False)
    pd.DataFrame(val_records).to_csv(repo_dir / "val.csv", index=False)
    (repo_dir / "map_vid_vidorID.json").write_text(json.dumps(mapping), encoding="utf-8")
    (root / "NExTVideo").mkdir()
    return root


def _write_video(root: Path, relative_path: str) -> Path:
    path = root / "NExTVideo" / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"video")
    return path.resolve()


def test_convert_dataset_preserves_official_splits_and_answer_letters(tmp_path, monkeypatch):
    train_records = [_record(video=1000 + answer, answer=answer, qid=answer) for answer in range(5)]
    val_records = [_record(video=2000, answer=4, qid=9, question="what happens next")]
    mapping = {str(record["video"]): f"{record['video']}/clip" for record in train_records + val_records}
    _write_dataset(tmp_path, train_records, val_records, mapping)
    for relative_path in mapping.values():
        _write_video(tmp_path, f"{relative_path}.mp4")
    monkeypatch.setattr(nextqa, "probe_audio_stream", lambda _video_path: None)

    output_dir = tmp_path / "output"
    stats = nextqa.convert_dataset(tmp_path, output_dir)
    train_rows = pq.read_table(output_dir / "train.parquet").to_pylist()
    validation_rows = pq.read_table(output_dir / "validation.parquet").to_pylist()

    assert stats["train"]["input"] == 5
    assert stats["train"]["kept"] == 5
    assert stats["train"]["audio"] == {
        "checked_videos": 5,
        "with_audio": 5,
        "missing_audio_stream": 0,
        "invalid_media": 0,
    }
    assert stats["train"]["answers"] == {letter: 1 for letter in "ABCDE"}
    assert stats["validation"]["input"] == 1
    assert stats["validation"]["kept"] == 1
    assert stats["validation"]["audio"] == {
        "checked_videos": 1,
        "with_audio": 1,
        "missing_audio_stream": 0,
        "invalid_media": 0,
    }
    assert len(train_rows) == 5
    assert len(validation_rows) == 1
    assert {row["extra_info"]["split"] for row in train_rows} == {"train"}
    assert {row["extra_info"]["split"] for row in validation_rows} == {"validation"}
    assert [row["reward_model"]["ground_truth"] for row in train_rows] == [
        f"<answer>{letter}</answer>" for letter in "ABCDE"
    ]

    row = train_rows[3]
    assert row["data_source"] == nextqa.DATA_SOURCE
    assert row["ability"] == nextqa.ABILITY
    assert row["prompt"][0]["content"] == nextqa.SYSTEM_PROMPT
    assert row["prompt"][1]["content"] == (
        "<video>how many children are in the video\nA. one\nB. three\nC. seven\nD. two\nE. five"
    )
    assert row["videos"][0]["video"] == str((tmp_path / "NExTVideo/1003/clip.mp4").resolve())
    assert Path(row["videos"][0]["video"]).is_file()
    assert row["videos"][0]["fps"] == 1.0
    assert row["videos"][0]["min_pixels"] == 32 * 28 * 28
    assert row["videos"][0]["max_pixels"] == 128 * 28 * 28
    assert row["videos"][0]["max_frames"] == 32
    assert row["extra_info"]["problem_id"] == "1003_3"
    assert row["extra_info"]["answer_index"] == 3
    assert row["extra_info"]["answer_letter"] == "D"
    assert json.loads(row["extra_info"]["options"]) == {
        "A": "one",
        "B": "three",
        "C": "seven",
        "D": "two",
        "E": "five",
    }

    for output_path in (output_dir / "train.parquet", output_dir / "validation.parquet"):
        parquet_file = pq.ParquetFile(output_path)
        for column_index in range(parquet_file.metadata.num_columns):
            encodings = parquet_file.metadata.row_group(0).column(column_index).encodings
            assert "RLE_DICTIONARY" not in encodings
            assert "PLAIN_DICTIONARY" not in encodings


def test_convert_split_counts_invalid_official_records(tmp_path, monkeypatch):
    valid = _record(video=1, qid=1)
    records = [
        valid,
        _record(video=2, qid=2),
        _record(video=3, qid=3),
        _record(video=1, qid=4, answer="not-an-index"),
        _record(video=1, qid=5, answer=5),
        _record(video=1, qid=6, question=None),
        _record(video=1, qid=7, a2=float("nan")),
    ]
    _write_dataset(tmp_path, records, [valid], {"1": "0001/video", "3": "0003/missing.mp4"})
    _write_video(tmp_path, "0001/video.mp4")
    monkeypatch.setattr(nextqa, "probe_audio_stream", lambda _video_path: None)

    output_path = tmp_path / "output/train.parquet"
    stats = nextqa.convert_split(
        tmp_path / "repo/train.csv",
        output_path,
        nextqa.load_video_mapping(tmp_path / "repo/map_vid_vidorID.json"),
        tmp_path / "NExTVideo",
        "train",
    )

    assert stats == {
        "input": 7,
        "kept": 1,
        "dropped": {
            "empty_option": 1,
            "empty_question": 1,
            "invalid_answer": 2,
            "missing_video": 1,
            "missing_video_mapping": 1,
        },
        "audio": {
            "checked_videos": 1,
            "with_audio": 1,
            "missing_audio_stream": 0,
            "invalid_media": 0,
        },
        "answers": {"D": 1},
        "output": str(output_path.resolve()),
    }


def test_convert_split_filters_audio_failures_and_caches_unique_videos(tmp_path, monkeypatch):
    records = [
        _record(video=1, qid=1),
        _record(video=1, qid=2),
        _record(video=2, qid=3),
        _record(video=2, qid=4),
        _record(video=3, qid=5),
    ]
    mapping = {str(video_id): f"{video_id:04d}/video" for video_id in (1, 2, 3)}
    _write_dataset(tmp_path, records, [records[0]], mapping)
    video_paths = {
        video_id: _write_video(tmp_path, f"{relative_path}.mp4")
        for video_id, relative_path in ((int(video_id), path) for video_id, path in mapping.items())
    }
    statuses = {
        str(video_paths[1]): None,
        str(video_paths[2]): "missing_audio_stream",
        str(video_paths[3]): "invalid_media",
    }
    probe_calls = []

    def fake_probe(video_path):
        probe_calls.append(video_path)
        return statuses[video_path]

    monkeypatch.setattr(nextqa, "probe_audio_stream", fake_probe)
    output_path = tmp_path / "output/train.parquet"
    stats = nextqa.convert_split(
        tmp_path / "repo/train.csv",
        output_path,
        nextqa.load_video_mapping(tmp_path / "repo/map_vid_vidorID.json"),
        tmp_path / "NExTVideo",
        "train",
    )

    assert probe_calls == [str(video_paths[video_id]) for video_id in (1, 2, 3)]
    assert stats == {
        "input": 5,
        "kept": 2,
        "dropped": {"invalid_media": 1, "missing_audio_stream": 2},
        "audio": {
            "checked_videos": 3,
            "with_audio": 1,
            "missing_audio_stream": 1,
            "invalid_media": 1,
        },
        "answers": {"D": 2},
        "output": str(output_path.resolve()),
    }
    assert {row["extra_info"]["video_id"] for row in pq.read_table(output_path).to_pylist()} == {"1"}


def test_probe_audio_stream_checks_metadata_and_decodes_one_audio_frame(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        stdout = "0\n" if command[0] == "ffprobe" else ""
        return SimpleNamespace(returncode=0, stdout=stdout)

    monkeypatch.setattr(nextqa.subprocess, "run", fake_run)

    assert nextqa.probe_audio_stream("video.mp4") is None
    assert [command[0] for command, _kwargs in calls] == ["ffprobe", "ffmpeg"]
    assert all(kwargs["timeout"] == nextqa.MEDIA_PROBE_TIMEOUT_SECONDS for _command, kwargs in calls)
    assert calls[1][0][-6:] == ["0:a:0", "-frames:a", "1", "-f", "null", "-"]


def test_probe_audio_stream_does_not_decode_when_audio_stream_is_missing(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(nextqa.subprocess, "run", fake_run)

    assert nextqa.probe_audio_stream("video.mp4") == "missing_audio_stream"
    assert [command[0] for command in calls] == ["ffprobe"]


@pytest.mark.parametrize("failure_site", ["ffprobe", "ffmpeg"])
def test_probe_audio_stream_maps_timeout_to_invalid_media(monkeypatch, failure_site):
    def fake_run(command, **kwargs):
        if command[0] == failure_site:
            raise nextqa.subprocess.TimeoutExpired(command, kwargs["timeout"])
        return SimpleNamespace(returncode=0, stdout="0\n")

    monkeypatch.setattr(nextqa.subprocess, "run", fake_run)

    assert nextqa.probe_audio_stream("video.mp4") == "invalid_media"


@pytest.mark.parametrize("failure_site", ["ffprobe", "ffmpeg"])
def test_probe_audio_stream_maps_process_failure_to_invalid_media(monkeypatch, failure_site):
    def fake_run(command, **kwargs):
        return SimpleNamespace(
            returncode=1 if command[0] == failure_site else 0,
            stdout="0\n",
        )

    monkeypatch.setattr(nextqa.subprocess, "run", fake_run)

    assert nextqa.probe_audio_stream("video.mp4") == "invalid_media"


@pytest.mark.parametrize("missing_binary", ["ffprobe", "ffmpeg"])
def test_probe_audio_stream_reports_missing_binary(monkeypatch, missing_binary):
    def fake_run(command, **kwargs):
        if command[0] == missing_binary:
            raise FileNotFoundError(command[0])
        return SimpleNamespace(returncode=0, stdout="0\n")

    monkeypatch.setattr(nextqa.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=rf"{missing_binary} is required"):
        nextqa.probe_audio_stream("video.mp4")


def test_resolve_video_path_handles_extension_and_rejects_escape(tmp_path):
    video_root = tmp_path / "NExTVideo"
    video_root.mkdir()
    video = _write_video(tmp_path, "0083/5572343997.mp4")

    assert nextqa.resolve_video_path(video_root, "0083/5572343997") == video
    assert nextqa.resolve_video_path(video_root, "0083/5572343997.mp4") == video
    assert nextqa.resolve_video_path(video_root, "../outside") is None


def test_convert_split_reports_missing_csv_columns(tmp_path):
    source_path = tmp_path / "train.csv"
    pd.DataFrame([_record()]).drop(columns=["a4", "qid"]).to_csv(source_path, index=False)

    with pytest.raises(ValueError, match=r"Missing required columns.*a4, qid"):
        nextqa.convert_split(source_path, tmp_path / "train.parquet", {}, tmp_path, "train")


def test_convert_dataset_rejects_nested_video_directory(tmp_path):
    _write_dataset(tmp_path, [_record()], [_record()], {"3238737531": "0001/video"})
    (tmp_path / "NExTVideo/NExTVideo").mkdir()

    with pytest.raises(ValueError, match=r"Invalid nested NExT-QA video directory.*unzip NExTVideo\.zip"):
        nextqa.convert_dataset(tmp_path, tmp_path / "output")


def test_convert_dataset_reports_missing_standard_video_directory(tmp_path):
    _write_dataset(tmp_path, [_record()], [_record()], {"3238737531": "0001/video"})
    (tmp_path / "NExTVideo").rmdir()

    with pytest.raises(FileNotFoundError, match=r"Required NExT-QA video directory.*do not use `-d NExTVideo`"):
        nextqa.convert_dataset(tmp_path, tmp_path / "output")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"fps": 0}, "fps must be greater than zero"),
        ({"min_pixels": 0}, "pixel limits must satisfy"),
        ({"min_pixels": 10, "max_pixels": 9}, "pixel limits must satisfy"),
        ({"max_frames": 1}, "max_frames must be at least 2"),
    ],
)
def test_convert_dataset_validates_sampling_parameters(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        nextqa.convert_dataset(tmp_path, tmp_path / "output", **overrides)
