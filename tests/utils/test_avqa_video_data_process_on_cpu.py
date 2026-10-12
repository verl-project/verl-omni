# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Original AVQA annotation schema and video identity conversion contracts."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

path = Path(__file__).parents[2] / "examples/gspo_trainer/data_process/avqa_video.py"
spec = importlib.util.spec_from_file_location("avqa_video_data_process", path)
avqa_video = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(avqa_video)


def record(**changes):
    """Match the schema of an original AVQA validation annotation."""
    row = {
        "id": 1139,
        "video_name": "06GyG4-wONQ_000006",
        "video_id": 1395,
        "question_text": "How many animals are there in the video?",
        "multi_choice": ["3", "One", "4", "2"],
        "answer": 3,
        "question_relation": "Both",
        "question_type": "Which",
    }
    return {**row, **changes}


def test_original_avqa_parquet_preserves_video_identity_and_split(tmp_path):
    (tmp_path / "06GyG4-wONQ_000006.mp4").touch()
    source = tmp_path / "val_qa.json"
    source.write_text(json.dumps([record(), record(id=1140, video_name="missing")]))
    target = tmp_path / "validation.parquet"
    summary = avqa_video.convert_split(source, tmp_path, target, "validation")
    assert summary == {"input": 2, "kept": 1, "dropped": {"missing_video": 1}}
    [row] = pd.read_parquet(target).to_dict("records")
    assert row["data_source"] == "avqa_video"
    assert row["reward_model"]["ground_truth"] == "<answer>D</answer>"
    assert row["extra_info"]["problem_id"] == "1139" and row["extra_info"]["split"] == "validation"
    assert row["videos"][0]["video"] == str((tmp_path / "06GyG4-wONQ_000006.mp4").resolve())
    assert row["videos"][0]["fps"] == 1 and row["videos"][0]["max_frames"] == 32
    assert "<video>" in row["prompt"][1]["content"] and "images" not in row and "audios" not in row
    assert json.loads(row["extra_info"]["options"])["D"] == "2"


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"answer": 4}, "invalid_answer"),
        ({"answer": 1.5}, "invalid_answer"),
        ({"answer": True}, "invalid_answer"),
        ({"video_name": "../escape"}, "invalid_video_name"),
        ({"multi_choice": ["A", "A", "C", "D"]}, "duplicate_options"),
        ({"multi_choice": ["A", "B", "C"]}, "invalid_options"),
        ({"question_text": " "}, "empty_question"),
        ({"id": None}, "invalid_question_id"),
    ],
)
def test_invalid_annotation_is_not_silently_coerced(tmp_path, changes, reason):
    (tmp_path / "06GyG4-wONQ_000006.mp4").touch()
    row, actual = avqa_video.build_rl_row(record(**changes), tmp_path, "train", 0)
    assert row is None and actual == reason


def test_symlink_escape_and_duplicate_question_id_are_rejected(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.touch()
    (videos / "06GyG4-wONQ_000006.mp4").symlink_to(outside)
    assert avqa_video.build_rl_row(record(), videos, "train", 0)[1] == "missing_video"
    source = tmp_path / "train_qa.json"
    source.write_text(json.dumps([record(), record()]))
    (tmp_path / "06GyG4-wONQ_000006.mp4").touch()
    with pytest.raises(ValueError, match="Duplicate question id"):
        avqa_video.convert_split(source, tmp_path, tmp_path / "train.parquet", "train")


def test_image_audio_r1_record_is_not_accepted_as_video(tmp_path):
    row = {
        "problem_id": 0,
        "problem": "question",
        "data_type": "image_audio",
        "path": {"image": "image.jpg", "audio": "audio.wav"},
        "solution": "<answer>A</answer>",
    }
    assert avqa_video.build_rl_row(row, tmp_path, "train", 0)[0] is None


def test_output_overwrite_refused_before_reading_source(tmp_path, monkeypatch):
    (tmp_path / "train.parquet").write_bytes(b"preserve existing data")
    monkeypatch.setattr(
        "sys.argv",
        [
            "avqa_video.py",
            "--train_json",
            "/missing/train.json",
            "--validation_json",
            "/missing/val.json",
            "--video_root",
            str(tmp_path),
            "--output_dir",
            str(tmp_path),
        ],
    )
    with pytest.raises(SystemExit):
        avqa_video.main()
    assert (tmp_path / "train.parquet").read_bytes() == b"preserve existing data"


def test_missing_media_cannot_hide_duplicate_question_id(tmp_path):
    source = tmp_path / "train_qa.json"
    source.write_text(json.dumps([record(video_name="missing"), record()]))
    (tmp_path / "06GyG4-wONQ_000006.mp4").touch()
    with pytest.raises(ValueError, match="Duplicate question id"):
        avqa_video.convert_split(source, tmp_path, tmp_path / "train.parquet", "train")
    assert not (tmp_path / "train.parquet").exists()


def test_source_audit_does_not_require_media_or_change_annotation_bytes(tmp_path):
    train, validation = tmp_path / "train.json", tmp_path / "validation.json"
    train.write_text(json.dumps([record(video_name="not_mounted")]))
    validation.write_text(json.dumps([record(id=1140, video_name="not_mounted")]))
    original = {path: path.read_bytes() for path in (train, validation)}
    audit = avqa_video.audit_annotation_splits(train, validation)
    assert audit["source_split_overlap"] == {"question_ids": 0, "video_names": 1}
    for split, path in (("train", train), ("validation", validation)):
        assert path.read_bytes() == original[path]
        assert audit["annotation_sources"][split] == {
            "annotation_sha256": hashlib.sha256(original[path]).hexdigest(),
            "questions": 1,
            "unique_question_ids": 1,
            "unique_video_names": 1,
        }


def test_official_clip_overlap_is_reported_without_repartitioning(tmp_path, monkeypatch):
    train, validation = tmp_path / "train.json", tmp_path / "validation.json"
    train.write_text(json.dumps([record(), record(id=1140, multi_choice=["A", " A ", "C", "D"])]))
    validation.write_text(json.dumps([record(id=1141)]))
    (tmp_path / "06GyG4-wONQ_000006.mp4").touch()
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "avqa_video.py",
            "--train_json",
            str(train),
            "--validation_json",
            str(validation),
            "--video_root",
            str(tmp_path),
            "--output_dir",
            str(output),
        ],
    )
    avqa_video.main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["source_split_overlap"] == {"question_ids": 0, "video_names": 1}
    assert (
        manifest["annotation_sources"]["train"]["annotation_sha256"] == hashlib.sha256(train.read_bytes()).hexdigest()
    )
    assert manifest["train"] == {"input": 2, "kept": 1, "dropped": {"duplicate_options": 1}}
    assert manifest["validation"]["kept"] == 1
    for split, identity in (("train", "1139"), ("validation", "1141")):
        [row] = pd.read_parquet(output / f"{split}.parquet").to_dict("records")
        assert row["extra_info"]["problem_id"] == identity and row["extra_info"]["split"] == split


def test_cross_split_question_overlap_is_rejected_before_any_output(tmp_path, monkeypatch):
    source = tmp_path / "annotations.json"
    source.write_text(json.dumps([record(video_name="missing")]))
    output = tmp_path / "output"
    monkeypatch.setattr(
        "sys.argv",
        [
            "avqa_video.py",
            "--train_json",
            str(source),
            "--validation_json",
            str(source),
            "--video_root",
            str(tmp_path),
            "--output_dir",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="question ids overlap"):
        avqa_video.main()
    assert not output.exists()
