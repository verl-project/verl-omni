# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for media leakage in the AVQA train/validation split."""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_module():
    path = Path(__file__).parents[2] / "examples/gspo_trainer/data_process/avqa_strict.py"
    spec = importlib.util.spec_from_file_location("avqa_strict", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


strict = _load_module()


def _row(tmp_path, name, image_bytes, audio_bytes):
    image, audio = tmp_path / f"{name}.jpg", tmp_path / f"{name}.wav"
    image.write_bytes(image_bytes)
    audio.write_bytes(audio_bytes)
    return {"sample": name, "images": [{"image": str(image)}], "audios": [str(audio)]}


def test_excludes_either_media_overlap_without_changing_original_splits(tmp_path):
    validation = [_row(tmp_path, "valid", b"shared-image", b"shared-audio")]
    train = [
        _row(tmp_path, "image-overlap", b"shared-image", b"other-audio"),
        _row(tmp_path, "audio-overlap", b"other-image", b"shared-audio"),
        _row(tmp_path, "keep", b"unique-image", b"unique-audio"),
    ]
    train_file, validation_file, output_file = (
        tmp_path / name for name in ("train.parquet", "validation.parquet", "train_strict.parquet")
    )
    pd.DataFrame(train).to_parquet(train_file, index=False, use_dictionary=False)
    pd.DataFrame(validation).to_parquet(validation_file, index=False, use_dictionary=False)
    train_before, validation_before = train_file.read_bytes(), validation_file.read_bytes()

    audit = strict.make_strict_train(train_file, validation_file, output_file)

    assert audit["source_train"] == 3 and audit["validation_unchanged"] == 1
    assert audit["strict_train"] == 1 and audit["excluded_count"] == 2
    assert audit["excluded"] == [
        {"row": 0, "shared_image": True, "shared_audio": False},
        {"row": 1, "shared_image": False, "shared_audio": True},
    ]
    assert pd.read_parquet(output_file)["sample"].tolist() == ["keep"]
    assert train_file.read_bytes() == train_before and validation_file.read_bytes() == validation_before


def test_refuses_to_overwrite_an_original_split(tmp_path):
    with pytest.raises(ValueError, match="must not overwrite"):
        strict.make_strict_train(
            tmp_path / "train.parquet", tmp_path / "validation.parquet", tmp_path / "train.parquet"
        )


def test_refuses_same_train_and_validation(tmp_path):
    with pytest.raises(ValueError, match="must be different"):
        strict.make_strict_train(tmp_path / "same.parquet", tmp_path / "same.parquet", tmp_path / "out.parquet")


@pytest.mark.parametrize("protected_name", ["train.parquet", "validation.parquet", "train_strict.parquet"])
def test_refuses_audit_path_that_would_overwrite_a_split(tmp_path, monkeypatch, protected_name):
    paths = [tmp_path / name for name in ("train.parquet", "validation.parquet", "train_strict.parquet")]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "avqa_strict.py",
            "--train_file",
            str(paths[0]),
            "--validation_file",
            str(paths[1]),
            "--output_file",
            str(paths[2]),
            "--audit_file",
            str(tmp_path / protected_name),
        ],
    )
    with pytest.raises(ValueError, match="Audit output must not overwrite"):
        strict.main()
