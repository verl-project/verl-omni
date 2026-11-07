# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""AudioMCQ conversion and scoring, including the existing prepared-parquet contract."""

import importlib.util
import json
import runpy
from pathlib import Path

import pandas as pd
import pytest

from tests.special_e2e.build_audiomcq_smoke_data import build

_CONVERTER = runpy.run_path(str(Path(__file__).parents[2] / "examples/gspo_trainer/data_process/audiomcq.py"))
build_rl_row, convert, split_rows = (_CONVERTER[name] for name in ("build_rl_row", "convert", "split_rows"))

# Reward functions can be loaded by path without initializing GPU engines.
_PATH = Path(__file__).parents[2] / "verl_omni/utils/reward_score/audio_mcq.py"
_SPEC = importlib.util.spec_from_file_location("audio_mcq_reward", _PATH)
reward = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reward)


@pytest.fixture
def record(tmp_path):
    (tmp_path / "clip.wav").write_bytes(b"fixture")
    return {"audio_path": "clip.wav", "question": "Which sound?", "choices": ["A bell", "A voice"], "answer": "A voice"}


def test_row_preserves_option_order_and_audio(record, tmp_path):
    row = build_rl_row(record, tmp_path, 7)
    assert row["prompt"][0]["content"].count("<audio>") == 1
    assert row["audios"] == [str(tmp_path / "clip.wav")]
    assert row["reward_model"]["ground_truth"]["correct_choice_index"] == 1
    assert row["reward_model"]["ground_truth"]["choices"] == record["choices"]


@pytest.mark.parametrize(
    "change",
    [
        {"audio_path": "../escape.wav"},
        {"audio_path": "missing.wav"},
        {"answer": "absent"},
        {"choices": ["A voice", "a voice"]},
        {"choices": ["A voice", "A voice."]},
        {"question": "<audio>twice"},
        {"choices": []},
    ],
)
def test_bad_inputs_fail_closed(record, tmp_path, change):
    with pytest.raises(ValueError):
        build_rl_row({**record, **change}, tmp_path, 0)


def test_split_groups_same_audio_and_is_deterministic(record, tmp_path):
    rows = []
    for i in range(4):
        path = tmp_path / f"{i}.wav"
        path.write_bytes(b"fixture")
        rows.append(build_rl_row({**record, "audio_path": path.name}, tmp_path, i))
    rows.append(rows[0])
    train, val = split_rows(rows, 2, 42)
    assert not {r["audios"][0] for r in train} & {r["audios"][0] for r in val}
    assert len(train) + len(val) == len(rows)
    assert (train, val) == split_rows(rows, 2, 42)


def test_conversion_reports_missing_assets_and_refuses_overwrite(record, tmp_path):
    (tmp_path / "other.wav").write_bytes(b"fixture")
    manifest = tmp_path / "data.jsonl"
    records = [record, {**record, "audio_path": "other.wav"}, {**record, "audio_path": "missing.wav"}]
    manifest.write_text("\n".join(json.dumps(r) for r in records))
    target = tmp_path / "prepared"
    report = convert(manifest, tmp_path, target, 1, 42)
    assert report["dropped"] == {"missing_audio": 1}
    assert len(pd.read_parquet(target / "train.parquet")) == 1
    before = (target / "train.parquet").read_bytes()
    with pytest.raises(FileExistsError):
        convert(manifest, tmp_path, target, 1, 42)
    assert (target / "train.parquet").read_bytes() == before


def test_conversion_requires_a_new_output_directory(tmp_path):
    target = tmp_path / "prepared"
    target.mkdir()
    with pytest.raises(FileExistsError):
        convert(tmp_path / "not_read.jsonl", tmp_path, target, 1, 42)
    assert not list(target.iterdir())


@pytest.mark.parametrize(
    "response,score,valid",
    [
        ("<answer>B</answer>", 1, 1),
        ("<answer>A voice</answer>", 1, 1),
        ("<answer> (B) </answer>", 1, 1),
        ("<answer>A</answer>", 0, 1),
        ("random model output", 0, 0),
        ("<answer></answer>", 0, 0),
        ("<answer>A</answer><answer>B</answer>", 0, 0),
        ("<answer>B</answer><answer>B</answer>", 1, 0),
        ("<answer>Ｂ</answer>", 1, 1),
        ("<answer>A bell or A voice</answer>", 0, 1),
    ],
)
def test_reward_matches_historical_contract(record, tmp_path, response, score, valid):
    target = build_rl_row(record, tmp_path, 0)["reward_model"]["ground_truth"]
    result = reward.compute_score(response, target, data_source="AudioMCQ", extra_info={})
    assert result == {"score": score, "content_correct": score, "format_valid": valid}


def test_reward_rejects_inconsistent_target(record, tmp_path):
    target = build_rl_row(record, tmp_path, 0)["reward_model"]["ground_truth"]
    target["correct_choice_indices"] = [0]
    with pytest.raises(ValueError):
        reward.compute_score("<answer>B</answer>", target)


def test_smoke_contains_decodable_nonempty_wav(tmp_path):
    import wave

    target = tmp_path / "smoke"
    build(target)
    splits = [pd.read_parquet(target / f"{name}.parquet") for name in ("train", "validation")]
    assert [len(split) for split in splits] == [16, 2]
    paths = [path for split in splits for audios in split.audios for path in audios]
    assert len(paths) == len(set(paths))
    for path in paths:
        with wave.open(path) as handle:
            assert (handle.getnchannels(), handle.getframerate()) == (1, 16000)
            assert handle.getnframes() > 0
            assert any(handle.readframes(handle.getnframes()))
