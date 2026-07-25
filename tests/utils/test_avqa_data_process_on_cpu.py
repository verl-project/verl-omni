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


def _load_module():
    path = Path(__file__).parents[2] / "examples/gspo_trainer/data_process/avqa.py"
    spec = importlib.util.spec_from_file_location("avqa_data_process", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


avqa = _load_module()


def _record(**overrides):
    record = {
        "problem_id": 7,
        "problem": "Which event is visible and audible?",
        "data_type": "image_audio",
        "problem_type": "multiple choice",
        "options": ["A. rain", "B. applause", "C. traffic", "D. birds"],
        "solution": "<answer>B</answer>",
        "path": {"image": "images/sample.jpg", "audio": "audios/sample.wav"},
        "data_source": "OmniInstruct_v1-AVQA",
    }
    record.update(overrides)
    return record


def test_build_rl_row_contains_image_audio_placeholders_and_media(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "audios").mkdir()
    (tmp_path / "images/sample.jpg").write_bytes(b"image")
    (tmp_path / "audios/sample.wav").write_bytes(b"audio")

    row, reason = avqa.build_rl_row(_record(), tmp_path, split="train", index=0)

    assert reason is None
    assert row["prompt"][0]["content"] == avqa.SYSTEM_PROMPT
    assert row["prompt"][1]["content"].startswith("<image><audio>Which event is visible and audible?\nOptions:\n")
    assert row["images"] == [{"image": str((tmp_path / "images/sample.jpg").resolve())}]
    assert row["audios"] == [str((tmp_path / "audios/sample.wav").resolve())]
    assert row["reward_model"]["ground_truth"] == "<answer>B</answer>"
    assert json.loads(row["extra_info"]["options"])["B"] == "applause"


def test_build_rl_row_rejects_missing_media_and_bad_solution(tmp_path):
    row, reason = avqa.build_rl_row(_record(), tmp_path, split="train", index=0)
    assert row is None
    assert reason == "missing_image"

    (tmp_path / "images").mkdir()
    (tmp_path / "audios").mkdir()
    (tmp_path / "images/sample.jpg").write_bytes(b"image")
    (tmp_path / "audios/sample.wav").write_bytes(b"audio")
    row, reason = avqa.build_rl_row(_record(solution="<answer>Z</answer>"), tmp_path, "train", 0)
    assert row is None
    assert reason == "invalid_solution"


def test_convert_split_writes_verl_parquet(tmp_path):
    split_dir = tmp_path / "train"
    (split_dir / "images").mkdir(parents=True)
    (split_dir / "audios").mkdir()
    (split_dir / "images/sample.jpg").write_bytes(b"image")
    (split_dir / "audios/sample.wav").write_bytes(b"audio")
    source = split_dir / "omni_rl_format_train.json"
    source.write_text(json.dumps([_record()]), encoding="utf-8")

    output = tmp_path / "out/train.parquet"
    stats = avqa.convert_split(source, output, split="train")
    frame = pd.read_parquet(output)

    assert stats["kept"] == 1
    assert stats["dropped"] == {}
    assert frame.loc[0, "data_source"] == avqa.DATA_SOURCE
    assert frame.loc[0, "ability"] == avqa.ABILITY
