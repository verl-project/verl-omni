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


def _load_module():
    path = Path(__file__).parents[2] / "examples/gspo_trainer/data_process/nextqa.py"
    spec = importlib.util.spec_from_file_location("nextqa_data_process", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


nextqa = _load_module()


def _record(problem_id=1, video_filename="NextQA/NExTVideo/0001/clip.mp4", **overrides):
    record = {
        "problem_id": problem_id,
        "problem": (
            "What happens after the person sits down?\nOptions:\n"
            "A. They stand up.\nB. They wave.\nC. They read.\nD. They sleep.\nE. They leave."
        ),
        "data_type": "video",
        "problem_type": ["reasoning"],
        "solution": "<answer>C</answer>",
        "video_filename": video_filename,
    }
    record.update(overrides)
    return record


def _write_video(root: Path, relative_path: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"video")
    return path.resolve()


def test_build_rl_row_validates_options_solution_and_video(tmp_path):
    row, reason = nextqa.build_rl_row(_record(), tmp_path, split="train", index=0)
    assert row is None
    assert reason == "missing_video"

    _write_video(tmp_path, "NextQA/NExTVideo/0001/clip.mp4")
    row, reason = nextqa.build_rl_row(_record(solution="C"), tmp_path, split="train", index=0)
    assert row is None
    assert reason == "invalid_solution"

    row, reason = nextqa.build_rl_row(_record(problem="Question without choices"), tmp_path, "train", 0)
    assert row is None
    assert reason == "invalid_options"


def test_convert_dataset_writes_group_split_parquets(tmp_path):
    first_video = _write_video(tmp_path, "NextQA/NExTVideo/0001/first.mp4")
    second_video = _write_video(tmp_path, "NextQA/NExTVideo/0002/second.mp4")
    records = [
        _record(problem_id=1, video_filename="./NextQA/NExTVideo/0001/first.mp4"),
        _record(problem_id=2, video_filename="./NextQA/NExTVideo/0001/first.mp4", solution="<answer>A</answer>"),
        _record(problem_id=3, video_filename="./NextQA/NExTVideo/0002/second.mp4", solution="<answer>B</answer>"),
    ]
    (tmp_path / nextqa.DEFAULT_INPUT_FILE).write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )

    output_dir = tmp_path / "output"
    stats = nextqa.convert_dataset(tmp_path, output_dir, validation_ratio=0.5, seed=7)
    train_rows = pq.read_table(output_dir / "train.parquet").to_pylist()
    validation_rows = pq.read_table(output_dir / "validation.parquet").to_pylist()

    assert stats["input"] == 3
    assert stats["kept"] == 3
    assert stats["train"] + stats["validation"] == 3
    assert stats["dropped"] == {}

    train_videos = {row["videos"][0]["video"] for row in train_rows}
    validation_videos = {row["videos"][0]["video"] for row in validation_rows}
    assert train_videos.isdisjoint(validation_videos)
    assert train_videos | validation_videos == {str(first_video), str(second_video)}

    row = (train_rows + validation_rows)[0]
    assert row["data_source"] == nextqa.DATA_SOURCE
    assert row["ability"] == nextqa.ABILITY
    assert row["prompt"][0]["content"] == nextqa.SYSTEM_PROMPT
    assert row["prompt"][1]["content"].startswith("<video>What happens")
    assert row["videos"][0]["fps"] == 1.0
    assert row["videos"][0]["min_pixels"] == 32 * 28 * 28
    assert row["videos"][0]["max_pixels"] == 128 * 28 * 28
    assert row["videos"][0]["max_frames"] == 32
    assert row["reward_model"]["ground_truth"].startswith("<answer>")
    assert set(json.loads(row["extra_info"]["options"])) == set("ABCDE")

    for output_path in (output_dir / "train.parquet", output_dir / "validation.parquet"):
        parquet_file = pq.ParquetFile(output_path)
        for column_index in range(parquet_file.metadata.num_columns):
            encodings = parquet_file.metadata.row_group(0).column(column_index).encodings
            assert "RLE_DICTIONARY" not in encodings
            assert "PLAIN_DICTIONARY" not in encodings
