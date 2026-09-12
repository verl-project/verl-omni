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
from PIL import Image


def _load_module():
    path = Path(__file__).parents[2] / "examples/flowgrpo_trainer/ltx2/prepare_ti2va_data.py"
    spec = importlib.util.spec_from_file_location("ltx2_prepare_ti2va_data", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prepare_ti2va_data = _load_module()


def _write_split(input_dir: Path, record: dict) -> None:
    (input_dir / "train.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_convert_split_serializes_one_condition_image(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "fox.png"
    image_path.parent.mkdir()
    Image.new("RGB", (32, 24), color="orange").save(image_path)
    _write_split(tmp_path, {"prompt": "The fox turns toward the camera.", "image": "images/fox.png"})

    frame = prepare_ti2va_data._convert_split(tmp_path, "train", max_samples=-1)
    output = tmp_path / "train.parquet"
    frame.to_parquet(output, row_group_size=500)
    row = pq.read_table(output).to_pylist()[0]

    assert row["data_source"] == "ltx2_ti2va"
    assert row["ability"] == "text_image_to_audio_video"
    assert row["prompt"] == [{"role": "user", "content": "<image>The fox turns toward the camera."}]
    assert row["images"][0]["bytes"].startswith(b"\x89PNG")
    assert row["extra_info"]["source_image"] == "images/fox.png"


@pytest.mark.parametrize(
    "record",
    [
        {"prompt": "Move.", "images": []},
        {"prompt": "Move.", "images": ["a.png", "b.png"]},
    ],
)
def test_convert_split_rejects_non_single_image_rows(tmp_path: Path, record: dict) -> None:
    _write_split(tmp_path, record)

    with pytest.raises(ValueError, match="exactly one image"):
        prepare_ti2va_data._convert_split(tmp_path, "train", max_samples=-1)
