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

"""UniRL 原始划分、指令、ID、参考图和现有编辑 reward 的数据合约。"""

import copy
import importlib.util
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "prepare_qwen21_edit", ROOT / "examples/diffusionnft_trainer/qwen_image_21/prepare_edit_data.py"
)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def sample(tmp_path, identifier="example", count=1, prompt="Add a blue hat."):
    media = []
    for index in range(count):
        path = tmp_path / f"input_{index}.png"
        if not path.exists():
            Image.new("RGBA", (3, 2), (index * 40, 0, 0, 128)).save(path)
        media.append({"modality": "image", "role": "condition", "uri": path.name})
    return {
        "prompt": prompt,
        "media": media,
        "metadata": {"global_id": identifier, "source": "published_source", "edit_reward_score": 1.5},
    }


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_actor_and_judge_share_ordered_original_bytes(tmp_path, count):
    source = sample(tmp_path, count=count)
    original = copy.deepcopy(source)
    row = prepare.convert_record(source, tmp_path, "train", 7)
    assert source == original
    assert row["prompt"][0]["content"].count("<image>") == count
    assert row["reward_model"] == {"style": "model", "ground_truth": source["prompt"]}
    for media, actor, judge in zip(source["media"], row["images"], row["extra_info"]["reference_images"], strict=True):
        assert actor["bytes"] == judge["bytes"] == (tmp_path / media["uri"]).read_bytes()
        assert actor is not judge
        with Image.open(BytesIO(actor["bytes"])) as image:
            assert image.size == (3, 2) and image.mode == "RGBA"
    assert row["extra_info"]["index"] == "example"
    assert row["extra_info"]["source_index"] == 7
    assert row["extra_info"]["split"] == "train"
    assert json.loads(row["extra_info"]["source_record"]) == source
    assert row["negative_prompt"] == [{"role": "user", "content": " "}]


def test_existing_image_marker_preserves_original_instruction(tmp_path):
    instruction = "<image>\nAdd a blue hat.  "
    source = sample(tmp_path, prompt=instruction)
    row = prepare.convert_record(source, tmp_path, "test", 0)
    assert row["prompt"][0]["content"] == instruction
    assert row["reward_model"]["ground_truth"] == instruction
    assert json.loads(row["extra_info"]["source_record"])["prompt"] == instruction
    source["media"] *= 2
    with pytest.raises(ValueError, match="markers"):
        prepare.convert_record(source, tmp_path, "test", 0)


@pytest.mark.parametrize("count", [0, 5])
def test_invalid_reference_count_is_explicit(tmp_path, count):
    with pytest.raises(ValueError, match="1–4"):
        prepare.convert_record(sample(tmp_path, count=count), tmp_path, "train", 0)


@pytest.mark.parametrize("field,value", [("role", "target"), ("modality", "video"), ("uri", "../outside.png")])
def test_invalid_condition_media_is_not_silently_converted(tmp_path, field, value):
    source = sample(tmp_path)
    source["media"][0][field] = value
    with pytest.raises(ValueError):
        prepare.convert_record(source, tmp_path, "train", 0)


def test_missing_source_image_is_not_skipped(tmp_path):
    source = sample(tmp_path)
    (tmp_path / source["media"][0]["uri"]).unlink()
    with pytest.raises(FileNotFoundError):
        prepare.convert_record(source, tmp_path, "train", 0)


def test_conversion_preserves_splits_order_and_repeated_images(tmp_path):
    from datasets import Dataset

    train = [sample(tmp_path, identifier=f"train:{index}", prompt=f"Edit {index}.") for index in range(35)]
    # 发布者可能在不同划分复用同一原图；保持这些记录，不自行去重。
    test = [sample(tmp_path, identifier="test:0", prompt="<image>\nRemove the hat.")]
    test.append(copy.deepcopy(test[0]))
    for split, records in [("train", train), ("test", test)]:
        (tmp_path / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8"
        )
    output = tmp_path / "converted"
    prepare.main(SimpleNamespace(input_dir=tmp_path, output_dir=output))
    for split, expected in [("train", train), ("test", test)]:
        rows = Dataset.from_parquet(str(output / f"{split}.parquet")).to_list()
        assert len(rows) == len(expected)
        for index, (row, source) in enumerate(zip(rows, expected, strict=True)):
            assert row["extra_info"]["index"] == source["metadata"]["global_id"]
            assert row["extra_info"]["source_index"] == index
            assert row["extra_info"]["split"] == split
            assert json.loads(row["extra_info"]["source_record"]) == source
            assert row["reward_model"] == {"style": "model", "ground_truth": source["prompt"]}
            assert row["images"] == row["extra_info"]["reference_images"]
