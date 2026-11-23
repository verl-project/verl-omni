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

"""验证 OCR 数据转换不替换样本、标签或数据划分。"""

from types import SimpleNamespace

import pytest
from datasets import Dataset

from examples.diffusionnft_trainer.qwen_image_21.prepare_ocr_data import LEGACY_SYSTEM_PROMPT, convert_record, main


def test_ocr_conversion_preserves_records_and_splits(tmp_path):
    row = {
        "data_source": "flow_grpo/ocr",
        "prompt": [
            {"role": "system", "content": LEGACY_SYSTEM_PROMPT},
            {"role": "user", "content": 'The image displays "HELLO".'},
        ],
        "negative_prompt": [{"role": "user", "content": " "}],
        "reward_model": {"style": "model", "ground_truth": "HELLO"},
        "extra_info": {"index": 91, "split": "train"},
    }
    train = tmp_path / "original-train.parquet"
    val = tmp_path / "original-test.parquet"
    Dataset.from_list([row]).to_parquet(str(train))
    val_row = {**row, "prompt": row["prompt"][1:], "extra_info": {"index": 17, "split": "test"}}
    Dataset.from_list([val_row]).to_parquet(str(val))
    output = tmp_path / "converted"
    main(SimpleNamespace(train=train, val=val, output_dir=output))
    for split, original in (("train", row), ("val", val_row)):
        converted = Dataset.from_parquet(str(output / f"{split}.parquet"))[0]
        assert converted["prompt"] == row["prompt"][1:]
        assert converted["reward_model"] == original["reward_model"]
        assert converted["extra_info"] == original["extra_info"]


def test_ocr_conversion_rejects_unrecognized_system_instructions():
    row = {"prompt": [{"role": "system", "content": "custom instructions"}, {"role": "user", "content": "hello"}]}
    with pytest.raises(ValueError, match="documented legacy system"):
        convert_record(row)
