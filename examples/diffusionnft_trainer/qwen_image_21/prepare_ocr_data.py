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

"""将现有 Flow-GRPO OCR parquet 转成 Qwen-Image 2.1 的单 user 数据合约。"""

import argparse
from pathlib import Path

from datasets import Dataset

LEGACY_SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
)


def convert_record(row):
    """保留原提示词、目标文字与样本编号，只移除旧模型专用的 system message。"""
    messages = row["prompt"]
    if len(messages) == 2 and messages[0] == {"role": "system", "content": LEGACY_SYSTEM_PROMPT}:
        messages = messages[1:]
    if len(messages) != 1 or messages[0]["role"] != "user" or not isinstance(messages[0]["content"], str):
        raise ValueError("Expected an OCR text prompt with the documented legacy system message or one user message.")
    if row["data_source"] != "flow_grpo/ocr" or not isinstance(row["reward_model"]["ground_truth"], str):
        raise ValueError("Expected the existing Flow-GRPO OCR source and ground-truth text.")
    return {"prompt": messages, "negative_prompt": [{"role": "user", "content": " "}]}


def main(args):
    """保留 train/val 划分，写入训练使用的 parquet。"""
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for split, source in (("train", args.train), ("val", args.val)):
        dataset = Dataset.from_parquet(str(source)).map(convert_record)
        dataset.to_parquet(str(args.output_dir / f"{split}.parquet"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args())
