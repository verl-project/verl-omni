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

"""
Preprocess the OCR dataset to parquet format for HTTP scorer service training.

Unlike qwenimage_ocr.py which extracts bare text as ground_truth, this script
stores the full prompt (containing quoted target text) as ground_truth so that
the HTTP scorer service can parse it directly.

Raw dataset: https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr
"""

import argparse
import os

import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="~/dataset/ocr/", help="Path to the raw OCR dataset directory.")
    parser.add_argument(
        "--output_dir", default="~/data/ocr_http", help="Directory to save the preprocessed parquet files."
    )
    args = parser.parse_args()

    local_dataset_path = os.path.expanduser(args.input_dir)
    dataset = datasets.load_dataset(local_dataset_path)

    data_source = "flow_grpo/ocr"
    system_prompt = (
        "Describe the image by detailing the color, shape, size, "
        "texture, quantity, text, spatial relationships of the objects and background:"
    )
    negative_user_prompt = " "

    def make_map_fn(split):
        def process_fn(example, idx):
            text = example.pop("text")
            data = {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                "negative_prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": negative_user_prompt},
                ],
                "ability": "ocr",
                "reward_model": {"style": "model", "ground_truth": text},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    train_dataset = dataset["train"].map(function=make_map_fn("train"), with_indices=True)
    test_dataset = dataset["test"].map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = os.path.expanduser(args.output_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    print(f"Saved to {local_save_dir}")
