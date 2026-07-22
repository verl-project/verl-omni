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
Preprocess the OCR dataset to parquet format for Boogu-Image FlowGRPO training.
You can obtain the raw dataset from https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr

The system prompts below are ported verbatim from the upstream
``BooguImagePipeline`` (https://github.com/boogu-project/Boogu-Image). Two
details matter for rollout/training consistency:

- The positive prompt uses the T2I system prompt.
- The upstream pipeline encodes **empty** instructions — including the default
  negative prompt ``""`` — with the *TI2I unified* system prompt, not the T2I
  one. The negative prompt below replicates that quirk exactly; changing
  either template silently shifts the policy's prompt distribution.
"""

import argparse
import os

import datasets
from verl.utils.hdfs_io import copy, makedirs

# Ported verbatim from BooguImagePipeline.__init__ (SYSTEM_PROMPT_4_T2I_UNIFIED).
BOOGU_SYSTEM_PROMPT_T2I = (
    "You are a helpful assistant that generates high-quality images based on user instructions. "
    "The instructions are as follows."
)

# Ported verbatim from BooguImagePipeline.__init__ (SYSTEM_PROMPT_4_TI2I_UNIFIED);
# upstream uses it for empty instructions, which is what the default negative
# prompt "" hits (SYSTEM_PROMPT_DROP).
BOOGU_SYSTEM_PROMPT_DROP = (
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency with the original input where appropriate."
)


def extract_solution(solution_str):
    # The solution is stored in the format: 'The image displays "xxx".'
    return solution_str.split('"')[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--input_dir", default="~/data/ocr/", help="Path to the raw OCR dataset directory.")
    parser.add_argument(
        "--output_dir", default="~/data/ocr/boogu_image", help="Directory to save the preprocessed parquet files."
    )

    args = parser.parse_args()
    local_dataset_path = os.path.expanduser(args.input_dir)

    data_source = "flow_grpo/ocr"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        raise NotImplementedError(
            "It is not existed in huggingface hub. "
            "Please get dataset from https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr"
        )

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    def make_map_fn(split):
        def process_fn(example, idx):
            text = example.pop("text")
            solution = extract_solution(text)
            data = {
                "data_source": data_source,
                "prompt": [
                    {"role": "system", "content": BOOGU_SYSTEM_PROMPT_T2I},
                    {"role": "user", "content": text},
                ],
                "negative_prompt": [
                    {"role": "system", "content": BOOGU_SYSTEM_PROMPT_DROP},
                    {"role": "user", "content": ""},
                ],
                "ability": "ocr",
                "reward_model": {"style": "model", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = os.path.expanduser(args.output_dir)
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
