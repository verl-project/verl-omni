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

"""Convert AlphaGRPO question JSONL to BAGEL FlowGRPO parquet."""

import argparse
import json
from pathlib import Path

import datasets
from transformers import AutoTokenizer

from verl_omni.utils.reward_score.dvreward import normalize_questions


def prepare_record(record: dict, tokenizer, max_prompt_length: int, split: str, index: int) -> dict:
    """Preserve native BAGEL token framing and the two question groups."""
    caption = record["prompt"]
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("DVReward prompt must be a non-empty string")
    caption = caption.strip()
    prompt_ids = [
        tokenizer.convert_tokens_to_ids("<|im_start|>"),
        *tokenizer.encode(caption, add_special_tokens=False),
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
    ]
    if len(prompt_ids) > max_prompt_length:
        raise ValueError(f"Prompt {index} exceeds max_prompt_length={max_prompt_length}; do not truncate its questions")
    return {
        "data_source": "alphagrpo/dvreward",
        "prompt": [{"role": "user", "content": caption}],
        "negative_prompt": [{"role": "user", "content": " "}],
        "prompt_token_ids": prompt_ids,
        "ability": "dvreward",
        "reward_model": {"style": "model", "ground_truth": caption},
        "extra_info": {
            "split": split,
            "index": index,
            "semantic_questions": normalize_questions(record["semantic_questions"]),
            "quality_questions": normalize_questions(record["quality_questions"]),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        with (args.input_dir / f"{split}.jsonl").open() as source:
            records = [
                prepare_record(json.loads(line), tokenizer, args.max_prompt_length, split, index)
                for index, line in enumerate(source)
            ]
        datasets.Dataset.from_list(records).to_parquet(str(args.output_dir / f"{split}.parquet"))
