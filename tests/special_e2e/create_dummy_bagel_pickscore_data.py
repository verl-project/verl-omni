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
"""Create dummy PickScore parquet data in BAGEL prompt-token format."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

USER_PROMPTS = [
    "A red circle on a white background",
    "A blue square on a black background",
    "A green triangle next to an orange rectangle",
    "A yellow star above a purple crescent moon",
]


def _tokenize_bagel_prompt(tokenizer, user_text: str, max_length: int = 256) -> list[int]:
    bos_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    raw_ids = tokenizer.encode(user_text, add_special_tokens=False)
    return ([bos_id] + raw_ids + [eos_id])[:max_length]


def build_rows(split: str, n: int, tokenizer, max_prompt_length: int):
    rows = []
    for i in range(n):
        caption = USER_PROMPTS[i % len(USER_PROMPTS)]
        prompt_token_ids = _tokenize_bagel_prompt(tokenizer, caption, max_length=max_prompt_length)
        rows.append(
            {
                "data_source": "flow_grpo/pickscore",
                "prompt": [{"role": "user", "content": caption}],
                "negative_prompt": [{"role": "user", "content": " "}],
                "prompt_token_ids": np.array(prompt_token_ids, dtype=np.int64),
                "ability": "pickscore",
                "reward_model": {"style": "model", "ground_truth": caption},
                "extra_info": {"split": split, "index": i},
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate dummy BAGEL PickScore parquet data")
    parser.add_argument(
        "--local_save_dir",
        default=os.path.expanduser("~/data/dummy_bagel_pickscore"),
    )
    parser.add_argument(
        "--model_path",
        default=os.path.expanduser("~/models/tiny-random/BAGEL-MoT"),
        help="Tokenizer path (tiny BAGEL checkpoint)",
    )
    parser.add_argument("--train_size", type=int, default=8)
    parser.add_argument("--val_size", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=64)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser(args.model_path), trust_remote_code=True)
    os.makedirs(args.local_save_dir, exist_ok=True)

    train_df = pd.DataFrame(build_rows("train", args.train_size, tokenizer, args.max_prompt_length))
    val_df = pd.DataFrame(build_rows("test", args.val_size, tokenizer, args.max_prompt_length))

    train_path = os.path.join(args.local_save_dir, "train.parquet")
    val_path = os.path.join(args.local_save_dir, "test.parquet")
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    print(f"Wrote {len(train_df)} train samples to {train_path}")
    print(f"Wrote {len(val_df)} val samples to {val_path}")


if __name__ == "__main__":
    main()
