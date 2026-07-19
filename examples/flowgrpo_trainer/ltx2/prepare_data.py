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

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_prompts(input_dir: Path, split: str) -> list[str]:
    text_path = input_dir / f"{split}.txt"
    jsonl_path = input_dir / f"{split}.jsonl"
    if text_path.is_file():
        with text_path.open(encoding="utf-8") as source:
            return [line.strip() for line in source if line.strip()]
    if jsonl_path.is_file():
        prompts = []
        with jsonl_path.open(encoding="utf-8") as source:
            for line in source:
                example = json.loads(line)
                prompt = example.get("prompt", example.get("text", example.get("caption")))
                if prompt is None:
                    raise KeyError(f"No prompt/text/caption field in {jsonl_path}: {example.keys()}.")
                prompts.append(str(prompt).strip())
        return [prompt for prompt in prompts if prompt]
    raise FileNotFoundError(f"Expected {text_path} or {jsonl_path}.")


def _convert_split(input_dir: Path, split: str, max_samples: int) -> pd.DataFrame:
    prompts = _read_prompts(input_dir, split)
    if max_samples >= 0:
        prompts = prompts[:max_samples]

    rows = []
    for index, prompt in enumerate(prompts):
        rows.append(
            {
                "data_source": "ltx2_t2av",
                "prompt": [{"role": "user", "content": prompt}],
                "negative_prompt": [{"role": "user", "content": ""}],
                "ability": "text_to_audio_video",
                "reward_model": {"style": "rule", "ground_truth": prompt},
                "extra_info": {"split": split, "index": index},
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_size", type=int, default=1024)
    parser.add_argument("--val_size", type=int, default=-1)
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _convert_split(input_dir, "train", args.train_size)
    validation = _convert_split(input_dir, "test", args.val_size)
    train.to_parquet(output_dir / "train.parquet", row_group_size=500)
    validation.to_parquet(output_dir / "test.parquet", row_group_size=500)
    print(f"Wrote {len(train)} training and {len(validation)} validation samples to {output_dir}")


if __name__ == "__main__":
    main()
