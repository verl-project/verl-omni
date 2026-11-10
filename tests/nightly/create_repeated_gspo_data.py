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
"""Write a one-prompt dataset for thinker GSPO nightly runs.

Every train and validation row is a copy of the same sample. ``extra_info.repeat_index``
is the run-stable row key used by dump comparison. Math rows match the GSM8K schema
used by ``tests/special_e2e/create_dummy_math_data.py``. AVQA rows go through the same
converter as ``tests/special_e2e/create_dummy_minicpmo_avqa_data.py``.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import os
from pathlib import Path
from typing import Any

import pandas as pd

_GSM8K_SOURCE = "openai/gsm8k"
_GSM8K_INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'
_GSM8K_QUESTION = "What is 2 + 3?"
_GSM8K_ANSWER = "5"


def _repeat(row: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    item = copy.deepcopy(row)
    extra = dict(item.get("extra_info") or {})
    extra["split"] = split
    extra["index"] = index
    extra["repeat_index"] = index
    item["extra_info"] = extra
    return item


def _write_split(path: Path, row: dict[str, Any], split: str, size: int) -> None:
    rows = [_repeat(row, split, index) for index in range(size)]
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"Wrote {size} repeated {split} rows to {path}")


def _math_row() -> dict[str, Any]:
    return {
        "data_source": _GSM8K_SOURCE,
        "prompt": [{"role": "user", "content": f"{_GSM8K_QUESTION} {_GSM8K_INSTRUCTION}"}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": _GSM8K_ANSWER},
        "extra_info": {
            "split": "train",
            "index": 0,
            "answer": _GSM8K_ANSWER,
            "question": _GSM8K_QUESTION,
            "repeat_index": 0,
        },
    }


def write_math(output_dir: Path, train_size: int, val_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    row = _math_row()
    _write_split(output_dir / "train.parquet", row, "train", train_size)
    _write_split(output_dir / "test.parquet", row, "test", val_size)


def _load_avqa_builder():
    path = Path(__file__).resolve().parents[1] / "special_e2e" / "create_dummy_minicpmo_avqa_data.py"
    spec = importlib.util.spec_from_file_location("create_dummy_minicpmo_avqa_data", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load AVQA dummy builder from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _avqa_row(builder: Any, output_dir: Path, split: str) -> dict[str, Any]:
    stem = f"{split}_0"
    image_name = f"{stem}.png"
    audio_name = f"{stem}.wav"
    builder._write_image(output_dir / image_name, 0)
    builder._write_audio(output_dir / audio_name, 0)
    row, drop_reason = builder.build_rl_row(
        {
            "problem": "Which sound is present in the clip?",
            "data_type": "image_audio",
            "options": ["A. A tone", "B. Silence"],
            "solution": "<answer>A</answer>",
            "path": {"image": image_name, "audio": audio_name},
            "problem_id": stem,
            "problem_type": "nightly",
        },
        output_dir,
        split,
        0,
    )
    if row is None:
        raise ValueError(f"Dummy AVQA row {stem} was rejected by the converter: {drop_reason}")
    return row


def write_avqa(output_dir: Path, train_size: int, val_size: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    builder = _load_avqa_builder()
    _write_split(output_dir / "train.parquet", _avqa_row(builder, output_dir, "train"), "train", train_size)
    _write_split(
        output_dir / "validation.parquet",
        _avqa_row(builder, output_dir, "validation"),
        "validation",
        val_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a repeated one-prompt thinker GSPO dataset")
    parser.add_argument("--kind", choices=("math", "avqa"), required=True)
    parser.add_argument("--local_save_dir", required=True, help="Directory for the parquet (and AVQA media)")
    parser.add_argument("--train_size", type=int, required=True)
    parser.add_argument("--val_size", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(os.path.expanduser(args.local_save_dir))
    if args.kind == "math":
        write_math(output_dir, args.train_size, args.val_size)
    else:
        write_avqa(output_dir, args.train_size, args.val_size)


if __name__ == "__main__":
    main()
