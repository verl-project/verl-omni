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
"""Create the deterministic dataset used by a nightly run.

OCR and AVQA rows repeat one sample. Math rows come from
``tests/special_e2e/create_dummy_math_data.py`` and keep that schema.
AVQA rows go through the same converter as ``tests/special_e2e/create_dummy_minicpmo_avqa_data.py``.
Every row gets ``extra_info.repeat_index``, the run-stable key used by dump comparison.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import os
from pathlib import Path
from typing import Any

import pandas as pd


def _repeat(rows: dict[str, Any] | list[dict[str, Any]], split: str, size: int | None = None) -> list[dict[str, Any]]:
    if isinstance(rows, dict):
        if size is None:
            raise ValueError("size is required when repeating one row")
        sources = [rows] * size
    else:
        sources = list(rows)
    repeated = []
    for index, row in enumerate(sources):
        item = copy.deepcopy(row)
        extra = dict(item.get("extra_info") or {})
        extra["split"] = split
        extra["repeat_index"] = index
        item["extra_info"] = extra
        repeated.append(item)
    return repeated


def _write_split(path: Path, rows: list[dict[str, Any]]) -> None:
    # Nested columns crash some PyArrow 25 readers when dictionary-encoded.
    pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False, use_dictionary=False)
    print(f"Wrote {len(rows)} rows to {path}")


def _ocr_row(split: str, data_source: str) -> dict[str, Any]:
    SYSTEM_PROMPT = (
        "Describe the image by detailing the color, shape, size, texture, quantity, text, "
        "spatial relationships of the objects and background:"
    )
    USER_PROMPT = "A clean white square image with the black text CI centered in large bold letters"
    GROUND_TRUTH = "CI"
    row = {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        "negative_prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "blurry, low quality, distorted text"},
        ],
        "reward_model": {"style": "rule", "ground_truth": GROUND_TRUTH},
        "extra_info": {"split": split, "index": 0, "source_index": 0},
    }
    return row


def _load_avqa_builder():
    path = Path(__file__).resolve().parents[1] / "special_e2e" / "create_dummy_minicpmo_avqa_data.py"
    spec = importlib.util.spec_from_file_location("create_dummy_minicpmo_avqa_data", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load AVQA dummy builder from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _avqa_row(builder: Any, output_dir: Path, split: str, data_source: str) -> dict[str, Any]:
    stem = f"{split}_0"
    image_name = f"{stem}.png"
    audio_name = f"{stem}.wav"
    builder._write_image(output_dir / image_name, 0)
    builder._write_audio(output_dir / audio_name, 0)
    row, drop_reason = builder.build_rl_row(
        record={
            "problem": "Which sound is present in the clip?",
            "data_type": "image_audio",
            "options": ["A. A tone", "B. Silence"],
            "solution": "<answer>A</answer>",
            "path": {"image": image_name, "audio": audio_name},
            "problem_id": stem,
            "problem_type": "nightly",
        },
        split_dir=output_dir,
        split=split,
        index=0,
    )
    if row is None:
        raise ValueError(f"Dummy AVQA row {stem} was rejected by the converter: {drop_reason}")
    row["data_source"] = data_source
    return row


def _load_math_builder():
    path = Path(__file__).resolve().parents[1] / "special_e2e" / "create_dummy_math_data.py"
    spec = importlib.util.spec_from_file_location("create_dummy_math_data", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load math dummy builder from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_ocr(data_source: str, train_size: int, val_size: int, train_path: Path, val_path: Path) -> None:
    _write_split(train_path, _repeat(_ocr_row("train", data_source), "train", train_size))
    _write_split(val_path, _repeat(_ocr_row("test", data_source), "test", val_size))


def write_math(train_size: int, val_size: int, train_path: Path, val_path: Path) -> None:
    """Write GSM8K-format different fixed rows and stamp ``extra_info.repeat_index``."""
    builder = _load_math_builder()
    for split, size, path in (("train", train_size, train_path), ("test", val_size, val_path)):
        _write_split(path, _repeat(builder.build_rows(split, size), split))


def write_avqa(
    output_dir: Path,
    data_source: str,
    train_size: int,
    val_size: int,
    train_path: Path,
    val_path: Path,
) -> None:
    builder = _load_avqa_builder()
    train_row = _avqa_row(builder, output_dir, "train", data_source)
    val_row = _avqa_row(builder, output_dir, "test", data_source)
    _write_split(train_path, _repeat(train_row, "train", train_size))
    _write_split(val_path, _repeat(val_row, "test", val_size))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate repeated single-sample L3 data")
    parser.add_argument(
        "--local_save_dir",
        default=os.path.expanduser("~/data/qwen_image_flowgrpo"),
        help="Directory to write train.parquet and test.parquet",
    )
    parser.add_argument("--train_size", type=int, default=16, help="Repeated train rows")
    parser.add_argument("--val_size", type=int, default=4, help="Repeated validation rows")
    parser.add_argument(
        "--data_source",
        default="qwen_image_flowgrpo",
        help="Value stored in the data_source column.",
    )
    parser.add_argument("--kind", choices=("ocr", "avqa", "math"), default="ocr")
    args = parser.parse_args()

    output_dir = Path(os.path.expanduser(args.local_save_dir))
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "test.parquet"

    # Some cases write their own parquet first, skipping default ocr creation.
    if train_path.is_file() and val_path.is_file():
        print(f"Dataset already exists in {output_dir}. Skipping creation.")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.kind == "ocr":
        write_ocr(args.data_source, args.train_size, args.val_size, train_path, val_path)
    elif args.kind == "avqa":
        write_avqa(output_dir, args.data_source, args.train_size, args.val_size, train_path, val_path)
    elif args.kind == "math":
        write_math(args.train_size, args.val_size, train_path, val_path)
    else:
        raise ValueError(f"Unsupported kind: {args.kind}")


if __name__ == "__main__":
    main()
