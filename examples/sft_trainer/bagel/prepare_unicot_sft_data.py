# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Convert BAGEL example data into Uni-COT SFT JSONL shards.

The converter is intentionally lightweight and local-only. It prepares rows for
``UniCOTSFTDataset`` without running VAE/ViT preprocessing or downloading data.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

IMAGE_START = "<image_start>"


def _save_image_bytes(payload: bytes, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    image.save(path)
    return str(path)


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _caption_from_row(row) -> str:
    captions = row.get("captions", None)
    if captions is None:
        return ""
    if isinstance(captions, str):
        try:
            captions = json.loads(captions)
        except json.JSONDecodeError:
            return captions
    if isinstance(captions, dict) and captions:
        return str(next(iter(captions.values())))
    return ""


def convert_t2i(input_dir: Path, image_dir: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parquet_path in sorted(input_dir.rglob("*.parquet")):
        frame = pd.read_parquet(parquet_path)
        for row_idx, row in frame.iterrows():
            if limit is not None and len(rows) >= limit:
                return rows
            caption = _caption_from_row(row)
            image_path = _save_image_bytes(
                row["image"],
                image_dir / "t2i" / f"{parquet_path.stem}_{row_idx}.png",
            )
            rows.append(
                {
                    "task_type": "t2i",
                    "image_list": [image_path],
                    "instruction_list": [caption],
                    "output_text_list": [IMAGE_START],
                    "extra_info": {"source": "t2i", "parquet": str(parquet_path), "row": int(row_idx)},
                }
            )
    return rows


def _normalise_instruction(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def convert_editing(input_dir: Path, image_dir: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for parquet_path in sorted(input_dir.rglob("*.parquet")):
        frame = pd.read_parquet(parquet_path)
        for row_idx, row in frame.iterrows():
            if limit is not None and len(rows) >= limit:
                return rows
            image_list = []
            for image_idx, payload in enumerate(row["image_list"]):
                image_list.append(
                    _save_image_bytes(
                        payload,
                        image_dir / "editing" / f"{parquet_path.stem}_{row_idx}_{image_idx}.png",
                    )
                )
            instructions = [_normalise_instruction(item) for item in row.get("instruction_list", [])]
            output_text_list = [IMAGE_START for _ in image_list[1:]]
            rows.append(
                {
                    "task_type": "editing",
                    "image_list": image_list,
                    "instruction_list": instructions,
                    "output_text_list": output_text_list,
                    "extra_info": {"source": "editing", "parquet": str(parquet_path), "row": int(row_idx)},
                }
            )
    return rows


def convert_vlm(jsonl_path: Path, image_root: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            if limit is not None and len(rows) >= limit:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            images = item.get("image", [])
            if isinstance(images, str):
                images = [images]
            image_list = [str(image_root / image) for image in images]
            instruction_list = []
            output_text_list = []
            for turn in item.get("conversations", []):
                value = str(turn.get("value", "")).replace("<image>", "").strip()
                if turn.get("from") == "human":
                    instruction_list.append(value)
                elif turn.get("from") == "gpt":
                    output_text_list.append(value)
            rows.append(
                {
                    "task_type": "vlm_sft",
                    "image_list": image_list,
                    "instruction_list": instruction_list,
                    "output_text_list": output_text_list,
                    "extra_info": {"source": "vlm_sft", "jsonl": str(jsonl_path), "row": int(row_idx)},
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BAGEL example data to Uni-COT SFT JSONL.")
    parser.add_argument("--bagel_example_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--limit_per_task", type=int, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_image_dir = args.output_dir / "images"
    rows: list[dict[str, Any]] = []
    rows.extend(convert_t2i(args.bagel_example_dir / "t2i", output_image_dir, args.limit_per_task))
    rows.extend(
        convert_editing(args.bagel_example_dir / "editing" / "seedxedit_multi", output_image_dir, args.limit_per_task)
    )
    rows.extend(
        convert_vlm(
            args.bagel_example_dir / "vlm" / "llava_ov_si.jsonl",
            args.bagel_example_dir / "vlm" / "images",
            args.limit_per_task,
        )
    )

    random.Random(args.seed).shuffle(rows)
    val_count = int(len(rows) * args.val_ratio)
    val_rows = rows[:val_count]
    train_rows = rows[val_count:]
    _write_jsonl(train_rows, args.output_dir / "train.jsonl")
    _write_jsonl(val_rows or train_rows[:1], args.output_dir / "val.jsonl")
    print(f"Saved {len(train_rows)} train rows and {len(val_rows or train_rows[:1])} val rows to {args.output_dir}")


if __name__ == "__main__":
    main()
