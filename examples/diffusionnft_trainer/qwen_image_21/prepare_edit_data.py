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

"""将 UniRL 原始 train/test JSONL 转为编辑训练 parquet，保留全部记录和原图。"""

import argparse
import json
from io import BytesIO
from itertools import islice
from pathlib import Path

from PIL import Image

UNIRL_DATASET = "Tencent-Hunyuan/UniRL"


def convert_record(sample: dict, input_dir: Path, split: str, source_index: int) -> dict:
    """适配现有 RLHFDataset 和编辑 reward 合约，不改变原始指令或参考图。"""
    instruction = sample["prompt"]
    metadata = sample["metadata"]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("The UniRL prompt must be nonempty text.")
    if not isinstance(metadata["global_id"], str) or not metadata["global_id"]:
        raise ValueError("Each UniRL record requires a nonempty metadata.global_id.")
    media = sample["media"]
    if not 1 <= len(media) <= 4:
        raise ValueError("Editing requires 1–4 conditioning images.")
    marker_count = instruction.count("<image>")
    if marker_count not in (0, len(media)):
        raise ValueError("Existing <image> markers must match the number of conditioning images.")
    prompt = instruction if marker_count else "<image>\n" * len(media) + instruction
    references = []
    root = input_dir.resolve()
    for item in media:
        if item["modality"] != "image" or item["role"] != "condition":
            raise ValueError("UniRL editing media must be images with role=condition.")
        uri = Path(item["uri"])
        image_path = (root / uri).resolve()
        if uri.is_absolute() or not image_path.is_relative_to(root):
            raise ValueError("Condition image paths must remain inside input_dir.")
        payload = image_path.read_bytes()
        with Image.open(BytesIO(payload)) as image:
            image.load()
        references.append({"bytes": payload})
    return {
        "data_source": "unirl/image_edit",
        "prompt": [{"role": "user", "content": prompt}],
        "negative_prompt": [{"role": "user", "content": " "}],
        # RLHFDataset 会改写 actor 的 image dict；reward 保留独立的原图 dict。
        "images": [dict(reference) for reference in references],
        "reward_model": {"style": "model", "ground_truth": instruction},
        "extra_info": {
            "index": metadata["global_id"],
            "task": "image_edit",
            "dataset": UNIRL_DATASET,
            "split": split,
            "source_index": source_index,
            "source_record": json.dumps(sample, ensure_ascii=False),
            "reference_images": references,
        },
    }


def convert_split(input_dir: Path, output_dir: Path, split: str) -> int:
    """按原始顺序分批写入，避免在内存中保留整个图片数据集。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    count = 0
    writer = None
    schema = None
    try:
        with (input_dir / f"{split}.jsonl").open(encoding="utf-8") as source:
            while lines := list(islice(source, 32)):
                records = [
                    convert_record(json.loads(line), input_dir, split, count + index)
                    for index, line in enumerate(lines)
                ]
                table = pa.Table.from_pylist(records, schema=schema)
                if writer is None:
                    schema = table.schema
                    writer = pq.ParquetWriter(output_dir / f"{split}.parquet", schema)
                writer.write_table(table)
                count += len(records)
    finally:
        if writer is not None:
            writer.close()
    if not count:
        raise ValueError(f"The UniRL {split} split is empty.")
    return count


def main(args) -> None:
    """直接转换发布者的 train/test 划分，不抽样、不去重或重新划分。"""
    args.output_dir.mkdir(parents=True, exist_ok=False)
    counts = {split: convert_split(args.input_dir, args.output_dir, split) for split in ("train", "test")}
    print(f"Wrote {counts['train']} training and {counts['test']} test records to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="Directory with UniRL train/test.jsonl and media paths"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    main(parser.parse_args())
