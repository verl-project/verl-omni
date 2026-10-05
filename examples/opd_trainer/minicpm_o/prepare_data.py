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
"""Convert MiniCPM simplex JSONL prompts and optional image/audio references to Parquet."""

import argparse
import json
from pathlib import Path

import pandas as pd


def convert_file(source: Path, destination: Path) -> None:
    """Embed images and resolve audio paths relative to a JSONL source file."""
    rows = []
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("videos"):
            raise ValueError("Video inputs are not supported by the simplex OPD recipe yet.")
        images = record.get("images", [])
        audios = record.get("audios", [])
        if len(audios) > 1:
            raise ValueError("At most one audio clip is supported per prompt.")
        prompt = record["prompt"]
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": "<image>\n" * len(images) + "<audio>\n" * len(audios) + prompt}]
        text = "".join(message["content"] for message in prompt)
        if text.count("<image>") != len(images) or text.count("<audio>") != len(audios):
            raise ValueError("Structured prompts must reference every supplied image/audio exactly once.")
        image_data = []
        for image in images:
            path = source.parent / (image["path"] if isinstance(image, dict) else image)
            image_data.append({"bytes": path.read_bytes(), "path": None})
        audio_paths = []
        for audio in audios:
            path = source.parent / (audio["path"] if isinstance(audio, dict) else audio)
            if not path.is_file():
                raise FileNotFoundError(path)
            audio_paths.append(str(path.resolve()))
        rows.append(
            {
                "data_source": record.get("data_source", "minicpm_o45_simplex"),
                "prompt": prompt,
                "images": image_data,
                "audios": audio_paths,
                "ability": "omni_understanding",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {"index": len(rows)},
            }
        )
    if not rows:
        raise ValueError(f"No prompts in {source}.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(destination, index=False)


def main():
    """Convert train.jsonl and test.jsonl into the recipe's two Parquet splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for split in ("train", "test"):
        convert_file(args.input_dir / f"{split}.jsonl", args.output_dir / f"{split}.parquet")


if __name__ == "__main__":
    main()
