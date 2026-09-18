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
"""Convert MiniMax H3 Ref2VA JSONL splits to verl-omni parquet files.

Each JSONL row carries a prompt plus any mix of image, video and audio
references (up to twelve files total). References may be given either as the
plural ``images``/``videos``/``audios`` lists or the singular
``image``/``video``/``audio`` keys.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

_MAX_IMAGES = 9
_MAX_VIDEOS = 3
_MAX_AUDIOS = 3
_MAX_REFERENCES = 12


def _as_list(example: dict[str, Any], plural: str, singular: str) -> list[Any]:
    values = example.get(plural)
    if values is None and singular in example:
        values = [example[singular]]
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError(f"{plural} must be a list.")
    return values


def _resolve_path(input_dir: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} entries must be non-empty paths.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = input_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field} file not found: {path}")
    return path


def _video_specs(input_dir: Path, values: list[Any]) -> tuple[list[dict[str, Any]], list[str]]:
    videos = []
    sources = []
    for value in values:
        if isinstance(value, str):
            path = _resolve_path(input_dir, value, field="video")
            start = None
        elif isinstance(value, dict):
            path = _resolve_path(
                input_dir,
                value.get("path", value.get("video", value.get("video_path"))),
                field="video",
            )
            start = value.get("start_time_seconds")
            if start is not None:
                start = float(start)
                if start < 0:
                    raise ValueError(f"video start_time_seconds must be non-negative, got {start}.")
        else:
            raise ValueError("videos entries must be paths or objects containing a path.")
        videos.append({"video": str(path), "start_time_seconds": start})
        sources.append(str(path))
    return videos, sources


def _convert_split(input_dir: Path, split: str, max_samples: int) -> pd.DataFrame:
    rows = []
    with (input_dir / f"{split}.jsonl").open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if max_samples >= 0 and index >= max_samples:
                break
            example = json.loads(line)
            prompt = str(example["prompt"]).strip()
            if not prompt:
                raise ValueError(f"Empty prompt at {split} row {index}.")

            image_values = _as_list(example, "images", "image")
            video_values = _as_list(example, "videos", "video")
            audio_values = _as_list(example, "audios", "audio")
            if not image_values and not video_values:
                raise ValueError(f"Ref2VA row {index} requires at least one image or video reference.")
            if len(image_values) > _MAX_IMAGES or len(video_values) > _MAX_VIDEOS or len(audio_values) > _MAX_AUDIOS:
                raise ValueError(
                    f"Ref2VA row {index} exceeds reference limits: "
                    f"images={len(image_values)}, videos={len(video_values)}, audios={len(audio_values)}."
                )
            if len(image_values) + len(video_values) + len(audio_values) > _MAX_REFERENCES:
                raise ValueError(f"Ref2VA row {index} exceeds the {_MAX_REFERENCES}-file reference limit.")

            image_paths = [_resolve_path(input_dir, value, field="image") for value in image_values]
            videos, video_sources = _video_specs(input_dir, video_values)
            audio_paths = [_resolve_path(input_dir, value, field="audio") for value in audio_values]
            content = "<image>" * len(image_paths) + "<video>" * len(videos) + "<audio>" * len(audio_paths) + prompt
            rows.append(
                {
                    "data_source": "minimax_h3_ref2va",
                    "prompt": [{"role": "user", "content": content}],
                    "ability": "reference_to_audio_video",
                    "images": [{"bytes": path.read_bytes()} for path in image_paths],
                    "videos": videos,
                    "audios": [str(path) for path in audio_paths],
                    "reward_model": {"style": "model", "ground_truth": prompt},
                    "extra_info": {
                        "split": split,
                        "index": index,
                        "source_images": [str(path) for path in image_paths],
                        "source_videos": video_sources,
                        "source_audios": [str(path) for path in audio_paths],
                    },
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_size", type=int, default=-1)
    parser.add_argument("--val_size", type=int, default=-1)
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = _convert_split(input_dir, "train", args.train_size)
    validation = _convert_split(input_dir, "test", args.val_size)
    train.to_parquet(args.output_dir / "train.parquet", row_group_size=500)
    validation.to_parquet(args.output_dir / "test.parquet", row_group_size=500)
    print(f"Wrote {len(train)} training and {len(validation)} validation rows to {args.output_dir}")


if __name__ == "__main__":
    main()
