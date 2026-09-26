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
"""Convert native OmniNFT prompt-group JSONL files to standard RLHF parquet."""

import argparse
import io
import json
import urllib.request
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

_OMNINFT_REVISION = "fb9237f6e74edf0d0f2a683f4d975b79fde588fe"
_DATA_BASE_URL = f"https://raw.githubusercontent.com/zghhui/OmniNFT/{_OMNINFT_REVISION}/dataset/vggsound"
_TRAIN_URL = f"{_DATA_BASE_URL}/train_metadata_20k.jsonl"
_VAL_URL = f"{_DATA_BASE_URL}/test_metadata.jsonl"
_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data/omninft/vggsound/verl_omni"
_PROMPT_FIELDS = ("prompt_av", "prompt_v", "prompt_a")
_PROVENANCE_FIELDS = ("idx", "category")


def _open_source(location: str) -> TextIO:
    if location.startswith(("http://", "https://")):
        request = urllib.request.Request(location, headers={"User-Agent": "verl-omni-data-preparation"})
        return io.TextIOWrapper(urllib.request.urlopen(request, timeout=60), encoding="utf-8")
    return Path(location).expanduser().open(encoding="utf-8")


def _read_records(location: str, split: str, max_samples: int) -> list[dict[str, Any]]:
    """Read prompt-group JSONL from a local path or URL into RLHF rows.

    Args:
        location: UTF-8 JSONL source; blank lines are skipped.
        split: Stored split name and default category (``test`` uses ``validation``).
        max_samples: Maximum accepted rows; negative reads all, zero returns none.
            Records after the limit are not parsed or validated.

    Returns:
        Rows mapping prompt_av to the user prompt and prompt_v/prompt_a to
        reward_inputs.text.video/audio, with an empty negative prompt. ``uid``
        is the string form of idx, a prompt-group identity rather than a rollout
        sample ID. When both idx and category are absent, use the accepted-row
        index and default category. extra_info retains provenance and the full
        source record; the source is not modified.

    Raises:
        ValueError: Malformed JSON/object, missing or blank prompt strings,
            partial/invalid provenance, or duplicate uid among accepted rows.
            Supplied idx must be a non-bool integer and category a nonempty string.
    """
    rows = []
    seen_uids = set()
    with _open_source(location) as source:
        for line_number, line in enumerate(source, start=1):
            if max_samples >= 0 and len(rows) >= max_samples:
                break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {location}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {location}:{line_number}")

            missing_prompts = [key for key in _PROMPT_FIELDS if key not in record]
            if missing_prompts:
                raise ValueError(f"Missing fields at {location}:{line_number}: {', '.join(missing_prompts)}")
            for key in _PROMPT_FIELDS:
                if not isinstance(record[key], str) or not record[key].strip():
                    raise ValueError(f"Field '{key}' must be a non-empty string at {location}:{line_number}")

            present_provenance = [key for key in _PROVENANCE_FIELDS if key in record]
            if present_provenance and len(present_provenance) != len(_PROVENANCE_FIELDS):
                missing = [key for key in _PROVENANCE_FIELDS if key not in record]
                raise ValueError(f"Missing fields at {location}:{line_number}: {', '.join(missing)}")
            if present_provenance:
                index = record["idx"]
                category = record["category"]
                if isinstance(index, bool) or not isinstance(index, int):
                    raise ValueError(f"Field 'idx' must be an integer at {location}:{line_number}")
                if not isinstance(category, str) or not category.strip():
                    raise ValueError(f"Field 'category' must be a non-empty string at {location}:{line_number}")
            else:
                index = len(rows)
                category = "validation" if split == "test" else split

            uid = str(index)
            if uid in seen_uids:
                raise ValueError(f"Duplicate prompt-group idx={uid} at {location}:{line_number}")
            seen_uids.add(uid)
            rows.append(
                {
                    "data_source": "omninft_vggsound",
                    "prompt": [{"role": "user", "content": record["prompt_av"]}],
                    "negative_prompt": [{"role": "user", "content": ""}],
                    "ability": "text_to_audio_video",
                    "uid": uid,
                    "reward_model": {"style": "model", "ground_truth": record["prompt_av"]},
                    "reward_inputs": {"text": {"video": record["prompt_v"], "audio": record["prompt_a"]}},
                    "extra_info": {
                        "split": split,
                        "index": index,
                        "category": category,
                        "metadata": record,
                    },
                }
            )
    return rows


def _convert_split(location: str, split: str, max_samples: int) -> pd.DataFrame:
    return pd.DataFrame(_read_records(location, split, max_samples))


def main() -> None:
    """Convert the selected train/validation rows to train.parquet and test.parquet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", default=_TRAIN_URL, help="Training JSONL path or URL")
    parser.add_argument("--val_file", default=_VAL_URL, help="Validation JSONL path or URL")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory for train.parquet and test.parquet",
    )
    parser.add_argument("--train_size", type=int, default=-1)
    parser.add_argument("--val_size", type=int, default=-1)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    train = _convert_split(args.train_file, "train", args.train_size)
    validation = _convert_split(args.val_file, "test", args.val_size)
    train.to_parquet(output_dir / "train.parquet", index=False, row_group_size=500)
    validation.to_parquet(output_dir / "test.parquet", index=False, row_group_size=500)
    print(f"Wrote {len(train)} training and {len(validation)} validation samples to {output_dir}")


if __name__ == "__main__":
    main()
