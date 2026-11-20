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
"""Build the duplex episode parquet from a transcripts manifest.

One JSON line per dialogue::

    {"dialogue_id": "dt-0001",
     "user_audio": "/data/dailytalk/user/dt-0001.wav",
     "user_words": [{"text": "hello", "start_s": 0.52, "end_s": 0.91}, ...],
     "model_words": [{"text": "hi", "start_s": 1.20, "end_s": 1.45}, ...]}

Times are seconds into each track (the two-track corpus records the
speakers on separate channels; ``user_audio`` must already be the user
channel as a mono file — split the stereo mix first). Output rows are what
``DuplexEpisodeDataset`` reads: the mined episode spec, the user words
inside each window (track-absolute — the dataset relativizes), and the
sample's base seed. Run on CPU, before any training.

Usage::

    python scripts/duplex_build_episodes.py --manifest manifest.jsonl \\
        --output episodes.parquet [--val-fraction 0.05] [--seed 2026]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from verl_omni.utils.dataset.duplex.mining import WordTimestamp, mine_episodes

_AXIS_WEIGHTS = {"pause": 1.0, "turn": 1.0, "interruption": 1.0}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="JSONL manifest (one dialogue per line)")
    parser.add_argument("--output", required=True, help="output parquet path")
    parser.add_argument("--val-fraction", type=float, default=0.05, help="held-out dialogues, by dialogue id hash")
    parser.add_argument("--seed", type=int, default=2026, help="base seed column stamped per sample")
    parser.add_argument("--max-episodes", type=int, default=0, help="cap total episodes (0 = no cap)")
    parser.add_argument("--per-axis-cap", type=int, default=0, help="cap episodes per axis (0 = no cap)")
    return parser.parse_args(argv)


def build_rows(manifest_path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Mine every manifest dialogue into dataset rows."""
    rows: list[dict[str, Any]] = []
    axis_counts = {"pause": 0, "turn": 0, "interruption": 0}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        dialogue = json.loads(line)
        user_words = [
            WordTimestamp(text=str(word["text"]), start_s=float(word["start_s"]), end_s=float(word["end_s"]))
            for word in dialogue.get("user_words") or []
        ]
        model_words = [
            WordTimestamp(text=str(word["text"]), start_s=float(word["start_s"]), end_s=float(word["end_s"]))
            for word in dialogue.get("model_words") or []
        ]
        specs = mine_episodes(
            user_words,
            model_words,
            user_audio_ref=str(dialogue["user_audio"]),
            dialogue_id=str(dialogue["dialogue_id"]),
        )
        for spec in specs:
            if args.per_axis_cap and axis_counts[spec.axis] >= args.per_axis_cap:
                continue
            window_words = [
                {"start_s": word.start_s, "end_s": word.end_s}
                for word in user_words
                if word.end_s > spec.start_s and word.start_s < spec.end_s
            ]
            rows.append(
                {
                    "episode": spec.to_dict(),
                    "user_words": window_words,
                    "duplex_seed": args.seed,
                    "data_source": str(dialogue.get("data_source", "dailytalk")),
                    # Held-out split by dialogue, not by episode, so no
                    # dialogue straddles train and val.
                    "is_val": _is_val(str(dialogue["dialogue_id"]), args.val_fraction),
                }
            )
            axis_counts[spec.axis] += 1
    if args.max_episodes and len(rows) > args.max_episodes:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.max_episodes]
    return rows


def _is_val(dialogue_id: str, fraction: float) -> bool:
    return fraction > 0 and (hash(dialogue_id) % 1000) < int(fraction * 1000)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = build_rows(Path(args.manifest), args)
    if not rows:
        raise SystemExit("manifest produced no episodes; check the word timestamps and mining thresholds")
    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    train, val = frame[~frame["is_val"]], frame[frame["is_val"]]
    train_path = output.with_name(output.stem + "_train.parquet")
    val_path = output.with_name(output.stem + "_val.parquet")
    train.drop(columns=["is_val"]).to_parquet(train_path)
    val.drop(columns=["is_val"]).to_parquet(val_path)
    counts = frame["episode"].map(lambda spec: spec["axis"]).value_counts().to_dict()
    print(json.dumps({"train": len(train), "val": len(val), "axes": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
