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
inside each window (track-absolute — the dataset relativizes), the source
``dialogue_id``, the sample's base seed, and ``commit=true``. This checkpoint
does not open a response on its own; the dataset reads that column and the
driver commits only when it is true. Run on CPU, before any training.

The val split is by dialogue, on ``crc32(dialogue_id)`` (stable across
processes). ``--per-axis-cap auto`` then balances the axes inside each split
to that split's smallest axis, so neither parquet is dominated by one axis.

Usage::

    python scripts/duplex_build_episodes.py --manifest manifest.jsonl \\
        --output episodes.parquet [--val-fraction 0.05] [--seed 2026] \\
        [--per-axis-cap auto|N] [--min-pause-s 0.6 --max-pause-s 1.8 --min-interrupt-s 1.5]
"""

from __future__ import annotations

import argparse
import json
import random
import zlib
from pathlib import Path
from typing import Any

import pandas as pd

from verl_omni.utils.dataset.duplex.mining import WordTimestamp, mine_episodes


def _per_axis_cap(value: str) -> int | None:
    """``auto`` → ``None`` (balance per split after the val split); an int cap; 0 = no cap."""
    if value == "auto":
        return None
    try:
        cap = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"per-axis cap must be 'auto' or an integer, got {value!r}") from error
    if cap < 0:
        raise argparse.ArgumentTypeError(f"per-axis cap must be >= 0, got {cap}")
    return cap


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="JSONL manifest (one dialogue per line)")
    parser.add_argument("--output", required=True, help="output parquet path")
    parser.add_argument(
        "--val-fraction", type=float, default=0.05, help="held-out dialogues, by crc32 of the dialogue id"
    )
    parser.add_argument("--seed", type=int, default=2026, help="base seed column stamped per sample")
    parser.add_argument("--max-episodes", type=int, default=0, help="cap total episodes (0 = no cap)")
    parser.add_argument(
        "--per-axis-cap",
        type=_per_axis_cap,
        default=None,
        help="cap episodes per axis: 'auto' (default) balances every axis to the smallest one inside each split "
        "after the val split; an integer caps each axis during mining; 0 = no cap",
    )
    parser.add_argument("--min-pause-s", type=float, default=0.6, help="shortest same-speaker silence mined as a pause")
    parser.add_argument("--max-pause-s", type=float, default=1.8, help="longest same-speaker silence mined as a pause")
    parser.add_argument(
        "--min-interrupt-s", type=float, default=1.5, help="shortest user utterance mined as an interrupting clip"
    )
    args = parser.parse_args(argv)
    if args.min_pause_s > args.max_pause_s:
        parser.error(f"--min-pause-s ({args.min_pause_s}) must not exceed --max-pause-s ({args.max_pause_s})")
    return args


def build_rows(manifest_path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Mine every manifest dialogue into dataset rows (the val split flagged per row)."""
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
            min_pause_s=args.min_pause_s,
            max_pause_s=args.max_pause_s,
            min_interrupt_s=args.min_interrupt_s,
        )
        dialogue_id = str(dialogue["dialogue_id"])
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
                    "dialogue_id": dialogue_id,
                    "commit": True,
                    # Held-out split by dialogue, not by episode, so no
                    # dialogue straddles train and val.
                    "is_val": _is_val(dialogue_id, args.val_fraction),
                }
            )
            axis_counts[spec.axis] += 1
    if args.max_episodes and len(rows) > args.max_episodes:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.max_episodes]
    return rows


def _is_val(dialogue_id: str, fraction: float) -> bool:
    """Whether a dialogue is held out: ``crc32(dialogue_id)`` against the fraction."""
    return fraction > 0 and (zlib.crc32(dialogue_id.encode("utf-8")) % 1000) < int(fraction * 1000)


def _balance_axes(split: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Cap every axis to ``split``'s smallest axis (seeded choice of the rows kept)."""
    if split.empty:
        return split
    axes = split["episode"].map(lambda spec: spec["axis"])
    cap = int(axes.value_counts().min())
    return split.groupby(axes).sample(n=cap, random_state=seed).sort_index()


def _split_summary(split: pd.DataFrame) -> dict[str, Any]:
    axes = split["episode"].map(lambda spec: spec["axis"]).value_counts().to_dict()
    return {"episodes": len(split), "axes": axes}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows = build_rows(Path(args.manifest), args)
    if not rows:
        raise SystemExit("manifest produced no episodes; check the word timestamps and mining thresholds")
    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    train, val = frame[~frame["is_val"]], frame[frame["is_val"]]
    if args.per_axis_cap is None:
        train, val = _balance_axes(train, args.seed), _balance_axes(val, args.seed)
    train_path = output.with_name(output.stem + "_train.parquet")
    val_path = output.with_name(output.stem + "_val.parquet")
    train.drop(columns=["is_val"]).to_parquet(train_path)
    val.drop(columns=["is_val"]).to_parquet(val_path)
    print(json.dumps({"train": _split_summary(train), "val": _split_summary(val)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
