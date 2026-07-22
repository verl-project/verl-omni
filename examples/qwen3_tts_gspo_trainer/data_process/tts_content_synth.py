# Copyright 2026 Gulp AI Inc and/or its affiliates
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
"""Preprocess conversation transcripts to verl parquet for Qwen3-TTS RL training.

Source: a JSONL file with one conversation per line, each carrying a ``messages`` list
(``[{"role": ..., "content": ...}, ...]``). We take the text of **every ``assistant`` message**
as a TTS prompt (the single-turn line to be spoken). The conversations have no reference audio,
so every line is cloned from ONE fixed voice (``--ref_audio``) to keep speaker-similarity usable
in the reward.

Exact-duplicate lines (e.g. a greeting that repeats thousands of times) are collapsed by default
(``--dedup``); pass ``--no-dedup`` to keep all lines.

Each parquet row:
    data_source         = "tts"
    prompt              = [{"role": "user", "content": <text>}]   # the line to speak
    reward_model        = {"style": "model", "ground_truth": <text>}  # CER target
    extra_info          = {"id", "text", "ref_audio", "category", "split", "index"}
"""

import argparse
import json
import os

import pandas as pd


def _iter_assistant_lines(path):
    """Yield (conversation_id, turn_index, text) for every non-empty assistant message."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            conv = json.loads(line)
            cid = conv.get("conversation_id") or "conv"
            n = 0
            for msg in conv.get("messages", []):
                if msg.get("role") != "assistant":
                    continue
                text = (msg.get("content") or "").strip()
                if not text:
                    continue
                yield cid, n, text
                n += 1


def _make_lid(lang):
    """Return a predicate(text) -> bool keeping only ``lang`` lines (None -> keep all).

    langdetect with a pinned seed (it is stochastic by default). Detection failures
    (raised on very short or symbol-only lines) KEEP the row; those are almost always
    English interjections and ``--min_chars`` already governs the tiny ones.
    """
    if not lang:
        return lambda text: True
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 0

    def _keep(text):
        try:
            return detect(text) == lang
        except LangDetectException:
            return True

    return _keep


def build_rows(path, ref_audio, dedup, min_chars, lid=None):
    rows = []
    seen = set()
    lid_keep = _make_lid(lid)
    n_lid_dropped = 0
    for cid, turn, text in _iter_assistant_lines(path):
        if len(text) < min_chars:
            continue
        if dedup:
            if text in seen:
                continue
            seen.add(text)
        if not lid_keep(text):
            n_lid_dropped += 1
            continue
        rows.append(
            {
                "data_source": "tts",
                "prompt": [{"role": "user", "content": text}],
                "ability": "tts",
                "reward_model": {"style": "model", "ground_truth": text},
                "extra_info": {
                    "id": f"{cid}#{turn}",
                    "text": text,
                    "ref_audio": ref_audio,
                    "category": None,
                },
            }
        )
    if lid:
        print(f"LID filter ({lid}): dropped {n_lid_dropped} lines, kept {len(rows)}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Conversations JSONL: one {'messages': [{'role', 'content'}, ...]} per line."
    )
    parser.add_argument(
        "--output_dir", default="~/data/tts_voice_synth", help="Directory to save train.parquet / test.parquet"
    )
    parser.add_argument(
        "--ref_audio", required=True, help="Fixed reference clip cloned for EVERY line (speaker-sim anchor)."
    )
    parser.add_argument(
        "--dedup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collapse exact-duplicate lines (default: on).",
    )
    parser.add_argument("--min_chars", type=int, default=8, help="Drop lines shorter than this many characters.")
    parser.add_argument(
        "--lid",
        default=None,
        help="Keep only lines langdetect identifies as this language (e.g. 'en'). "
        "Off by default; detection failures keep the row.",
    )
    parser.add_argument("--val_size", type=int, default=256, help="Number of held-out lines for the test split.")
    parser.add_argument("--max_train", type=int, default=None, help="Optional cap on the number of training rows.")
    args = parser.parse_args()

    rows = build_rows(args.input, args.ref_audio, args.dedup, args.min_chars, lid=args.lid)
    # Stable, deterministic ordering then a fixed head/tail split (no RNG, reproducible).
    rows.sort(key=lambda r: r["extra_info"]["id"])
    val_rows = rows[: args.val_size]
    train_rows = rows[args.val_size :]
    if args.max_train is not None:
        train_rows = train_rows[: args.max_train]

    for i, r in enumerate(train_rows):
        r["extra_info"]["split"] = "train"
        r["extra_info"]["index"] = i
    for i, r in enumerate(val_rows):
        r["extra_info"]["split"] = "test"
        r["extra_info"]["index"] = i

    out_dir = os.path.expanduser(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(os.path.join(out_dir, "train.parquet"))
    pd.DataFrame(val_rows).to_parquet(os.path.join(out_dir, "test.parquet"))
    print(f"wrote {len(train_rows)} train + {len(val_rows)} test rows to {out_dir} (dedup={args.dedup})")
