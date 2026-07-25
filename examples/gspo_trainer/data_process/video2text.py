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
"""Convert NExT-QA (MC) → verl RL parquet (video → text).

NExT-QA MC sample (ModelScope AI-ModelScope/NExTQA, MC/test-*.parquet):
  video    : <int video id, e.g. 2574374895>
  question : "what did the baby do after ..."
  answer   : "2"   (0-4 index into a0..a4)
  a0..a4   : option texts (5 options; trailing ones may be empty)
  qid, type: metadata

输出 verl RL parquet:
  data_source, prompt[system+user(<video>+Question+Options)], videos[绝对路径],
  reward_model.ground_truth="<answer>X</answer>", extra_info

answer 数字 0-4 → 字母 A-E, 复用 choice_reward 精确匹配 (<answer>X</answer> search)。
SYSTEM_PROMPT 要求模型只输出字母 <answer>X</answer> (同 AVQA/audio2text)。
视频媒体外置 (videos/{video_id}.mp4), 所有节点同路径挂载。
按 video_id 分组切分, 避免 train/val 视频重叠泄漏。
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

DATA_SOURCE = "nextqa_video_qa"
ABILITY = "video_qa"
DATASET_NAME = "NExTQA-MC"

SYSTEM_PROMPT = (
    "Please think about this question as if you were a human pondering deeply, carefully considering the "
    "video information before answering, engaging in an internal dialogue using expressions such as "
    "let me think, wait, hmm, oh I see, or let's break it down, including self-reflection or verification in the "
    "reasoning process, providing the detailed reasoning between the <think> </think> tags, and finally giving "
    "only the single option letter (e.g., A, B, C, D, E) as the final answer within the <answer> </answer> tags."
)

NUM_OPTIONS = 5  # NExT-QA MC has a0..a4


def _video_id_str(vid):
    try:
        return str(int(vid))
    except (ValueError, TypeError):
        s = str(vid).strip()
        return s


def build_rl_row(record, videos_dir, index):
    """Convert one NExT-QA MC record → (row, drop_reason)."""
    vid_str = _video_id_str(record.get("video"))
    if not vid_str or vid_str == "0":
        return None, "no_video_id"

    question = str(record.get("question") or "").strip()
    if not question:
        return None, "empty_question"

    # Collect non-empty options a0..a4 (preserve order).
    options = []
    for i in range(NUM_OPTIONS):
        txt = str(record.get(f"a{i}") or "").strip()
        if txt:
            options.append(txt)
    if len(options) < 2:
        return None, "too_few_options"

    # answer (0-4 index) → letter. NOTE: answer=0 is valid (option A); do NOT
    # coalesce with `or ""` (0 is a falsy int and would be mis-dropped as bad_answer).
    try:
        ans_idx = int(record.get("answer"))
    except (ValueError, TypeError):
        return None, "bad_answer"
    if ans_idx < 0 or ans_idx >= len(options):
        # Correct option is one of the dropped (empty) trailing options → unusable.
        return None, "answer_out_of_range"
    letter = chr(ord("A") + ans_idx)
    gt = f"<answer>{letter}</answer>"

    # Video file: NExT-QA names clips {video_id}.mp4.
    video_path = videos_dir / f"{vid_str}.mp4"
    if not video_path.is_file():
        return None, "missing_video_file"

    option_lines = [f"{chr(ord('A') + i)}. {t}" for i, t in enumerate(options)]
    user_content = f"<video>{question}\nOptions:\n" + "\n".join(option_lines)

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    option_map = {chr(ord("A") + i): t for i, t in enumerate(options)}
    # video item as dict so qwen_vl_utils.smart_nframes reads fps/max_frames (verl does
    # NOT forward mm_processor_kwargs to process_vision_info). max_frames=32 caps video
    # tokens ~8-10k so prompt fits max_model_len=16384; fps must match mm_processor_kwargs.fps
    # so processor's video_second_per_grid (=temporal_patch_size/fps) stays consistent.
    return {
        "data_source": DATA_SOURCE,
        "prompt": prompt,
        "videos": [{"video": str(video_path), "fps": 2.0, "max_frames": 32}],
        "ability": ABILITY,
        "reward_model": {"style": "rule", "ground_truth": gt},
        "extra_info": {
            "split": "mc",
            "index": index,
            "dataset": DATASET_NAME,
            "qid": str(record.get("qid") or ""),
            "type": str(record.get("type") or ""),
            "raw_answer": str(record.get("answer")),
            "answer_letter": letter,
            "options": json.dumps(option_map, ensure_ascii=False),
            "video_id": vid_str,
        },
    }, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="MC/test-*.parquet")
    ap.add_argument("--videos_dir", required=True, help="dir of {video_id}.mp4")
    ap.add_argument("--output_dir", required=True, help="output dir for train/validation.parquet")
    ap.add_argument("--val_size", type=int, default=200, help="approx val sample count (grouped by video)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    videos_dir = Path(args.videos_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)
    print(f"read {len(df)} rows from {in_path.name}, cols: {list(df.columns)}")

    rows = []
    dropped = Counter()
    for i, r in df.iterrows():
        row, reason = build_rl_row(r, videos_dir, i)
        if row is None:
            dropped[reason] += 1
        else:
            rows.append(row)
    print(f"kept {len(rows)} / dropped {dict(dropped)}")

    if not rows:
        raise ValueError(f"no valid rows; dropped={dict(dropped)}")

    # Group by video_id so train/val never share a video (avoids leakage).
    by_video = defaultdict(list)
    for row in rows:
        by_video[row["extra_info"]["video_id"]].append(row)
    video_ids = sorted(by_video.keys())
    random.seed(args.seed)
    random.shuffle(video_ids)

    # Accumulate whole videos into val until we reach ~val_size samples.
    val_vids = []
    val_cnt = 0
    for vid in video_ids:
        if val_cnt >= args.val_size:
            break
        val_vids.append(vid)
        val_cnt += len(by_video[vid])
    val_vid_set = set(val_vids)
    val = [r for vid in val_vids for r in by_video[vid]]
    train = [r for vid in video_ids if vid not in val_vid_set for r in by_video[vid]]

    pd.DataFrame(train).to_parquet(out_dir / "train.parquet", index=False, engine="pyarrow")
    pd.DataFrame(val).to_parquet(out_dir / "validation.parquet", index=False, engine="pyarrow")

    train_ans = Counter(r["extra_info"]["answer_letter"] for r in train)
    val_ans = Counter(r["extra_info"]["answer_letter"] for r in val)
    print(f"train {len(train)} (videos {len(video_ids) - len(val_vids)}) / val {len(val)} (videos {len(val_vids)})")
    print(f"train answer dist: {dict(sorted(train_ans.items()))}")
    print(f"val answer dist: {dict(sorted(val_ans.items()))}")
    print(f"sample prompt[user]: {train[0]['prompt'][1]['content'][:200]}")
    print(f"sample ground_truth: {train[0]['reward_model']['ground_truth']}")
    print(f"sample video: {train[0]['videos'][0]}")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
