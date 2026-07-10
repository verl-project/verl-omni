# Omni-Preference Dataset for verl-omni Offline MLLM DPO

Last updated: 07/09/2026

This document describes how to download, prepare, and convert the
[Omni-Preference](https://huggingface.co/datasets/Omni-RRM/Omni-Preference)
dataset into the parquet format required by verl-omni's offline MLLM DPO pipeline.

Omni-Preference is the rubric-grounded preference dataset introduced in
**Omni-RRM: Advancing Omni Reward Modeling via Automatic Rubric-Grounded Preference
Synthesis** ([anonymous repo](https://anonymous.4open.science/r/Omni-RRM-CC08/readme.md),
[Hugging Face dataset](https://huggingface.co/datasets/Omni-RRM/Omni-Preference)).
It contains ~41K retained preference pairs across **image**, **video**, and **audio**
modalities, with teacher-reconciled scores and verdicts.

For verl-omni DPO training we convert `final_rl_data.jsonl` into multimodal
parquet files: each row is a preference pair where the prompt is **media + question**
and `chosen` / `rejected` are the two candidate answers.

---

## 1. Source Dataset Overview

### 1.1 Modality breakdown

| Modality | Source datasets | Retained samples (paper) | `final_rl_data.jsonl` rows (local clone) |
|----------|-----------------|--------------------------|------------------------------------------|
| Image | RLAIF-V | ~17.0K | 6,637 |
| Video | ActivityNet, Charades, Ego4D, NextQA, YouCook2 | ~12.2K | 4,327 |
| Audio | Clotho-AQA | ~11.8K | 5,707 |
| **Total** | — | **~41.0K** | **~16.7K raw RL rows** |

Each modality also ships an SFT JSONL (`final_sft_data.jsonl`) for reward-model
SFT / GRPO in the original Omni-RRM repo. The verl-omni DPO converter reads
**`final_rl_data.jsonl` only**.

### 1.2 Preference construction (summary)

1. **Candidate pair generation** — for each multimodal input, a stronger and a
   weaker model produce Candidate A and Candidate B.
2. **Rubric-grounded teacher reconciliation** — heterogeneous teachers assign
   scalar scores, a categorical verdict, and five-criterion rationales.
3. **Consensus filtering** — a pair is kept only when both teachers agree on a
   non-tie verdict consistent with score ranking.

Evaluation uses five shared criteria: fluency/coherence, relevance,
accuracy/completeness, reasoning quality, and safety/ethical alignment.

---

## 2. Download

Omni-Preference is hosted on Hugging Face and uses Git LFS for media files.

### 2.1 Clone the full dataset

```bash
git lfs install
git clone https://huggingface.co/datasets/Omni-RRM/Omni-Preference
cd Omni-Preference
```

> **Note:** The full clone includes images, videos, and audio. Download size is
> large; ensure sufficient disk space and a working Git LFS setup.

### 2.2 Partial / resume download

If the clone is interrupted, resume from inside the repo:

```bash
cd Omni-Preference
git lfs pull
```

---

## 3. Folder Structure

After a full download, the dataset root looks like this:

```text
Omni-Preference/
├── dataset_jsonl/
│   ├── image/
│   │   ├── final_rl_data.jsonl      # RL / DPO preference pairs (image)
│   │   └── final_sft_data.jsonl       # SFT targets for Omni-RRM training
│   ├── video/
│   │   ├── final_rl_data.jsonl
│   │   └── final_sft_data.jsonl
│   └── audio/
│       ├── final_rl_data.jsonl
│       └── final_sft_data.jsonl
├── rlaif-v-dataset/                   # Image media root
│   ├── llava1.5_raw_images/
│   ├── coco2017/
│   ├── gqa/
│   └── ...
├── video-dataset/                     # Video media root
│   └── academic_source/
│       ├── Charades/
│       ├── activitynet/
│       ├── ego4d/
│       ├── NextQA/
│       └── youcook2/
└── audio_files/                       # Audio media root
    └── audio_files/                   # Flat WAV files (by basename)

```

### 3.1 Path mapping used by verl-omni

JSONL media paths use a `/data/...` prefix. The converter resolves them as follows:

| Modality | JSONL path example | Resolved on disk |
|----------|-------------------|------------------|
| Image | `/data/rlaif-v-dataset/llava1.5_raw_images/00013/000137479.jpg` | `{dataset_root}/rlaif-v-dataset/llava1.5_raw_images/...` |
| Video | `/data/academic_source/Charades/2D98B.mp4` | `{dataset_root}/video-dataset/academic_source/Charades/...` |
| Audio | `/data/audio/Lake_Wars_extract.wav` | `{dataset_root}/audio_files/**/{basename}.wav` |

Rows whose media file cannot be resolved locally are **skipped** (logged as
`skipped_missing_media`).

## 4. Raw JSONL Schema

### 4.1 RL row (`final_rl_data.jsonl`) — used for DPO

Each line is one preference-evaluation record. Example (video):

```json
{
  "videos": ["/data/academic_source/Charades/2D98B.mp4"],
  "messages": [
    {
      "role": "user",
      "content": "... ### Context\nVideo file: ...\nQuestion: ...\nCandidate A: ...\nCandidate B: ..."
    }
  ],
  "solution": "{\"score_A\": 6, \"score_B\": 8, \"better\": \"B\", \"reasoning\": \"...\", \"final_verdict\": \"<answer>[[B]]</answer>\"}"
}
```

Image rows use `"images": [...]` and `Image file:` in the Context block.
Audio rows use `"audios": [...]` and `Audio file:` in the Context block.

The converter parses the `### Context` section:

```text
{Image|Video|Audio} file: <media_path>
Question: <question>
Candidate A: <answer_a>
Candidate B: <answer_b>
```

And the `solution` JSON:

| Field | Description |
|-------|-------------|
| `score_A` / `score_B` | Teacher scores (0–10) for each candidate |
| `better` | `"A"`, `"B"`, or `"equal"` |
| `reasoning` | Rubric-grounded comparative rationale |
| `final_verdict` | Parsed verdict tag, e.g. `<answer>[[B]]</answer>` |

**DPO conversion rules:**

- `better == "equal"` → row is **skipped** (no clear preference).
- `better == "A"` → `chosen = Candidate A`, `rejected = Candidate B`.
- `better == "B"` → `chosen = Candidate B`, `rejected = Candidate A`.
- `win_score` / `lose_score` in parquet = scores of the chosen / rejected candidate.


## 5. Preprocessing for verl-omni Offline DPO

Script:
[`examples/dpo_trainer/data_process/omni_preference_dpo_multisource.py`](../data_process/omni_preference_dpo_multisource.py)

### 5.1 One-shot: all three modalities

```bash
export DATASET_ROOT=${DATASET_ROOT:-"$HOME/Omni-Preference"}
export OUTPUT_DIR=${OUTPUT_DIR:-"$DATASET_ROOT/parquet_dpo"}

python examples/dpo_trainer/data_process/omni_preference_dpo_multisource.py \
  --dataset_root "$DATASET_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --modalities image video audio \
  --test_ratio 0.10 \
  --seed 42
```


### 5.2 CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset_root` | *(required)* | Omni-Preference repo root |
| `--output_dir` | `~/data/omni_preference_dpo/parquet` | Parquet output directory |
| `--modalities` | `image video audio` | Modalities to convert |
| `--test_ratio` | `0.10` | Target validation fraction |
| `--seed` | `42` | Random seed for train/test split |
| `--max_samples` | `-1` | Truncate per modality (debug) |

### 5.3 Train / test split

- Split unit: **media file name** (same image / video / audio never appears in
  both train and test).
- Target test size: `round(total_parsed_rows × test_ratio)`.
- Because grouping is by media file, the actual row ratio may differ slightly
  from `--test_ratio`.
- Media-missing rows are dropped **after** split assignment, which can further
  skew the final train/test ratio if many files are absent locally.

### 5.4 Outputs

| File | Description |
|------|-------------|
| `$OUTPUT_DIR/image/train.parquet` | Image + question DPO training split |
| `$OUTPUT_DIR/image/test.parquet` | Image + question DPO validation split |
| `$OUTPUT_DIR/video/train.parquet` | Video + question DPO training split |
| `$OUTPUT_DIR/video/test.parquet` | Video + question DPO validation split |
| `$OUTPUT_DIR/audio/train.parquet` | Audio + question DPO training split |
| `$OUTPUT_DIR/audio/test.parquet` | Audio + question DPO validation split |

---

## 6. Offline DPO Parquet Schema

Parquet rows follow the preference schema consumed by the Qwen3-Omni VeOmni
VLM DPO smoke task.

### 6.1 Columns

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `data_source` | `str` | Yes | `"omni_preference/image"`, `".../video"`, or `".../audio"` |
| `prompt` | `list[dict]` | Yes | Chat messages: system + user (typed media + question text) |
| `chosen` | `str` | Yes | Preferred candidate answer |
| `rejected` | `str` | Yes | Less preferred candidate answer |
| `win_score` | `float` | No | Score of the chosen candidate (from `score_A` / `score_B`) |
| `lose_score` | `float` | No | Score of the rejected candidate |
| `ability` | `str` | Yes | `"image_qa"`, `"video_qa"`, or `"audio_qa"` |
| `reward_model` | `dict` | Yes | `{"style": "preference"}` |
| `extra_info` | `dict` | Yes | Metadata: `split`, `question`, `source_media`, `better`, `{modality}_path`, … |

---

## 7. Training Configuration (VeOmni)

Point the VeOmni VLM DPO data config at the generated parquet files:

```yaml
sources:
  - /path/to/Omni-Preference/parquet_dpo/image/train.parquet
  - /path/to/Omni-Preference/parquet_dpo/video/train.parquet
  - /path/to/Omni-Preference/parquet_dpo/audio/train.parquet
names:
  - Omni-Preference-Image
  - Omni-Preference-Video
  - Omni-Preference-Audio
schedule:
  - schedule_type: const
    weights: [0.34, 0.33, 0.33]
```

To train on a single modality, list only that modality's parquet path and set
the corresponding source name/weight.

See
[`tests/special_e2e/run_qwen3_omni_thinker_only_offline_mllm_dpo_lora_smoke.sh`](../../../tests/special_e2e/run_qwen3_omni_thinker_only_offline_mllm_dpo_lora_smoke.sh)
for an example launch script.

---

## 8. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `skipped_missing_media` is large | Media not fully downloaded | Run `git lfs pull` inside `Omni-Preference` |
| `skipped_parse > 0` | Context block doesn't match expected template | Inspect failing rows; check for format drift |
| `skipped_equal` | Teacher verdict is tie | Expected; DPO requires a strict preference |
| Test set much smaller than `--test_ratio` | Many test media files missing locally | Complete media download or lower `--test_ratio` |
| Audio paths not found | WAVs stored flat under `audio_files/audio_files/` | Converter matches by **basename**; ensure LFS pull completed |

---

## 9. References

- Omni-RRM paper repo (anonymous): https://anonymous.4open.science/r/Omni-RRM-CC08/readme.md
- Omni-Preference on Hugging Face: https://huggingface.co/datasets/Omni-RRM/Omni-Preference
