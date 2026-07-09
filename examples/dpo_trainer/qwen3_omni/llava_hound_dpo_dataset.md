# LLaVA-Hound-DPO Dataset for Qwen3-Omni Offline DPO

Last updated: 07/08/2026

This document describes how to obtain, preprocess, and convert the
[LLaVA-Hound-DPO](https://github.com/RifleZhang/LLaVA-Hound-DPO) preference
dataset into the parquet format required by verl-omni's offline MLLM DPO
pipeline. Following VeOmni's Qwen3-Omni multisource data style, one raw
preference pair is converted into three fixed-ratio sources:

| Source | Modality | Purpose | Sampling weight |
|--------|----------|---------|-----------------|
| `llava_hound_dpo/image` | image + text | image understanding DPO, using the first video frame | 0.4 |
| `llava_hound_dpo/text` | text | text-only preference DPO, using the question without media | 0.2 |
| `llava_hound_dpo/video` | video + text | video understanding DPO, using the original clip | 0.4 |

The generated validation split targets 10% of raw samples and is split by
video name, so no source video appears in both train and validation.

---

## 1. Source Dataset

### 1.1 Annotation files

All annotation files are hosted on Hugging Face under
[`ShareGPTVideo/train_video_and_instruction`](https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction).

| File | Rows | Purpose |
|------|------|---------|
| `video_instruction/train/sft/video_240k_caption_15k.jsonl` | ~240 k | SFT supervision (caption + instruction) |
| `video_instruction/train/sft/video_caption_300k.jsonl` | ~300 k | SFT supervision (caption) |
| `video_instruction/train/dpo/sft_dpo_17k.jsonl` | ~17 k | **DPO preference pairs** (chosen + rejected) |

Only `sft_dpo_17k.jsonl` is used for DPO training.

### 1.2 DPO row schema (raw JSONL)

Each row in `sft_dpo_17k.jsonl` has the following structure:

```json
{
  "id": "video_id__0",
  "video": "relative/path/to/video.mp4",
  "conversations": [
    {"from": "human", "value": "<video>\nQuestion text here"}
  ],
  "chosen":   {"from": "gpt", "value": "Preferred answer text"},
  "rejected": {"from": "gpt", "value": "Less preferred answer text"}
}
```

- `video` is a relative path inside the unpacked video archive (see §2.2).
- `conversations` is a single human turn that begins with the `<video>` token.
- `chosen` / `rejected` are single assistant turns.

### 1.3 Video archive

Videos are distributed as 16 gzip-compressed tar shards
(`chunk_0.tar.gz` … `chunk_15.tar.gz`) under
`train_300k/` in the same HuggingFace dataset.

---

## 2. Data Download

You can either run the helper script to download the DPO annotations and unpack
all video shards automatically:

```bash
DATA_DIR=${DATA_DIR:-"$HOME/data/llava_hound_dpo"} \
  bash examples/dpo_trainer/data_process/download_llava_hound_dpo.sh
```

Or follow the manual download steps below.

### 2.1 Annotation files

```bash
DATA_DIR=${DATA_DIR:-"$HOME/data/llava_hound_dpo"}
DPO_DIR="$DATA_DIR/annotations/dpo"
mkdir -p "$DPO_DIR"

wget -c -O "$DPO_DIR/sft_dpo_17k.jsonl" \
  "https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/resolve/main/video_instruction/train/dpo/sft_dpo_17k.jsonl?download=true"
```

### 2.2 Video shards

Download all 16 shards in parallel, then unpack them:

```bash
VIDEO_ZIP_DIR="$DATA_DIR/video_zip"
VIDEO_DIR="$DATA_DIR/videos"
mkdir -p "$VIDEO_ZIP_DIR" "$VIDEO_DIR"

for i in $(seq 0 15); do
  wget -c -O "$VIDEO_ZIP_DIR/chunk_${i}.tar.gz" \
    "https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/resolve/main/train_300k/chunk_${i}.tar.gz?download=true" &
done
wait
echo "All shards downloaded."

for chunk in "$VIDEO_ZIP_DIR"/chunk_*.tar.gz; do
  tar -xzf "$chunk" -C "$VIDEO_DIR" &
done
wait
echo "All shards unpacked."
```

After extraction the video files reside under `$VIDEO_DIR/` and can be
referenced by the relative paths stored in `sft_dpo_17k.jsonl`.

---

## 3. Preprocessing Script

`llava_hound_dpo_multisource.py` converts the raw JSONL + videos into three
parquet sources consumed by verl-omni's offline DPO pipeline. It also writes a
VeOmni-style multisource YAML with `sources`, `names`, and user-specified
`schedule.weights`.

The image source is derived by extracting the first frame of each video with
`ffmpeg`. Install `ffmpeg` before generating the image source.

Choose the source sampling weights for image, text, and video explicitly. The
example below follows VeOmni's Qwen3-Omni ratio of `0.4 / 0.2 / 0.4` and writes
the YAML to `examples/dpo_trainer/qwen3_omni/llava_hound_dpo_multisource.yaml`
by default.

```bash
export DATA_DIR=${DATA_DIR:-"$HOME/data/llava_hound_dpo"}

python3 examples/dpo_trainer/data_process/llava_hound_dpo_multisource.py \
  --dpo_jsonl  "$DATA_DIR/annotations/dpo/sft_dpo_17k.jsonl" \
  --video_dir  "$DATA_DIR/videos" \
  --image_dir  "$DATA_DIR/images" \
  --output_dir "$DATA_DIR/parquet" \
  --source_weights 0.4 0.2 0.4 \
  --multisource_config_path examples/dpo_trainer/qwen3_omni/llava_hound_dpo_multisource.yaml \
  --test_ratio 0.10 \
  --seed 42
```

**Outputs**

| File | Description |
|------|-------------|
| `$DATA_DIR/parquet/image/train.parquet` | Image + text training split |
| `$DATA_DIR/parquet/image/test.parquet` | Image + text validation split |
| `$DATA_DIR/parquet/text/train.parquet` | Text-only training split |
| `$DATA_DIR/parquet/text/test.parquet` | Text-only validation split |
| `$DATA_DIR/parquet/video/train.parquet` | Video + text training split |
| `$DATA_DIR/parquet/video/test.parquet` | Video + text validation split |
| `examples/dpo_trainer/qwen3_omni/llava_hound_dpo_multisource.yaml` | VeOmni-style source list with user-specified weights |
| `$DATA_DIR/images/**/*.jpg` | Extracted image frames |

The generated YAML mirrors the VeOmni data config shape. The weights come from
the `--source_weights IMAGE TEXT VIDEO` arguments:

```yaml
sources:
- /path/to/parquet/image/train.parquet
- /path/to/parquet/text/train.parquet
- /path/to/parquet/video/train.parquet
names:
- LLaVA-Hound-DPO-Image
- LLaVA-Hound-DPO-Text
- LLaVA-Hound-DPO-Video
schedule:
- schedule_type: const
  weights: [0.4, 0.2, 0.4]
val_sources:
- /path/to/parquet/image/test.parquet
- /path/to/parquet/text/test.parquet
- /path/to/parquet/video/test.parquet
```

---

## 4. Offline DPO Parquet Schema

The preprocessing script writes parquet files whose rows follow the **offline MLLM DPO**
schema consumed by
`verl_omni.utils.dataset.offline_mllm_dpo_dataset.offline_mllm_dpo_collate_fn`.

### 4.1 Common columns

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `data_source` | `str` | Yes | `"llava_hound_dpo/image"`, `"llava_hound_dpo/text"`, or `"llava_hound_dpo/video"` |
| `prompt` | `list[dict]` | Yes | Chat-template messages (system + user). Image/video rows embed typed media plus a `{"type": "text", …}` question; text rows use a plain string user turn. |
| `chosen` | `str` | Yes | Preferred assistant response. |
| `rejected` | `str` | Yes | Less preferred assistant response. |
| `win_score` | `float` | No | Optional score metadata for the preferred response. Defaults to `1.0` when absent in the raw record. |
| `lose_score` | `float` | No | Optional score metadata for the less preferred response. Defaults to `0.0` when absent in the raw record. |
| `ability` | `str` | Yes | `"image_qa"`, `"text_qa"`, or `"video_qa"` |
| `reward_model` | `dict` | Yes | `{"style": "preference"}` |
| `extra_info` | `dict` | Yes | `{"split", "index", "sample_id", "question", "source_video", "source_video_name", ...}` |

### 4.2 Image DPO user-turn content

```python
[
    {"type": "image", "image": "/abs/path/to/frame.jpg"},
    {"type": "text",  "text": "Question text here"},
]
```

### 4.3 Text DPO user-turn content

```python
"Question text here"
```

### 4.4 Video DPO user-turn content

```python
[
    {"type": "video", "video": "/abs/path/to/video.mp4"},
    {"type": "text",  "text": "Question text here"},
]
```

### 4.5 Full row example (video)

```python
{
    "data_source": "llava_hound_dpo/video",
    "prompt": [
        {"role": "system",    "content": "You are a helpful assistant."},
        {"role": "user",      "content": [
            {"type": "video", "video": "/data/videos/v_abc123.mp4"},
            {"type": "text",  "text":  "What sport is being played?"},
        ]},
    ],
    "chosen":   "The athlete is playing basketball.",
    "rejected": "I cannot identify the sport from this clip.",
    "win_score": 1.0,
    "lose_score": 0.0,
    "ability":  "video_qa",
    "reward_model": {"style": "preference"},
    "extra_info": {
        "split":      "train",
        "index":      0,
        "sample_id":  "v_abc123__0",
        "question":   "What sport is being played?",
        "source_video":      "clips/v_abc123.mp4",
        "source_video_name": "v_abc123.mp4",
        "video_path":        "/data/videos/clips/v_abc123.mp4",
    },
}
```

---

## 5. Dataset Class

Data loading reuses the upstream `RLHFDataset` directly; no custom dataset
class is required. `RLHFDataset.__getitem__` returns all parquet columns that
are not recognised as media-placeholder fields, so `chosen` and `rejected` pass
through alongside `raw_prompt`.  The structured multimodal content in `prompt`
(list-of-dicts with `{"type":"image",…}` or `{"type":"video",…}`) is handled
transparently by `RLHFDataset._build_messages` without any modification.

The only custom piece is the collate function
`verl_omni.utils.dataset.offline_mllm_dpo_dataset.offline_mllm_dpo_collate_fn`,
which expands each `(prompt, chosen, rejected)` row into two adjacent samples
tagged `is_chosen: True` and `is_chosen: False`.

Configure offline DPO training with `verl_omni.trainer.main_omni` and the
Qwen3-Omni thinker DPO recipe:

```bash
python3 -m verl_omni.trainer.main_omni \
  --config-path=examples/dpo_trainer/qwen3_omni/config \
  --config-name=qwen3_omni_thinker_offline_mllm_dpo \
algorithm.sample_source=offline
actor_rollout_ref.actor.policy_loss.loss_mode=dpo
data.custom_cls.path=pkg://verl_omni.utils.dataset.offline_mllm_dpo_dataset
data.custom_cls.collate_fn=offline_mllm_dpo_collate_fn
```

For multisource training, pass all three train parquet files and all three val
parquet files. `RLHFDataset` concatenates them and the DataLoader shuffle
produces mixed batches:

```bash
data.train_files=[/path/to/image/train.parquet,/path/to/text/train.parquet,/path/to/video/train.parquet]
data.val_files=[/path/to/image/test.parquet,/path/to/text/test.parquet,/path/to/video/test.parquet]
```

---

## 6. Split and Sampling Details

The preprocessing script first groups raw rows by `Path(video).name`, then
assigns whole video-name groups to the validation split until it reaches the
target `--test_ratio` of rows. The image, text, and video sources reuse this
same split assignment, so derived samples from the same source video cannot
leak across train and validation.

The `llava_hound_dpo_multisource.yaml` file records the intended user-selected
multisource schedule, matching the VeOmni config style used by
[`tulu_sharegpt4v_llavavideo.yaml`](https://github.com/ByteDance-Seed/VeOmni/blob/main/configs/multimodal/data/tulu_sharegpt4v_llavavideo.yaml):

```yaml
schedule:
- schedule_type: const
  weights: [0.4, 0.2, 0.4]
```

Offline DPO does not require a reward model. The parquet `reward_model` column
is only a pass-through metadata field that marks rows as preference pairs.

---

## 7. File Layout Summary

```
$DATA_DIR/
├── annotations/
│   └── dpo/
│       └── sft_dpo_17k.jsonl          # raw DPO annotation file
├── video_zip/
│   ├── chunk_0.tar.gz                 # downloaded video shards
│   └── …
├── videos/                            # unpacked MP4 files
│   └── <relative_paths_from_jsonl>/
├── images/                            # extracted first-frame JPG files
│   └── <relative_paths_from_jsonl>.jpg
└── parquet/
    ├── image/
    │   ├── train.parquet
    │   └── test.parquet
    ├── text/
    │   ├── train.parquet
    │   └── test.parquet
    ├── video/
    │   ├── train.parquet
    │   └── test.parquet
```

The default multisource YAML path is outside `$DATA_DIR`:

```text
examples/dpo_trainer/qwen3_omni/llava_hound_dpo_multisource.yaml
```
