# MiniMax H3 FL2VA First-Frame Data

Offline data pipeline for MiniMax H3 FL2VA (text+image to audio-video) RL
training: turn a prompt list into FLUX reference images, pair them into
train/test JSONL, and feed the FL2VA `prepare_data.py` converter.

A ready-made dataset built with this pipeline is published at
https://huggingface.co/datasets/zyfenghit/dancegrpo-t2av

## Pipeline

```
prompts.txt ──► gen_flux_images.py ──► images/{index:06d}.jpg
                     │                        │
                     │                        ▼
                     │              build_fl2va_jsonl.py
                     │                        │
                     ▼                        ▼
              (same index)          train.jsonl / test.jsonl
                                             │
                                             ▼
                              prepare_data.py --frame_mode first
                                             │
                                             ▼
                                train.parquet / test.parquet
```

## Getting the prompt file

`dancegrpo_consist-id.txt` is the filtered ConsisID prompt list released by
DanceGRPO (27,815 prompts, one per line). Download it directly from the
DanceGRPO repository:

```bash
curl -L -o dancegrpo_consist-id.txt \
  https://raw.githubusercontent.com/XueZeyue/DanceGRPO/main/assets/consist-id.txt
```

The prompts originate from
[ConsisID-preview-Data](https://huggingface.co/datasets/BestWishYsh/ConsisID-preview-Data)
captions; DanceGRPO ships the filtered result as-is and does not document the
exact filtering criteria, so downloading the released file is the reproducible
way to get the identical prompt set (verified line-for-line against the copy
used for the published dataset).

## Scripts

### `gen_flux_images.py`

Multi-GPU FLUX batch image generator. Reads one prompt per line, shards the
prompts across ranks (`torchrun`), and writes one JPEG per prompt plus a
per-rank `metadata_rankN.jsonl` (image <-> prompt <-> index mapping).
Deterministic per-prompt seed (`seed + index`), and re-running skips images
that already exist, so interrupted jobs resume safely. Defaults mirror
DanceGRPO's online reference pipeline (400x640, 30 steps, guidance 3.5,
max_sequence_length 512), so prompt and condition image are semantically
aligned by construction.

```bash
torchrun --nproc_per_node=8 examples/diffusionnft_trainer/minimax_h3/gen_flux_images.py \
    --prompt_file dancegrpo_consist-id.txt  # see "Getting the prompt file" \
    --model_path /path/to/FLUX.1-dev \
    --output_dir data/flux_images \
    --height 400 --width 640
```

### `build_fl2va_jsonl.py`

Pairs each prompt with its same-index image, verifies all images exist,
shuffles with a fixed seed, and writes `train.jsonl` / `test.jsonl` with
relative image paths — the input format of `prepare_data.py`:

```bash
python3 examples/diffusionnft_trainer/minimax_h3/build_fl2va_jsonl.py \
    --prompt_file dancegrpo_consist-id.txt  # see "Getting the prompt file" \
    --image_dir data/flux_images/images \
    --output_dir data/flux_images \
    --test_size 128 --seed 42
```

## Reference dataset recipe

- Prompts: 27,815 English video captions from
  [ConsisID-preview-Data](https://huggingface.co/datasets/BestWishYsh/ConsisID-preview-Data),
  as filtered by [DanceGRPO](https://github.com/XueZeyue/DanceGRPO)
  (`assets/consist-id.txt`)
- Images: FLUX.1-dev, 400x640, 30 steps, guidance 3.5, per-index seeds
- Split: seed-42 shuffle -> 27,687 train / 128 test
- Convert: `prepare_data.py --frame_mode first`, then train with
  `rollout.pipeline.task=fl2va` and `frame_indices='[0]'`. The rollout
  pipeline LANCZOS-resizes condition images to the sampling resolution, so
  training at e.g. 288x464 (same ~1:1.61 aspect) works directly.

## License

- Prompts: CC-BY-4.0 (ConsisID-preview-Data)
- Images: generated with FLUX.1-dev (non-commercial license); datasets built
  with this pipeline inherit the non-commercial restriction
