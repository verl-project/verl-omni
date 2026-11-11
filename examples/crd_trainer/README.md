# CRD Trainer

Last updated: 10/25/2026.

This example post-trains `stabilityai/stable-diffusion-3.5-medium` with
[CRD](../../docs/algo/crd.md) on OCR, using LoRA, the V1 synchronous trainer,
`vllm-omni` rollout and `Qwen/Qwen2.5-VL-3B-Instruct` as the reward model.

## Installation

Follow the [installation guide](../../docs/start/install.md) and accept the
SD3.5-Medium access terms on Hugging Face. Install OCR dependencies:

```bash
uv pip install -e ".[ocr]"
```

The recipe defaults to two GPUs: one actor/rollout GPU and one OCR reward GPU.

## Prepare the dataset

Get `train.txt` and `test.txt` from the
[Flow-GRPO OCR dataset](https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr),
place them under `$OCR_WORKSPACE/data/ocr`, and convert them:

```bash
export OCR_WORKSPACE=/path/to/workspace
mkdir -p "$OCR_WORKSPACE/data/ocr/sd3"
python3 examples/flowgrpo_trainer/data_process/sd3_ocr.py \
  --input_dir "$OCR_WORKSPACE/data/ocr" \
  --output_dir "$OCR_WORKSPACE/data/ocr/sd3"
```

## Run

Run from the repository root:

```bash
bash examples/crd_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh
```

Defaults are 100 iterations at 512×512, four prompts per actor GPU and six images
per prompt, with validation on 32 prompts before training and every 10 iterations.
Validation uses the `old` sampling EMA adapter.

Use `NUM_GPUS_ACTOR_ROLLOUT`, `NUM_GPUS_REWARD`, `GROUP_SIZE`, `IMAGE_RESOLUTION`,
`TOTAL_TRAINING_STEPS` and `OUTPUT_DIR` to override the defaults. Hydra overrides
are accepted last. Whole prompt groups must fit in each actor micro batch.
The recipe starts fresh and retains the latest two checkpoints. Resume with
`trainer.resume_mode=resume_path` and
`trainer.resume_from_path=/path/to/global_step_N`.

For a DGPO comparison with the same prompts, reward, initialization and training
budget:

```bash
bash examples/crd_trainer/sd35/run_dgpo_comparison.sh
```

The scripts use separate default output directories. Both recipes sample at CFG 1
with 10 steps and evaluate at CFG 4.5 with 40 steps. The comparison recipe uses
CRD's training settings with DGPO's loss and sampling EMA schedule.

## Tested configuration

- Two NVIDIA A100-SXM4-80GB GPUs, with one actor/rollout GPU and one reward GPU.
- Python 3.12, PyTorch `2.13.0+cu130`, vLLM `0.30.0`, repository-pinned
  vLLM-Omni and verl, diffusers `0.40.0`, native/SDPA attention.

CRD and the matched DGPO comparison each completed 100 training iterations and
400 optimizer updates at 512×512, using four prompts and six images per prompt.
Both runs evaluated the same 128 fixed held-out prompts before training and every
10 iterations, with identical generation seeds and settings.

| Metric | CRD | DGPO |
| --- | --- | --- |
| Initial held-out OCR reward | 0.7671 | 0.7671 |
| Final held-out OCR reward | 0.9267 | 0.8208 |
| Best observed held-out OCR reward | 0.9601 (iteration 90) | 0.8400 (iteration 90) |
| Mean training iteration, excluding evaluation | 37.2 s | 35.3 s |
| Training images per second, excluding evaluation | 0.646 | 0.679 |
| Sampled actor/rollout GPU memory peak | 50.4 GiB | 50.3 GiB |
| Sampled reward GPU memory peak | 57.1 GiB | 57.1 GiB |
