# DGPO Trainer

Last updated: 10/10/2026

This example post-trains `stabilityai/stable-diffusion-3.5-medium` with
[DGPO](../../docs/algo/dgpo.md) on the OCR task, using `vllm-omni` rollout and
`Qwen/Qwen2.5-VL-3B-Instruct` as the OCR reward model.

DGPO is an online direct-preference algorithm: rollouts are deterministic, the
actor trains from the final clean latents with a group-level preference weight,
and an `old` LoRA adapter serves as the rollout policy while the `default`
adapter is updated.

## Installation

Follow the [installation guide](../../docs/start/install.md), then install the
OCR reward dependency:

```bash
uv pip install -e ".[ocr]"
```

The script defaults to one node with 2 actor/rollout GPUs and 1 reward GPU
(`NUM_GPUS_ACTOR_ROLLOUT`, `NUM_GPUS_REWARD`).

## Prepare the dataset

Get the raw OCR prompts from the original Flow-GRPO repository
(https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr), place them under
`$WORKSPACE/data/ocr` (`WORKSPACE` defaults to `$HOME`), and convert them:

```bash
python3 examples/flowgrpo_trainer/data_process/sd3_ocr.py \
  --input_dir $WORKSPACE/data/ocr \
  --output_dir $WORKSPACE/data/ocr/sd3
```

## Run

```bash
bash examples/dgpo_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh
```

Every rollout group must fit in one actor micro batch on one rank, so keep
`actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` a multiple of
`actor_rollout_ref.rollout.n` and `actor_rollout_ref.actor.ppo_mini_batch_size`
divisible by the number of actor GPUs. The trainer checks both at start.

## Tested configuration

- 2x NVIDIA A800 80GB PCIe (no NVLink) with an Intel Xeon Gold 6336Y, Linux, BF16
- Python 3.12, vllm 0.30.0, vllm-omni 0.30.1.dev29+g12e928001, torch 2.13.0+cu132,
  diffusers 0.40.0, transformers 5.14.1
- `NUM_GPUS_ACTOR_ROLLOUT=1 NUM_GPUS_REWARD=1`, attention `native` with rollout
  `TORCH_SDPA`, and `actor_rollout_ref.rollout.free_cache_engine=False` (keeps the
  rollout pipeline resident instead of offloading it to pinned host memory)
- A 20-step run of `sd35/run_sd35_medium_ocr_lora_v1.sh` in this setup, validating
  on 128 test prompts at steps 0, 10 and 20, took 62.6 s per training step; the
  OCR validation reward went from 0.721 at step 0 to 0.807 at step 10 and 0.840 at
  step 20
