# FLUX.1-dev DanceGRPO

This directory contains an Ascend NPU-only launcher for full-parameter
FLUX.1-dev DanceGRPO training with HPSv2 reward. Its default configuration uses:
720x720 images, 16 denoising steps, rollout `guidance_scale=3.5`, actor
`guidance_scale=1.0`, 0.3 DanceSDE noise, and a random 60% subset after dropping
the final transition.

FLUX.1-dev uses a distilled guidance embedding. This adapter does not run a
negative-prompt CFG branch and deliberately has no `true_cfg_scale` option.
The rollout and actor guidance values are intentionally different; they keep
their existing `guidance_scale` config names because their locations already
distinguish sampling from actor log-probability recomputation.

## Prepare prompts

Prompt files are not included. Supply two UTF-8 text files with one prompt per
line so the train/test boundary is explicit and reviewable. Public sources you
can evaluate include the [HPSv2 project and HPD benchmark](https://github.com/tgxs002/HPSv2)
and the [DanceGRPO reference implementation](https://github.com/XueZeyue/DanceGRPO).

Convert the files to the same chat-style parquet schema used by the other
DanceGRPO examples:

```bash
python3 examples/dancegrpo_trainer/flux1/dataprocess/prepare_prompts.py \
  --train-path /path/to/train.txt \
  --test-path /path/to/test.txt \
  --output-dir "${WORKSPACE:-$HOME}/data/hpsv2"
```

When `--output-dir` is omitted, the processor writes `train.parquet` and
`test.parquet` to `~/data/hpsv2`, matching the launcher's default paths. The
processor never downloads, embeds, randomly splits, or copies prompt text into
this repository.

## Prepare FLUX.1-dev and HPSv2

Install the HPSv2 dependency on top of the standard verl-omni environment. The
following version has been validated with this recipe:

```bash
pip install open_clip_torch==3.3.0
```

Prepare a local diffusers-format
[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) checkpoint and
download these HPSv2.1 files:

- [`HPS_v2.1_compressed.pt`](https://huggingface.co/xswu/HPSv2/tree/main)
- [`open_clip_pytorch_model.bin`](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main)

Only use checkpoints you trust: the HPSv2 loader uses PyTorch checkpoint
deserialization for compatibility with the published weights.

## Run with HPSv2

From the repository root:

```bash
export MODEL_NAME=/path/to/FLUX.1-dev
export HPSV2_PRETRAINED_PATH=/path/to/open_clip_pytorch_model.bin
export CUSTOM_REWARD_MODEL_PATH=/path/to/HPS_v2.1_compressed.pt

bash examples/dancegrpo_trainer/flux1/run_flux1_dev_dancegrpo_fullparam.sh
```

## Validate with one training step

Use the same model, reward checkpoints, and prepared parquet files, but limit
the launcher to one step:

```bash
TOTAL_TRAINING_STEPS=1 \
EXPERIMENT_NAME=flux1_dev_fullparam_hpsv2_smoke \
bash examples/dancegrpo_trainer/flux1/run_flux1_dev_dancegrpo_fullparam.sh
```

A successful smoke run reaches `training/global_step:1` and reports
`critic/hpsv2_raw/mean`, `rollout_corr/logprob_abs_diff_mean`, `actor/loss`,
and `timing_s/step`. This exercises rollout generation, HPSv2 scoring, Actor
log-probability recomputation and update, and weight synchronization.

By default, the launcher reads `$WORKSPACE/data/hpsv2/train.parquet` and
`$WORKSPACE/data/hpsv2/test.parquet`, where `WORKSPACE` defaults to `$HOME`.
Set `TRAIN_FILES_PATH` and `VAL_FILES_PATH` to override these locations.

HPSv2 defaults to two CPU-resident reward workers (`REWARD_NUM_WORKERS=2` and
`REWARD_DEVICE=cpu`). The OpenCLIP model and both checkpoints are loaded on
CPU, so reward scoring does not consume rollout NPU memory. Synchronous scorer
calls run through the reward manager's executor, and HPSv2 uses autocast during
inference. To opt into NPU
scoring, set `REWARD_DEVICE=npu:0` explicitly. This is not the default because
verl RewardLoopWorkers do not reserve Ray accelerator resources; an explicit
NPU scorer can therefore contend with actor and rollout allocations.

The launcher defaults to eight NPUs and full-parameter training
(`model.lora_rank=0`). Hardware and batch settings can be overridden with
`ASCEND_RT_VISIBLE_DEVICES`, `NUM_GPUS`, `ROLLOUT_TP`, `TRAIN_BATCH_SIZE`,
`ROLLOUT_N`, `REWARD_NUM_WORKERS`, `TRAIN_FILES_PATH`, and `VAL_FILES_PATH`.
