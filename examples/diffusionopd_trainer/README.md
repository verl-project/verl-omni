# Diffusion On-Policy Distillation Trainer

Last updated: 08/31/2026

This example distills an OCR-tuned `SD3.5-Medium` teacher into a fresh `SD3.5-Medium` student. The student generates images with its own policy, a frozen teacher scores every denoising step of those trajectories, and the student minimizes the KL between its transition and the teacher's (`distill_kl`). The OCR reward is only monitored, never optimized — the reward curve shows the student reaching the teacher's reward level through distillation alone.

Compared with running FlowGRPO again, distillation needs no reward in the loss and converges in far fewer steps. See the [algorithm doc](../../docs/algo/diffusion_opd.md) for the loss and all config knobs.

## Installation

Follow the [installation guide](../../docs/start/install.md) to set up the base environment, then install the OCR reward dependency:

```bash
pip install Levenshtein
```

The provided script uses a single node with `3` GPUs: 2 for actor + rollout (the frozen teacher shares them), 1 for the reward model server.

## Prepare the dataset

Obtain the raw OCR dataset from the original Flow-GRPO repository:

- https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr

Place it under `$WORKSPACE/data/ocr` (where `WORKSPACE` defaults to `$HOME`), then preprocess it into parquet files:

```bash
python3 examples/flowgrpo_trainer/data_process/sd3_ocr.py \
  --input_dir $WORKSPACE/data/ocr \
  --output_dir $WORKSPACE/data/ocr/sd3
```

The script reads:

```bash
ocr_train_path=$WORKSPACE/data/ocr/sd3/train.parquet
ocr_test_path=$WORKSPACE/data/ocr/sd3/test.parquet
```

## Prepare the teacher

The teacher is a full diffusers checkpoint from the same pipeline family as the student, with the same scheduler. The natural way to get one is the [SD3.5 FlowGRPO OCR example](../flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora.sh): train the LoRA, merge it into the base transformer (`peft` `merge_and_unload`), and save the merged pipeline. Any stronger same-family checkpoint works the same way.

## Run

```bash
TEACHER_PATH=/path/to/merged-teacher \
  bash examples/diffusionopd_trainer/sd35/run_sd35_medium_ocr_distill.sh
```

The multi-teacher recipe (`run_sd35_medium_mopd_distill.sh`) routes each row to
its task's teacher by `data_source`; `run_sd35_medium_mopd_distill_v1.sh` is
the same recipe on the v1 sync trainer.

## What to expect

- `actor/distill_kl_loss` starts clearly positive (the teacher's weights differ from the student's) and falls by more than an order of magnitude over the first ~40 steps.
- The validation OCR reward climbs from the base-model level to the teacher's level within a few tens of steps, even though the loss never sees it.
- `timing_s/teacher` reports the once-per-step teacher scoring stage.

To combine distillation with a task reward instead of replacing it, keep `diffusion_loss.loss_mode=flow_grpo` and set `actor.use_distill_loss=True` — see the [algorithm doc](../../docs/algo/diffusion_opd.md) for details.
