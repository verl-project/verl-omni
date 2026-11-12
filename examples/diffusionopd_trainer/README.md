# Diffusion On-Policy Distillation Trainer

Last updated: 10/11/2026

This example distills an OCR-tuned `SD3.5-Medium` teacher into a fresh `SD3.5-Medium` student. The student generates images with its own policy, a frozen teacher scores every denoising step of those trajectories, and the student minimizes the KL between its transition and the teacher's (`distill_kl`). The OCR reward is only monitored, never optimized — the reward curve shows the student reaching the teacher's reward level through distillation alone.

Compared with running FlowGRPO again, distillation needs no reward in the loss and converges in far fewer steps. See the [algorithm doc](../../docs/algo/diffusion_opd.md) for the loss and all config knobs.

Both recipes sample deterministically (`noise_level=0`, 10 steps, CFG 4.5, 512px) with no SDE window, so every step but the last is distilled; the student also trains with CFG 4.5.

## Installation

Follow the [installation guide](../../docs/start/install.md) to set up the base environment, then install the OCR reward dependency:

```bash
uv pip install -e ".[ocr]"
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

The teacher is a full diffusers checkpoint from the same pipeline family as the student, with the same scheduler. The natural way to get one is the [SD3.5 FlowGRPO OCR example](https://github.com/verl-project/verl-omni/blob/main/examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora.sh): train the LoRA, merge it into the base transformer (`peft` `merge_and_unload`), and save the merged pipeline. Any stronger same-family checkpoint works the same way.

## Run

```bash
TEACHER_PATH=/path/to/merged-teacher \
  bash examples/diffusionopd_trainer/sd35/run_sd35_medium_ocr_distill.sh
```

Both recipes run on the v1 trainer (`main_diffusion_v1`, `sync` mode).

## Multi-teacher mixture

`run_sd35_medium_mopd_distill.sh` distills the three task teachers of
[quanhaol/DiffusionOPD](https://huggingface.co/quanhaol/DiffusionOPD) (OCR, Aes
and GenEval LoRAs, merged into the base transformer in fp32) into one student on
the OCR + PickScore + GenEval mixture. Prepare the two extra datasets from the
Flow-GRPO splits:

```bash
python3 examples/flowgrpo_trainer/data_process/sd3_pickscore_sfw.py \
  --input-dir $WORKSPACE/data/pickscore --output-dir $WORKSPACE/data/pickscore/sd3 \
  --data-source flow_grpo/pickscore
python3 examples/flowgrpo_trainer/data_process/sd3_geneval.py \
  --input_dir $WORKSPACE/data/geneval --output_dir $WORKSPACE/data/geneval/sd3
```

```bash
OCR_TEACHER_PATH=... AES_TEACHER_PATH=... GENEVAL_TEACHER_PATH=... \
  bash examples/diffusionopd_trainer/sd35/run_sd35_medium_mopd_distill.sh
```

Each row is routed to its task's teacher by `data_source`; the OCR and Aes
teachers run CFG 4.5 and the GenEval teacher CFG 1.0
(`distillation.teacher_models.<name>.guidance_scale`). `TaskBalancedSampler`
(`data.sampler`) gives every generation batch the same number of rows per task,
and one optimizer step spans the whole batch, so an update matches feeding one
single-task batch per teacher under gradient accumulation. Rewards are monitored
only: GenRM OCR for OCR rows and PickScore for PickScore rows. GenEval rows get a
constant score of 0, since the detector-based GenEval scorer is not ported;
evaluate GenEval offline.

## What to expect

- `actor/distill_kl_loss` starts clearly positive (the teacher's weights differ from the student's) and falls by more than an order of magnitude over the first ~40 steps.
- The validation OCR reward climbs from the base-model level to the teacher's level within a few tens of steps, even though the loss never sees it.
- `timing_s/teacher` reports the once-per-step teacher scoring stage.

To combine distillation with a task reward instead of replacing it, keep `diffusion_loss.loss_mode=flow_grpo` and set `actor.use_distill_loss=True` — see the [algorithm doc](../../docs/algo/diffusion_opd.md) for details.
