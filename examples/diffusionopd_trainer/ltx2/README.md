# LTX-2.3 audio-video on-policy distillation

Last updated: 10/15/2026

This recipe distills a frozen `dg845/LTX-2.3-Diffusers` teacher into an LTX-2.3
LoRA student on the text-to-audio-video (T2AV) task. The student generates joint
audio-video rollouts with its own policy, the teacher replays every stored CPS
transition of those trajectories, and the student minimizes the KL between its
transition means and the teacher's (`distill_kl`). The CLAP and ImageBind rewards
are monitored only -- they never enter the loss -- so the reward curve shows the
student reaching the teacher's reward level through distillation alone.

The recipe runs on the v1 trainer (`main_diffusion_v1`, `sync` mode) and
auto-detects Ascend NPU or GPU. See the [algorithm doc](../../../docs/algo/diffusion_opd.md)
for the loss and every config knob, and the
[LTX-2.3 Flow-GRPO example](../../flowgrpo_trainer/ltx2/README.md) for the
rollout and reward setup this recipe inherits.

## Installation

Follow the [installation guide](../../../docs/start/install.md), then install the
optional ImageBind reward dependency (CC-BY-NC-SA 4.0):

```bash
pip install git+https://github.com/facebookresearch/ImageBind.git pytorchvideo
```

## Prepare the dataset

Reuse the Flow-GRPO T2AV prompt corpus and converter:

```bash
python3 examples/flowgrpo_trainer/ltx2/prepare_data.py \
  --input_dir ./dataset/vid_prompt \
  --output_dir "$WORKSPACE/data/vid_prompt/verl_omni" \
  --val_size 128
```

The script reads `$DATA_DIR/train.parquet` and `$DATA_DIR/test.parquet`, where
`DATA_DIR` defaults to `$WORKSPACE/data/vid_prompt/verl_omni`.

## Prepare the teacher

The teacher must be a full diffusers checkpoint from the same pipeline family as
the student, resolving to the same scheduler configuration -- worker init compares
`scheduler.config` against the student's and raises on a mismatch. The natural way
to get one is the
[LTX-2.3 T2AV Flow-GRPO example](../../flowgrpo_trainer/ltx2/run_ltx2_3_t2av_lora_v1.sh):
train the LoRA, merge it into the base transformer (`peft` `merge_and_unload`), and
save the merged pipeline. LoRA adapters must be merged before use; the teacher never
loads them.

## Run

```bash
TEACHER_PATH=/path/to/merged-teacher \
  bash examples/diffusionopd_trainer/ltx2/run_ltx2_3_t2av_opd_npu.sh
```

The defaults use a single node of 16 NPUs for actor, rollout and the colocated
teacher, with rollout TP 4 and the two reward functions on `npu:0` and `npu:1`.
`NUM_GPUS`, `ROLLOUT_TP`, `MODEL_PATH`, `DATA_DIR`, `CLAP_MODEL_PATH`,
`IMAGEBIND_MODEL_PATH` and `TOTAL_TRAINING_STEPS` are all overridable, and any
trailing arguments are forwarded to Hydra.

### Memory, and the standalone teacher pool

The teacher is built with `lora_rank=0`, so it is a full-weight copy of the LTX-2.3
transformer sharing the actor's devices. With `guidance_scale=4.0` it also runs a
negative pass, so teacher scoring costs two forwards per stored step. If the
colocated run runs out of memory, either lower
`actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` or move the teacher onto its
own pool:

```bash
TEACHER_PATH=/path/to/merged-teacher TEACHER_NNODES=1 TEACHER_NPUS=4 \
  bash examples/diffusionopd_trainer/ltx2/run_ltx2_3_t2av_opd_npu.sh
```

A standalone pool isolates the teacher's memory from the actor and from the rollout
engine's sleep/wake cycle. Each step the teacher scores the whole training batch,
`data.train_batch_size × actor_rollout_ref.rollout.n` trajectories (256 by default),
in micro-batches of `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu` (1 by
default) per rank. Size the pool so `TEACHER_NNODES × TEACHER_NPUS ×` that micro-batch
size divides the batch; otherwise the batch is padded up to the next multiple and the
teacher scores rows that are then discarded.

## What to expect

- `actor/distill_kl_loss` starts clearly positive -- the teacher's weights differ from
  the student's -- and falls as the student matches the teacher.
- Validation CLAP and ImageBind rewards climb toward the teacher's level even though
  the loss never sees them.
- `timing_s/teacher` reports the once-per-step teacher scoring stage.
- Validation samples land under `trainer.validation_data_dir` as `.mp4` files with the
  generated audio multiplexed in. A `.pt` file appears instead when ffmpeg fails, and
  the JSONL index for that step records the reason.

The KL is computed over the concatenated `[video; audio]` latent rows, so the two
modalities are weighted by their row counts. At the default 256x384x81 geometry the
video rows dominate; there is no per-modality weighting.

To combine distillation with the task reward instead of replacing it, keep
`diffusion_loss.loss_mode=flow_grpo` and set `actor.use_distill_loss=True` -- see the
[algorithm doc](../../../docs/algo/diffusion_opd.md).
