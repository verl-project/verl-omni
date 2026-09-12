# Diffusion V1 training

Last updated: 09/11/2026

This guide runs the diffusion V1 trainer in synchronous or separate-asynchronous
mode using the provided Stable Diffusion 3.5 Medium FlowGRPO OCR recipes.
Qwen-Image FlowGRPO now has a matching V1 sync LoRA recipe as well. The V1
trainer uses TransferQueue and ReplayBuffer to move rollout trajectories into
the training loop. Synchronous mode waits for a complete rollout batch before
each training step. Wan2.2 DanceGRPO on CUDA also defaults to the V1 sync
recipe; see {doc}`../examples/dancegrpo_trainer`.

The examples support a single-node NVIDIA GPU setup. Sync mode uses two GPUs for
the colocated actor and rollout plus one reward GPU. Separate-async mode also
requires dedicated standalone rollout GPUs.

## Prerequisites

Install VeRL-Omni and its training dependencies by following the
{doc}`installation guide <install>`. Run all commands below from the repository
root in the same Python environment.

The OCR reward also requires Levenshtein:

```bash
uv pip install -e ".[ocr]"
```

Verify that TransferQueue and the V1 entrypoint can be imported:

```bash
python -c "import transfer_queue; import verl_omni.trainer.main_diffusion_v1; print('Diffusion V1 dependencies are ready')"
```

The package is installed as `TransferQueue` and imported in Python as
`transfer_queue`.

## Prepare the OCR dataset

Set `WORKSPACE` to a writable directory. It defaults to `$HOME` in the run
script:

```bash
export WORKSPACE=${WORKSPACE:-$HOME}
```

Download `train.txt` and `test.txt` from the original
[Flow-GRPO OCR dataset](https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr)
and place them in `$WORKSPACE/data/ocr`. Convert them to parquet files:

```bash
python3 examples/flowgrpo_trainer/data_process/sd3_ocr.py \
  --input_dir "$WORKSPACE/data/ocr" \
  --output_dir "$WORKSPACE/data/ocr/sd3"
```

This creates:

- `$WORKSPACE/data/ocr/sd3/train.parquet`
- `$WORKSPACE/data/ocr/sd3/test.parquet`

See the {doc}`FlowGRPO quickstart <flowgrpo_quickstart>` for the dataset format
and custom-dataset instructions.

## Run V1 sync mode

Launch the V1 SD3.5 Medium LoRA recipe:

```bash
bash examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh
```

The script selects the V1 synchronous path with:

```text
python3 -m verl_omni.trainer.main_diffusion_v1
trainer.use_v1=true
trainer.v1.trainer_mode=sync
```
Hydra settings can be appended to the command. For example, to run fewer steps
and disable W&B:

```bash
bash examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh \
  trainer.total_training_steps=10 \
  trainer.logger='["console"]'
```

Checkpoints are written by default to:

```text
checkpoints/flow_grpo/sd35_medium_ocr_lora_v1
```

### Qwen-Image FlowGRPO

Qwen-Image FlowGRPO uses the same V1 sync entrypoint and flags. Launch the
4-GPU LoRA OCR recipe with:

```bash
bash examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_v1.sh
```

Model, LoRA, reward, pipeline, and SDE knobs match the v0 script
`run_qwen_image_ocr_lora.sh`. Prepare the Qwen-Image OCR parquet files as in
the {doc}`FlowGRPO quickstart <flowgrpo_quickstart>` (use `qwenimage_ocr.py`,
not the SD3 converter above). Checkpoints default to
`checkpoints/flow_grpo/qwen_image_ocr_lora_v1`.

### Wan2.2 DanceGRPO (default CUDA recipe)

Wan2.2 DanceGRPO on CUDA now defaults to the same V1 sync trainer:

```bash
bash examples/dancegrpo_trainer/wan22/run_wan22_5b_t2v_hpsv3_v1.sh
```

See {doc}`../examples/dancegrpo_trainer` for dataset and HPSv3 setup. The
legacy v0 auto-detect script (`run_wan22_5b_t2v_hpsv3_auto.sh`) is
**deprecated** for CUDA and remains for NPU.

## Run V1 separate-async mode

Launch the separate-async recipe:

```bash
bash examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1_separate_async.sh
```

This mode runs standalone rollout workers on dedicated GPUs. For
`parameter_sync_step=N`, each outer step consumes `N` complete PPO mini-batches,
keeps `π_old` fixed at the cycle-start actor weights, and synchronizes rollout
weights once after all `N` actor updates. Configure batch sizes with:

```text
data.train_batch_size =
    trainer.v1.separate_async.parameter_sync_step *
    actor_rollout_ref.actor.ppo_mini_batch_size
```

For example:

```bash
PARAMETER_SYNC_STEP=4 \
TRAIN_BATCH_SIZE=8 \
bash examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1_separate_async.sh
```

`sync_compatible=true` pauses standalone generation during actor updates. It
requires `num_warmup_batches=0`; set it to `false` to retain rollout/training
overlap.

### Qwen-Image LoRA policy snapshots

For Qwen-Image (`QwenImagePipeline`) with FSDP2, a single `default` LoRA
adapter and sequence parallel size 1, separate-async CPU snapshots retain only
the trainable parameters. This includes any trainable parameters outside the
LoRA layers. The cycle-start `π_old` and current-actor restoration semantics
are unchanged; no new configuration is required.

This reduces the snapshot payload, not the cost of materializing the actor
when manual parameter offload is enabled. FSDP1, other models/backends,
multiple policy adapters, native FSDP2 CPU offload policy, and full-parameter
training retain full snapshots. Changing the trainable parameter set or shard
layout within a snapshot's lifetime is unsupported and fails before restore.

To measure snapshot overhead with a locally available pretrained Qwen-Image
checkpoint, run the standalone benchmark outside fast CI:

```bash
torchrun --standalone --nproc-per-node=2 scripts/benchmark_trainable_snapshot.py \
    --model /path/to/Qwen-Image --output /tmp/qwen-snapshot-60-offload \
    --run-id qwen-image-depth60-offload-20260911-a \
    --checkpoint-revision FULL_MODEL_COMMIT --source-revision FULL_CODE_COMMIT \
    --checkpoint-manifest /path/to/transformer-shards.manifest \
    --layers 60 --lora-rank 64 --sync-steps 3 --warmup 2 --repeats 10 --offload
```

Use a fresh output directory for each run. Omit `--offload` for GPU-resident
parameters. Depths 10 and 30 select prefixes of the original pretrained
transformer, preserving its width; these are not full-model quality benchmarks.
The script requires Linux/glibc and exactly two CUDA ranks. Budget host memory
for two retained snapshots, temporary cloning, and manual offload, in addition
to model loading; the tiny regression's memory allocation is not sufficient.
Both source and model directories must be Git checkouts at the declared
revisions. The manifest has one `path|bytes|sha256` row per transformer shard,
obtained from that revision's Git LFS pointers. Every shard is hashed before
loading; the source/metrics bundle is also digested. The fixed matrix uses
depth 10/30/60, rank 64, seed 20260911, sync steps 3, two warmup pairs and ten
timed pairs. Change the experiment contract before adding another setting.
Before CUDA or NCCL initialization, rank zero creates the output directory,
verifies all pinned metadata and shard hashes, and atomically publishes a
self-identifying run manifest plus preflight status. Other ranks independently
verify their imported source checkouts and wait up to 900 seconds for that
same run ID. Every partial and final result references the manifest digest.

The benchmark loads once per depth/offload run and alternates full/trainable
snapshot pairs. `summary.json` reports the slowest-rank latency per pair;
`result-rank*.json` retains raw timings, exact tensor payloads, and a separate
5-ms sampled RSS/GPU-memory pass. RSS is allocator-sensitive and sampled, not
an exact peak. Bootstrap intervals describe repeats in that allocation, not
independent runs. Forward/backward, rollout and reward are excluded, so
`training_cycle_fraction` is unavailable and no end-to-end speedup is implied.

## Important settings

- `trainer.use_v1=true` selects the V1 trainer instead of the legacy diffusion
  trainer.
- `trainer.v1.trainer_mode` selects `sync` or `separate_async`.
- `trainer.v1.separate_async.parameter_sync_step` controls the number of local
  actor updates per rollout-weight synchronization cycle.
- `actor_rollout_ref.rollout.agent.num_workers` controls the rollout worker
  count.
- `trainer.v1.sampler.drop_incomplete_groups=true` evicts a training prompt
  group when any of its rollout sessions fails and submits the same number of
  replacement prompts. This policy is supported only with
  `trainer.v1.trainer_mode=sync`; validation sampling is unchanged.
- `trainer.v1.sampler.max_incomplete_group_refill_rounds` bounds consecutive
  replacement rounds within one training sample call. Exact refill uses a
  generation batch size of one while the policy is enabled.
- `transfer_queue.backend.SimpleStorage.total_storage_size` controls the
  maximum number of experience samples held by the default backend.
- `transfer_queue.backend.SimpleStorage.num_data_storage_units` controls the
  number of in-memory storage units.

The configurable incomplete-group refill policy above applies to `sync` mode.
In `separate_async`, the upstream async replay buffer automatically evicts and
replaces stale or failed prompt groups. `colocate_async` is not yet supported.

## Troubleshooting

`ModuleNotFoundError: No module named 'transfer_queue'`
: Install TransferQueue in the same environment used to launch Ray, then run
  the import verification command above.

Ray workers cannot import `transfer_queue`
: Stop the existing Ray cluster with `ray stop`, activate the environment where
  TransferQueue is installed, and launch the recipe again.
