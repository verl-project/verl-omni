# Diffusion V1 training

Last updated: 09/10/2026

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

### Hybrid rollout switching

The colocated rollout replicas share GPUs with the actor. By default they are
lent to generation only during warmup and validation: they start in rollout
mode after initialization, and the first training step reclaims them
(aborts their in-flight requests, sleeps them, and removes them from the
load balancer) before any actor update. Validation lends them out again and
the next step reclaims them.

Enable dynamic switching so the trainer also lends them to the next step's
generation whenever the replay buffer is short:

```bash
trainer.v1.separate_async.sync_compatible=false \
trainer.v1.separate_async.hybrid_rollout.enable_switch=true \
bash examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1_separate_async.sh
```

At the end of a step, the trainer syncs the standalone replicas as usual and
then estimates how many prompt groups the next step still misses. If the
expected wait exceeds the recent cost of a switch round trip, it installs the
new actor weights into the colocated replicas, resumes them, and registers them
with the standalone load balancer. The next step submits its prompts, waits
until `switch_threshold_ratio * train_batch_size` groups (at least one
mini-batch) are sampleable, and reclaims the replicas. Aborted diffusion
samples are retried as whole samples on the remaining replicas. Switching
happens at most once per step; `adaptive_switch_threshold` raises the
threshold after sustained sampling waits and lowers it after calm steps.

`enable_switch=true` requires `sync_compatible=false` and
`actor_rollout_ref.rollout.disaggregation.enabled=false`. The remaining
knobs follow the upstream `HybridRolloutSwitchConfig` defaults:
`switch_threshold_ratio`, `adaptive_switch_threshold`,
`switch_threshold_step_up`, `switch_threshold_step_down`,
`switch_threshold_release_steps`, and `switch_cost_window_size`.

Logged metrics: `timing_s/switch_wait`, `timing_s/switch_to_rollout`,
`timing_s/switch_to_trainer`, `separate_async/switch/*`, and
`separate_async/decision/*`. Throughput metrics normalize by the total of
actor and standalone rollout GPUs.

Reclaiming waits for the batch the colocated replicas are executing to
finish: the diffusion engine runs whole request batches and only then
processes the abort and the sleep, so `timing_s/switch_to_trainer` includes
up to one batch of generation. The tiny smoke covers this path:

```bash
ENABLE_SWITCH=1 NUM_WARMUP_BATCHES=1 \
bash tests/special_e2e/run_flowgrpo_qwen_image_v1_separate_async.sh
```

## Important settings

- `trainer.use_v1=true` selects the V1 trainer instead of the legacy diffusion
  trainer.
- `trainer.v1.trainer_mode` selects `sync` or `separate_async`.
- `trainer.v1.separate_async.parameter_sync_step` controls the number of local
  actor updates per rollout-weight synchronization cycle.
- `trainer.v1.separate_async.hybrid_rollout.enable_switch` lends the colocated
  rollout replicas to generation between steps when the replay buffer is
  short.
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
