# Diffusion V1 training

Last updated: 09/24/2026

This guide runs the diffusion V1 trainer in synchronous or separate-asynchronous
mode using the provided Stable Diffusion 3.5 Medium FlowGRPO OCR recipes.
Qwen-Image FlowGRPO now has a matching V1 sync LoRA recipe as well. The V1
trainer uses TransferQueue and ReplayBuffer to move rollout trajectories into
the training loop. Synchronous mode waits for a complete rollout batch before
each training step. Wan2.2 DanceGRPO on CUDA also defaults to the V1 sync
recipe; see {doc}`../examples/dancegrpo_trainer`.

Since v0.3.0 the V1 trainer is the **default for every diffusion model**:
`trainer.use_v1` defaults to `true`, and `python -m verl_omni.trainer.main_diffusion_v1`
launches it without extra flags. The legacy v0 trainer
(`python -m verl_omni.trainer.main_diffusion` or `trainer.use_v1=false`) is
**deprecated** — see [Legacy v0 trainer (deprecated)](#legacy-v0-trainer-deprecated).

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

`trainer.use_v1=true` and `trainer.v1.trainer_mode=sync` are the defaults since
v0.3.0; the scripts keep them for explicitness.
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
lent to generation only during validation: they start in rollout mode after
initialization and are reclaimed (removed from the load balancer, in-flight
requests aborted, slept) before the warmup batches are fed. Validation lends
them out again and the next step reclaims them.

Enable dynamic switching so the trainer also lends them to the next step's
generation whenever the replay buffer is short:

```bash
SYNC_COMPATIBLE=false bash examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1_separate_async.sh \
    trainer.v1.separate_async.hybrid_rollout.enable_switch=true
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
`separate_async/decision/*`.

This is the v1 analog of verl's fully_async_policy
`DynamicResourceController` (verl#6556). That controller is not wired on
`main_diffusion_v1`; setting `async_training.use_dynamic_resource_scheduling=true`
raises at startup.

### Checkpoint recovery

Separate-async checkpoints save TransferQueue next to the actor and
dataloader (`global_step_N/transfer_queue/`). Resume with
`trainer.resume_mode=auto` (or `resume_path`) restores queued prompt groups,
re-issues pending and running groups, and tops warmup up to
`num_warmup_batches * train_batch_size` instead of enqueueing that many new
batches. Requires TransferQueue 0.1.9. Sync mode does not write a queue
snapshot. Old checkpoints without `transfer_queue/` warn and start the queue
empty.

Streaming refill already uses `data.gen_batch_size=1` in `separate_async`
(and whenever exact incomplete-group refill is on). Do not set a larger
`gen_batch_size` expecting it to stick; the trainer overrides it.

Throughput metrics of every `separate_async` run, with switching on or off, now
also count the standalone rollout GPUs in the denominator, so they read lower
than earlier runs of the same recipe.

Reclaiming waits for the batch the colocated replicas are executing to
finish: the diffusion engine runs whole request batches and only then
processes the abort and the sleep, so `timing_s/switch_to_trainer` includes
up to one batch of generation. The tiny smoke covers this path:

```bash
ENABLE_SWITCH=1 NUM_WARMUP_BATCHES=1 \
bash tests/special_e2e/run_flowgrpo_qwen_image_v1_separate_async.sh
```

## How TransferQueue supports the V1 trainer

TransferQueue (pip package `TransferQueue`, imported as `transfer_queue`) is the
streaming queue the V1 control plane uses to hand rollout data to the trainer.
Every V1 diffusion run depends on it:

1. **Install.** TransferQueue must be importable in the environment that
   launches Ray *and* in every Ray worker. CI pins `pip install
   TransferQueue==0.1.9`; use the same version unless a newer one is announced.
   The import check in [Prerequisites](#prerequisites) fails fast when it is
   missing.
2. **Force-enabled lifecycle.** The yaml default `transfer_queue.enable` is
   `false`, but a V1 launch force-sets it to `true` before `ray.init()` — a V1
   run cannot start without TransferQueue. `ray.init` then exports
   `TRANSFER_QUEUE_ENABLE=1` through the Ray runtime env so all workers join the
   same queue, and the task runner wraps training in `tq.init(config.transfer_queue)`
   … `tq.close()`. If Ray workers report `ModuleNotFoundError: No module named
   'transfer_queue'`, stop the cluster (`ray stop`) and relaunch from the
   environment where TransferQueue is installed.
3. **Rollout-side producer.** The diffusion agent loop ships a TransferQueue
   writer (`diffusion_agent_loop_tq.py`) that serializes each finished rollout
   session — prompts, latents, rewards, and an explicit allowlist of extra
   fields (e.g. `img_shapes` for Qwen-Image 2D RoPE). The allowlist is
   intentional: silently forwarding unknown fields has broken metadata before,
   so new fields must be added there explicitly.
4. **Trainer-side consumer.** The trainer converts queued rows back to
   `DataProto` batches (`tq_utils.diffusion_tq_batch_to_dataproto`) and feeds
   them to the ReplayBuffer, which drives `sync` batching, staleness eviction,
   and `separate_async` partial rollout.
5. **Tuning.** `transfer_queue.backend.SimpleStorage.total_storage_size` caps
   how many experience samples the default backend holds;
   `num_data_storage_units` sets the in-memory storage units; metrics can be
   exposed via `transfer_queue.metrics.*`. See [Important settings](#important-settings).

Omni models get the same support through verl's `TaskRunnerV1`
(`python -m verl_omni.trainer.main_omni`); omni inherits verl's TransferQueue
setup and requires TransferQueue >= 0.1.9 for async checkpoint resume.

## Important settings

- `trainer.use_v1` (default `true` since v0.3.0) selects the V1 trainer; setting
  it to `false` explicitly selects the deprecated legacy v0 trainer.
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

## Legacy v0 trainer (deprecated)

The legacy v0 diffusion trainer — launched by
`python -m verl_omni.trainer.main_diffusion` or by
`trainer.use_v1=false` — is **deprecated for every model** since v0.3.0:

- Every v0 launch of a path that has a V1 equivalent emits a
  `DeprecationWarning`; the v0 trainer will be removed in a future release.
- Models without a landed V1 recipe (tracked in
  [verl-project/verl-omni#389](https://github.com/verl-project/verl-omni/issues/389))
  keep working on v0 until their port lands; the warning is expected there.
  Their scripts pin `trainer.use_v1=false` explicitly, so the launch stays on
  v0 even if the entrypoint is swapped to `main_diffusion_v1`.
- **Stays on v0 by design** (no warning, no V1 planned): diffusion offline DPO
  (`examples/dpo_trainer/sd35/`) and omni offline DPO
  (`algorithm.sample_source=offline` + `algorithm.trainer_type=direct_preference`).

Migrating a v0 recipe to V1 sync takes three lines — swap the entrypoint and
the two (now-default) switches:

```text
python3 -m verl_omni.trainer.main_diffusion   ->  python3 -m verl_omni.trainer.main_diffusion_v1
trainer.use_v1=false                          ->  trainer.use_v1=true
trainer.v1.trainer_mode=sync
```

Batch math is unchanged for `sync`; `separate_async` adds the
`train_batch_size = parameter_sync_step * ppo_mini_batch_size` identity
described above. Install TransferQueue as described in
[Prerequisites](#prerequisites) before the first V1 launch.

## Troubleshooting

`ModuleNotFoundError: No module named 'transfer_queue'`
: Install TransferQueue in the same environment used to launch Ray, then run
  the import verification command above.

Ray workers cannot import `transfer_queue`
: Stop the existing Ray cluster with `ray stop`, activate the environment where
  TransferQueue is installed, and launch the recipe again.
