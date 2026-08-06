# Diffusion V1 training

Last updated: 07/27/2026

This guide runs the diffusion V1 trainer in synchronous mode using the provided
Stable Diffusion 3.5 Medium FlowGRPO OCR recipe. The V1 trainer uses
TransferQueue and ReplayBuffer to move rollout trajectories into the training
loop. Synchronous mode waits for a complete rollout batch before each training
step.

The current V1 example supports a single-node NVIDIA GPU setup. The provided
recipe uses three GPUs: two for the actor and rollout, and one for the reward
model.

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

## Important settings

- `trainer.use_v1=true` selects the V1 trainer instead of the legacy diffusion
  trainer.
- `trainer.v1.trainer_mode=sync` selects synchronous rollout and training.
- `actor_rollout_ref.rollout.agent.num_workers` controls the rollout worker
  count.
- `transfer_queue.backend.SimpleStorage.total_storage_size` controls the
  maximum number of experience samples held by the default backend.
- `transfer_queue.backend.SimpleStorage.num_data_storage_units` controls the
  number of in-memory storage units.

Only `sync` mode is currently implemented for the diffusion V1 trainer.
`colocate_async` and `separate_async` are not yet supported.

## Troubleshooting

`ModuleNotFoundError: No module named 'transfer_queue'`
: Install TransferQueue in the same environment used to launch Ray, then run
  the import verification command above.

Ray workers cannot import `transfer_queue`
: Stop the existing Ray cluster with `ray stop`, activate the environment where
  TransferQueue is installed, and launch the recipe again.

