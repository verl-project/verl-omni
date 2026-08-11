# DanceGRPO Trainer

Last updated: 08/07/2026

This example shows how to post-train `Wan2.2-TI2V-5B` with DanceGRPO on text-to-video generation tasks. DanceGRPO extends FlowGRPO with a score-based SDE step formulation for improved numerical stability during rollout sampling.

For the base Flow-GRPO setup, see [Examples - FlowGRPO Trainer](https://verl-omni.readthedocs.io/en/latest/examples/flowgrpo_trainer.html). For algorithm details, see [Algorithms - Flow-GRPO](../../docs/algo/flowgrpo.md).

## Installation

Follow the [installation guide](../../docs/start/install.md) to set up the base environment.

The provided script auto-detects whether NPUs or GPUs are available and configures the run accordingly (16 NPUs or 8 GPUs on a single node).

## Prepare the dataset

The Text-to-Video generation task uses text prompts for video generation. There is a pre-split sample dataset with 1,233 prompts from [LanguageBind/Open-Sora-Plan-v1.2.0](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.2.0/blob/main/anno_json/v1.1.0_HQ_part3.json). Pre-split train/test prompt files are available at [video_prompts](https://huggingface.co/datasets/Hao7/video_prompts/tree/main):

- `train.txt` — training prompts
- `test.txt` — test prompts

Download them and place under `examples/dancegrpo_trainer/data_process/video_prompts/` before running the conversion script.

You can also prepare your own dataset. Create two plain-text files (one for training, one for testing) with one prompt per line. Lines with Chinese characters are automatically filtered out during preprocessing.

### Convert to parquet

```bash
python3 examples/dancegrpo_trainer/data_process/wan22_hpsv3.py \
  --output_dir $WORKSPACE/data/hpsv3
```

The script reads `video_prompts/train.txt` and `video_prompts/test.txt` by default. To use custom files, pass `--train_path` and `--test_path` explicitly.

This produces:

- `$WORKSPACE/data/hpsv3/train.parquet`
- `$WORKSPACE/data/hpsv3/test.parquet`

## Prepare the models

**Policy model (Wan2.2-TI2V-5B):** the script uses the Hugging Face Hub ID `Wan-AI/Wan2.2-TI2V-5B-Diffusers` directly - no manual download is required. Hugging Face will cache the weights automatically on first run. To use a local copy instead, set the `MODEL_NAME` environment variable or edit the `model_name` variable in the script.

**Reward model for HPSv3:** download the HPSv3 checkpoint and place it at `$WORKSPACE/CKPT/HPSv3/HPSv3.safetensors`. To use a different path, set the `CUSTOM_REWARD_MODEL_PATH` environment variable. See the [DanceGRPO repository](https://github.com/XueZeyue/DanceGRPO) for download instructions.

## Run training

### HPSv3 reward

Launch the HPSv3 example from the repository root:

```bash
bash examples/dancegrpo_trainer/wan22/run_wan22_5b_t2v_hpsv3_auto.sh
```

The script auto-detects the device (`npu` via `npu-smi info`, or `gpu` via `nvidia-smi`) and exits with an error if neither is found.

#### Configurable environment variables

All of the following can be overridden via environment variables before launching the script:

| Variable | Default | Description |
| --- | --- | --- |
| `TRAIN_FILES_PATH` | `$WORKSPACE/data/hpsv3/train.parquet` | Training parquet file |
| `VAL_FILES_PATH` | `$WORKSPACE/data/hpsv3/test.parquet` | Validation parquet file |
| `MODEL_NAME` | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | Policy model path or Hub ID |
| `CUSTOM_REWARD_MODEL_PATH` | `$WORKSPACE/CKPT/HPSv3/HPSv3.safetensors` | HPSv3 reward model checkpoint |
| `ROLLOUT_TP` | `1` | Rollout tensor parallel size |
| `TRAIN_BATCH_SIZE` | `64` | Training batch size |
| `WORKSPACE` | `$HOME` | Base directory for data and checkpoints |

Example with custom overrides:

```bash
TRAIN_FILES_PATH=/data/my_train.parquet \
VAL_FILES_PATH=/data/my_val.parquet \
MODEL_NAME=/path/to/local/model \
CUSTOM_REWARD_MODEL_PATH=/path/to/HPSv3.safetensors \
bash examples/dancegrpo_trainer/wan22/run_wan22_5b_t2v_hpsv3_auto.sh
```

The script runs `python3 -m verl_omni.trainer.main_diffusion` with:

- `algorithm.adv_estimator=dance_grpo`
- `actor_rollout_ref.model.path=Wan-AI/Wan2.2-TI2V-5B-Diffusers`
- `actor_rollout_ref.rollout.name=vllm_omni`
- `actor_rollout_ref.actor.diffusion_loss.loss_mode=dance_grpo`
- `actor_rollout_ref.rollout.algo.sde_type=dance_sde`
- `actor_rollout_ref.rollout.algo.noise_level=1.2`
- `actor_rollout_ref.rollout.algo.sde_window_size=2`
- `reward.custom_reward_function.name=compute_score_hpsv3`
- `trainer.n_gpus_per_node=16` (NPU) or `8` (GPU)
- `trainer.total_training_steps=120`

## SDE variants

DanceGRPO supports three SDE step variants via `actor_rollout_ref.rollout.algo.sde_type`:

| `sde_type` | Source | Description |
| --- | --- | --- |
| `dance_sde` | [DanceGRPO](https://github.com/XueZeyue/DanceGRPO) | Score-based SDE correction with `eta` controlling stochasticity. Numerically stable when sigma is close to 1. **(Recommended)** |
| `sde` | [FlowGRPO](https://arxiv.org/abs/2505.05470) | Original FlowGRPO SDE variant. May be numerically unstable when sigma is close to 1. |
| `cps` | — | Consistency-preserving sampling variant. |

## Logging

W&B or TensorBoard logging can be configured in the example script:

```bash
# TensorBoard (default)
trainer.logger='["console", "tensorboard"]'

# W&B
export WANDB_API_KEY=<your_wandb_api_key>
trainer.logger='["console", "wandb"]'
```

The script sets:

```bash
trainer.project_name=dance_grpo
trainer.experiment_name=wan22_5b_t2v_hpsv3_npu  # or wan22_5b_t2v_hpsv3_gpu
```

These values are set automatically based on the detected device. Override them by editing the `PROJECT_NAME` and `EXPERIMENT_NAME` variables in the script.

### Diffusion-specific metrics

See the [Metrics Documentation](../../docs/start/metrics.md) for a full description of all diffusion-specific training metrics.
