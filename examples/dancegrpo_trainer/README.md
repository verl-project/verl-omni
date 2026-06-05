# DanceGRPO Trainer

This example shows how to post-train `Wan2.2-TI2V-5B` with DanceGRPO on text-to-video generation tasks. DanceGRPO extends FlowGRPO with a score-based SDE step formulation for improved numerical stability during rollout sampling.

For the base Flow-GRPO setup, see [`examples/flowgrpo_trainer/README.md`](../flowgrpo_trainer/README.md). For algorithm details, see [`docs/algo/flowgrpo.md`](../../docs/algo/flowgrpo.md).

## Installation

Follow the [installation guide](../../docs/start/install.md) to set up the base environment.

The provided scripts are configured for a single node with `8` NPUs.

## Prepare the dataset

The Text-to-Video generation task uses text prompts for video generation. There is a pre-split sample dataset with 1,233 prompts from [LanguageBind/Open-Sora-Plan-v1.2.0](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.2.0/blob/main/anno_json/v1.1.0_HQ_part3.json). Pre-split train/test prompt files are provided under `examples/dancegrpo_trainer/data_process/video_prompts/`:

- `train.txt` — training prompts
- `test.txt` — test prompts

Each file contains one prompt per line. Lines with Chinese characters are automatically filtered out during preprocessing (following the original DanceGRPO convention).

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

**Policy model (Wan2.2-TI2V-5B):** the script uses the Hugging Face Hub ID `Wan-AI/Wan2.2-TI2V-5B-Diffusers` directly — no manual download is required. Hugging Face will cache the weights automatically on first run. To use a local copy instead, edit the `model_name` variable in the script directly.

**Reward model for HPSv3:** download the HPSv3 checkpoint and place it at `$WORKSPACE/CKPT/HPSv3/HPSv3.safetensors`. See the [DanceGRPO repository](https://github.com/XueZeyue/DanceGRPO) for download instructions.

## Run training

### HPSv3 reward

Launch the HPSv3 example from the repository root:

```bash
bash examples/dancegrpo_trainer/run_wan22_5b_t2v_hpsv3_npu.sh
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
- `trainer.n_gpus_per_node=8`

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
trainer.project_name=dance_grpo_npu
trainer.experiment_name=wan22_hpsv3_npu
```

Override these values on the command line if you want to log under a different project or run name.

### Diffusion-specific metrics

See the [Metrics Documentation](../../docs/start/metrics.md) for a full description of all diffusion-specific training metrics.
