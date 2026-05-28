# GRPO-Guard Trainer

This example shows how to post-train `Qwen-Image` with GRPO-Guard on an OCR-style image generation task. GRPO-Guard extends Flow-GRPO with a reverse-SDE proposal-mean drift correction and per-step loss rescaling for improved training stability.

For algorithm details, see [`docs/algo/grpo_guard.md`](../../docs/algo/grpo_guard.md). For the base Flow-GRPO setup this example builds on, see [`examples/flowgrpo_trainer/README.md`](../flowgrpo_trainer/README.md).

## Installation

Follow the [installation guide](../../docs/start/install.md) to set up the base environment, then install the GRPO-Guard-specific dependency:

```bash
pip install Levenshtein
```

The provided script is configured for a single node with `4` GPUs.

## Prepare the dataset

Obtain the raw OCR dataset from the original Flow-GRPO repository:

- https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr

Place the raw dataset under `$WORKSPACE/data/ocr` (where `WORKSPACE` defaults to `$HOME`), then preprocess it into parquet files:

```bash
python3 examples/flowgrpo_trainer/data_process/qwenimage_ocr.py \
  --input_dir $WORKSPACE/data/ocr \
  --output_dir $WORKSPACE/data/ocr/qwen_image
```

This produces:

- `$WORKSPACE/data/ocr/qwen_image/train.parquet`
- `$WORKSPACE/data/ocr/qwen_image/test.parquet`

## Prepare the models

**Policy model (Qwen-Image):** the script uses the Hugging Face Hub ID `Qwen/Qwen-Image` directly — no manual download is required. Hugging Face will cache the weights automatically on first run. To use a local copy instead, edit the `model_name` variable in the script directly.

**Reward model (Qwen3-VL-8B-Instruct):** the script defaults to the Hugging Face Hub ID `Qwen/Qwen3-VL-8B-Instruct`, so no manual download is required — Hugging Face will cache it automatically on first run. To use a local copy instead, edit the `reward_model_name` variable in the script directly.

## Run training

Launch the example from the repository root:

```bash
bash examples/grpoguard_trainer/run_qwen_image_ocr_lora.sh
```

The script runs `python3 -m verl_omni.trainer.main_diffusion` with:

- `algorithm.adv_estimator=flow_grpo`
- `actor_rollout_ref.model.path=Qwen/Qwen-Image`
- `actor_rollout_ref.model.lora_rank=64`
- `actor_rollout_ref.model.lora_alpha=128`
- `actor_rollout_ref.rollout.name=vllm_omni`
- `actor_rollout_ref.actor.diffusion_loss.loss_mode=grpo_guard`
- `actor_rollout_ref.actor.diffusion_loss.clip_ratio=2e-6`
- `actor_rollout_ref.rollout.algo.sde_type=sde`
- `reward.custom_reward_function.name=compute_score_ocr`
- `trainer.n_gpus_per_node=4`

## Logging

W&B logging is enabled by default in the example script:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
```

The script sets:

```bash
trainer.logger='["console", "wandb"]'
trainer.project_name=grpo_guard
trainer.experiment_name=qwen_image_ocr_lora
```

Override these values on the command line if you want to log under a different project or run name.

### Diffusion-specific metrics

See the [Metrics Documentation](../../docs/start/metrics.md) for a full description of all diffusion-specific training metrics.
