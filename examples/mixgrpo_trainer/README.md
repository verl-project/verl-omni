# MixGRPO Trainer

This example shows how to post-train `Qwen-Image` with MixGRPO on an OCR-style image generation task using `vllm-omni` rollout and a visual generative reward model (`Qwen3-VL-8B-Instruct` in this example).

MixGRPO extends FlowGRPO with a **Mixed ODE-SDE rollout** and a **sliding-window training schedule**. This greatly cuts the cost of online RL fine-tuning of flow-matching diffusion models by using deterministic ODE sampling outside a contiguous window of denoising steps and stochastic SDE sampling inside the window.

For algorithm details, configuration reference, and tuning guides, see `docs/algo/mixgrpo.md`. For the full installation and base FlowGRPO quickstart guide, see `docs/start/flowgrpo_quickstart.md`.

## Installation

Follow the [installation guide](../../docs/start/install.md) to set up the base environment, then install the MixGRPO/FlowGRPO-specific dependency:

```bash
pip install Levenshtein
```

The provided GPU script is configured for a single node with `4` GPUs. An NPU script for Ascend 800T A2 with `8` NPUs is also available (see [Run training](#run-training) below).

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

**GPU (4 GPUs):**

```bash
bash examples/mixgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_mixgrpo.sh
```

**NPU (8 NPUs, Atlas 800T A2):**

The NPU script requires the CANN software stack. Before running, set the `ASCEND_HOME_PATH` environment variable (defaults to `/usr/local/Ascend/cann-9.0.0`).

```bash
bash examples/mixgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_mixgrpo_npu.sh
```

### MixGRPO Tuning

The default script uses the recommended "reference recipe" (10-step trajectory, 2-step SDE window, random strategy). To tune MixGRPO (e.g. for longer trajectories), adjust these variables in the script:

- `actor_rollout_ref.model.algorithm=mix_grpo`
- `actor_rollout_ref.rollout.algo.sample_strategy=random` (or `progressive`)
- `actor_rollout_ref.rollout.algo.sde_window_size=2`
- `actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=50`

See `docs/algo/mixgrpo.md` for full tuning recommendations.

## Logging

W&B logging is enabled by default in the example script:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
```

The script sets:

```bash
trainer.logger='["console", "wandb"]'
trainer.project_name=mix_grpo
trainer.experiment_name=qwen_image_ocr_lora_mixgrpo
```

Override these values on the command line if you want to log under a different project or run name.

### Diffusion-specific metrics

See the [Metrics Documentation](../../docs/start/metrics.md) for a full description of all diffusion-specific training metrics.
