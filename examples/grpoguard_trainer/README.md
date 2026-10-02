# GRPO-Guard Trainer

Last updated: 06/30/2026

This example shows how to post-train `Qwen-Image` with GRPO-Guard on an OCR-style image generation task. GRPO-Guard extends Flow-GRPO with a reverse-SDE proposal-mean drift correction and per-step loss rescaling for improved training stability.

For algorithm details, see [Algorithms - GRPO-Guard](../../docs/algo/grpo_guard.md). For the base Flow-GRPO setup this example builds on, see [Examples - FlowGRPO Trainer](https://verl-omni.readthedocs.io/en/latest/examples/flowgrpo_trainer.html).

## Installation

Follow the [installation guide](../../docs/start/install.md) to set up the base environment, then install the GRPO-Guard-specific dependency:

```bash
uv pip install -e ".[ocr]"
```

The GPU scripts default to a single node with `4` GPUs. An NPU script for Ascend 800T A2 with `8` NPUs is also available (see [Run training](#run-training) below).

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

**Policy model (Qwen-Image):** the scripts use the Hugging Face Hub ID `Qwen/Qwen-Image` directly — no manual download is required. For the V1 script, set `MODEL_PATH` to use a local copy.

**Reward model (Qwen3-VL-8B-Instruct):** the scripts default to the Hugging Face Hub ID `Qwen/Qwen3-VL-8B-Instruct`. For the V1 script, set `REWARD_MODEL_PATH` to use a local copy.

## Run training

Launch the example from the repository root:

**GPU (4 GPUs):**

```bash
bash examples/grpoguard_trainer/qwen_image/run_qwen_image_ocr_lora.sh
```

**GPU V1 sync:**

```bash
bash examples/grpoguard_trainer/qwen_image/run_qwen_image_ocr_lora_v1.sh
```

The V1 recipe keeps the V0 GRPO-Guard loss, SDE, LoRA, reward, and batch settings. It uses `main_diffusion_v1`, `model.algorithm=flow_grpo`, and `trainer.v1.trainer_mode=sync`. Set `NUM_GPUS_ACTOR_ROLLOUT_REWARD`, `ROLLOUT_TP`, and `REWARD_TP` for a different GPU layout; for example, `8`, `2`, and `4` respectively.

See the [5090 V0/V1 results](qwen_image/results/20260927-5090-smoke/README.md) for step timing and OCR reward measurements.

For three RTX 5090 GPUs, a separate candidate recipe uses two actor/rollout GPUs and one OCR reward GPU, with FSDP2 and rollout CPU offload:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
bash examples/grpoguard_trainer/qwen_image/run_qwen_image_ocr_lora_v1_5090.sh
```

This candidate uses four prompts and four images per prompt, a learning rate of `3e-5`, and 512×512 images with ten denoising steps. Allow at least 300 GiB of available host memory before starting the offloaded workload. It validates before training and every 20 steps, retaining one checkpoint. The lower learning rate and larger sampling group are being evaluated after the batch-4/group-2 run at `3e-4` lost validation reward; convergence has not yet been established. The original V1 recipe above preserves the V0 defaults.

**NPU (8 NPUs, Atlas 800T A2):**

The NPU script requires the CANN software stack. Before running, set the `ASCEND_HOME_PATH` environment variable (defaults to `/usr/local/Ascend/cann-9.0.0`).

```bash
bash examples/grpoguard_trainer/qwen_image/run_qwen_image_ocr_lora_npu.sh
```

The recipes use these settings:

- `algorithm.adv_estimator=flow_grpo`
- `actor_rollout_ref.model.path=Qwen/Qwen-Image`
- `actor_rollout_ref.model.lora_rank=64`
- `actor_rollout_ref.model.lora_alpha=128`
- `actor_rollout_ref.rollout.name=vllm_omni`
- `actor_rollout_ref.actor.diffusion_loss.loss_mode=grpo_guard`
- `actor_rollout_ref.actor.diffusion_loss.clip_ratio=2e-6`
- `actor_rollout_ref.rollout.algo.sde_type=sde`
- `reward.custom_reward_function.name=compute_score_ocr`

Due to differences in memory capacity, the NPU and GPU configurations differ as follows:

| Parameter | GPU | NPU |
|---|---|---|
| `trainer.device` | gpu (default) | `npu` |
| `actor_rollout_ref.model.attn_backend` | default | `_native_npu` |
| `trainer.n_gpus_per_node` | 4 | 8 |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | 1 | 2 |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | 16 | 4 |

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
