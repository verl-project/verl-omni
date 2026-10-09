# FlowDPPO Trainer

Last updated: 06/30/2026

This example shows how to post-train `Qwen-Image` with Flow-DPPO on an OCR-style image generation task using `vllm-omni` rollout and a visual generative reward model (`Qwen3-VL-8B-Instruct` in this example).

Flow-DPPO reuses the FlowGRPO training stack, but replaces ratio clipping with a divergence-based mask over denoising transitions. For algorithm details, see [Algorithms - Flow-DPPO](../../docs/algo/flowdppo.md); for the shared Qwen-Image OCR setup, see [Examples - FlowGRPO Trainer](https://verl-omni.readthedocs.io/en/latest/examples/flowgrpo_trainer.html).

## Installation

Follow the [installation guide](../../docs/start/install.md) to set up the base environment, then install the OCR reward dependency:

```bash
uv pip install -e ".[ocr]"
```

The provided scripts default to a single node with `4` GPUs.

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

**Policy model (Qwen-Image):** the script uses the Hugging Face Hub ID `Qwen/Qwen-Image` directly, so no manual download is required. Hugging Face will cache the weights automatically on first run. To use a local copy instead, set `MODEL_PATH` when launching.

**Reward model (Qwen3-VL-8B-Instruct):** the script defaults to the Hugging Face Hub ID `Qwen/Qwen3-VL-8B-Instruct`, so no manual download is required. Hugging Face will cache it automatically on first run. To use a local copy instead, set `REWARD_MODEL_PATH` when launching.

## Run training

Launch the example from the repository root:

```bash
bash examples/flowdppo_trainer/qwen_image/run_qwen_image_ocr_lora.sh
```

For the V1 `sync` trainer, use the matching recipe:

```bash
bash examples/flowdppo_trainer/qwen_image/run_qwen_image_ocr_lora_v1.sh
```

The V1 recipe keeps the V0 Flow-DPPO loss, SDE, LoRA, reward, and batch settings. It uses `main_diffusion_v1`, `model.algorithm=flow_grpo`, and `trainer.v1.trainer_mode=sync`. Set `NUM_GPUS_ACTOR_ROLLOUT_REWARD`, `ROLLOUT_TP`, and `REWARD_TP` for a different GPU layout; for example, `8`, `2`, and `4` respectively.

See the [5090 V0/V1 results](qwen_image/results/20260927-5090-smoke/README.md) for step timing and OCR reward measurements.

The script accepts normal Hydra overrides after the command:

```bash
bash examples/flowdppo_trainer/qwen_image/run_qwen_image_ocr_lora.sh trainer.total_training_steps=100
```

Both scripts use these Flow-DPPO-specific settings:

- `algorithm.adv_estimator=flow_grpo`
- `actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_dppo`
- `actor_rollout_ref.actor.diffusion_loss.kl_mask_threshold=1e-5`
- `actor_rollout_ref.actor.diffusion_loss.add_kl_coefficient=True`

The policy LoRA uses:

- `actor_rollout_ref.model.lora_rank=64`
- `actor_rollout_ref.model.lora_alpha=128`

## Logging

W&B logging is enabled by default in the example script:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
```

The script sets:

```bash
trainer.logger='["console", "wandb"]'
trainer.project_name=flow_dppo
trainer.experiment_name=qwen_image_ocr_lora
```

Override these values on the command line if you want to log under a different project or run name.

### Diffusion-specific metrics

See the [Metrics Documentation](../../docs/start/metrics.md) for a full description of all diffusion-specific training metrics.

## Relationship to FlowGRPO

Flow-DPPO keeps the same data pipeline, rollout backend, reward model, LoRA setup, and group-relative advantage estimator as FlowGRPO. The main changes are:

- `loss_mode=flow_dppo` enables the divergence-masked policy objective.
- `kl_mask_threshold=1e-5` controls the per-step divergence trust region.
- `add_kl_coefficient=True` normalizes transition drift by the scheduler SDE noise scale.
- `actor_rollout_ref.rollout.algo.sde_type=sde` keeps rollout and replay aligned with the SDE transition variance used by the loss.

See [Algorithms - Flow-DPPO](../../docs/algo/flowdppo.md) for the mathematical objective and references.
