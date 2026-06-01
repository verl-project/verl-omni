# DiffusionNFT Trainer

This example shows how to post-train `Qwen-Image` with DiffusionNFT on an OCR-style image generation task using `vllm-omni` rollout and a visual generative reward model (`Qwen3-VL-8B-Instruct` in this example).

DiffusionNFT is a direct-preference / forward-process algorithm. Unlike PPO-style FlowGRPO training, this example trains from final clean latents and uses an `old` LoRA adapter as the rollout policy while updating the `default` adapter.

For the full installation guide, see `docs/start/install.md`. For implementation details on adding or extending diffusion algorithms, see `docs/contributing/integrating_a_new_algorithm_for_diffusion_model.md`.

## Installation

Follow the [installation guide](../../docs/start/install.md) to set up the base environment, then install the OCR reward dependency:

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
  --output_dir $WORKSPACE/data/ocr
```

The script reads:

```bash
ocr_train_path=$WORKSPACE/data/ocr/train.parquet
ocr_test_path=$WORKSPACE/data/ocr/test.parquet
```

This produces:

- `$WORKSPACE/data/ocr/train.parquet`
- `$WORKSPACE/data/ocr/test.parquet`

Override `WORKSPACE` when launching if your data is elsewhere:

```bash
WORKSPACE=/path/to/workspace bash examples/diffusionnft_trainer/run_qwen_image_ocr_lora.sh
```

## Prepare the models

**Policy model (Qwen-Image):** the script uses the Hugging Face Hub ID `Qwen/Qwen-Image` directly, so no manual download is required. Hugging Face will cache the weights automatically on first run. To use a local copy instead, edit the `model_name` variable in the script directly.

**Reward model (Qwen3-VL-8B-Instruct):** the script defaults to the Hugging Face Hub ID `Qwen/Qwen3-VL-8B-Instruct`, so no manual download is required. Hugging Face will cache it automatically on first run. To use a local copy instead, edit the `reward_model_name` variable in the script directly.

## Run training

Launch the example from the repository root:

```bash
bash examples/diffusionnft_trainer/run_qwen_image_ocr_lora.sh
```

The script accepts normal Hydra overrides after the command:

```bash
bash examples/diffusionnft_trainer/run_qwen_image_ocr_lora.sh trainer.total_training_steps=100
```

The script runs `python3 -m verl_omni.trainer.main_diffusion` with DiffusionNFT-specific settings:

- `actor_rollout_ref.model.algorithm=diffusion_nft`
- `algorithm.trainer_type=direct_preference`
- `actor_rollout_ref.actor.diffusion_loss.loss_mode=diffusion_nft`
- `actor_rollout_ref.model.policy_state_adapters='["default","old"]'`
- `actor_rollout_ref.rollout.calculate_log_probs=False`
- `actor_rollout_ref.rollout.algo.collect_mode=final_latent`
- `actor_rollout_ref.rollout.rollout_adapter=old`
- `actor_rollout_ref.rollout.n=24`
- `algorithm.timestep_fraction=1.0`
- `algorithm.old_policy_decay_type=2`
- `algorithm.old_policy_update_interval=2`
- `algorithm.adv_mode=continuous`
- `actor_rollout_ref.actor.diffusion_loss.mix_beta=0.1`
- `actor_rollout_ref.actor.diffusion_loss.ref_kl_coef=0.0001`
- `trainer.n_gpus_per_node=4`

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
trainer.project_name=diffusion_nft
trainer.experiment_name=qwen_image_ocr_lora_n_24_v_20
```

Override these values on the command line if you want to log under a different project or run name.

### Diffusion-specific metrics

See the [Metrics Documentation](../../docs/start/metrics.md) for a full description of diffusion-specific training metrics.

## Performance

> All experiments were conducted on *NVIDIA H200* GPUs using the OCR reward.

| Script | Model | Algorithm | Hybrid Engine | # Cards | Reward Fn | # GPUs for Actor | # GPUs for Rollout | # GPUs for Async Reward | Batch Size | `rollout.n` | lr   | # Val Samples | Training Samples per Step | `ppo_micro_batch_size_per_gpu` | Throughput (Samples / GPU / Seconds) | Time per Step (Seconds) |
| --- | --- | --- | --- | --- | --- | --- | --- |-------------------------| --- | --- |------| --- | --- | --- |------------------------------| --------------------------------|
| `run_qwen_image_ocr_lora.sh` | Qwen-Image | DiffusionNFT | True | 4 | qwenvl-ocr-vllm | 4 | 4 | 0 (sync)                | 24 | 16 | 3e-4 | 1k (full set) | 24×16=384 | 12 | 0.166                        | 570 |

<table align="center" style="border: none;">
  <tr style="border: none;">
    <td style="text-align: center; border: none; padding: 10px;">
      <h5 style="margin-bottom: 5px;">Validation Performance</h5>
      <img width="400" alt="nft_val" src="https://github.com/user-attachments/assets/915b717d-5fb7-4a67-89f9-b6a4193aad2c" />
    </td>
    <td style="text-align: center; border: none; padding: 10px;">
      <h5 style="margin-bottom: 5px;">Training Progression</h5>
      <img width="400" alt="nft_critic" src="https://github.com/user-attachments/assets/e9f62bbe-2ebd-41ae-85b3-403f163c7da6" />
    </td>
  </tr>
</table>

> **Note:** Reward curves may differ from the references above mainly due to rollout-side stochasticity: diffusion rollouts sample random latents/noise, and the example scripts do not fix the data seed, so prompt ordering can vary between runs.
