(performance)=
# Performance Reference

Last updated: 06/17/2026

Below are reference benchmark results for VeRL-Omni training runs.

## FlowGRPO: LoRA Training on Qwen-Image OCR

> All experiments used NVIDIA H800 GPUs, LoRA rank 64, `ppo_micro_batch_size_per_gpu` 16, and the full 1k validation set. Training images per step = batch size × images per prompt = 32 × 16 = 512.

### Experiment Settings and Throughput

| Script | # GPUs | # GPUs for Actor | # GPUs for Rollout | # GPUs for Async Reward | Batch Size | Images per Prompt | LR | Throughput (images/GPU/s) | Time per Step (s) |
|--------|--------|------------------|--------------------|-------------------------|------------|-------------------|----|-----------------------|-------------------|
| `run_qwen_image_ocr_lora.sh` | 4 | 4 | 4 | 0 (sync) | 32 | 16 | 3e-4 | 0.305 | 420 |
| `run_qwen_image_ocr_lora_async_reward.sh` | 5 | 4 | 4 | 1 | 32 | 16 | 3e-4 | 0.280 | 360 |

### Training - Zero Standard Deviation Ratio and Reward Curve

<div align="center">
<img width="600" alt="LoRA FlowGRPO OCR training zero standard deviation ratio and reward curve" src="https://github.com/user-attachments/assets/256cb424-5e2c-4ba5-8c24-3d1b86ac7860" />
</div>

- `qwen_image_ocr_lora`: sync reward, 4 GPUs (`run_qwen_image_ocr_lora.sh`)
- `qwen_image_ocr_lora_async_reward`: async reward on a dedicated 5th GPU (`run_qwen_image_ocr_lora_async_reward.sh`)

### Validation Reward Curve

Evaluated with `trainer.val_before_train=True`:

<div align="center">
<img width="600" alt="LoRA FlowGRPO OCR validation reward curve" src="https://github.com/user-attachments/assets/1094beaf-fed9-4661-8a6a-1c3983150648" />
</div>

- `qwen_image_ocr_lora`: sync reward, 4 GPUs (`run_qwen_image_ocr_lora.sh`)
- `qwen_image_ocr_lora_async_reward`: async reward on a dedicated 5th GPU (`run_qwen_image_ocr_lora_async_reward.sh`)

> **Note:** Reward curves may differ from the references above mainly due to rollout-side stochasticity: diffusion rollouts sample random latents/noise, and the example scripts do not fix the data seed, so prompt ordering can vary between runs.

## FlowGRPO: non-CFG Full Model Training on Qwen-Image OCR

> Experiments used NVIDIA H200 GPUs, lr 3e-5, clip_ratio 1e-5, optimizer state fp32. The other parameters are consistent with the LoRA setting.

> Note that the initial reward is expected to be low for non-CFG full model training.

### Full-Model Experiment Settings and Throughput

| Script | # GPUs | # GPUs for Actor | # GPUs for Rollout | # GPUs for Async Reward | Batch Size | Images per Prompt | LR | Throughput (images/GPU/s) | Time per Step (s) |
|--------|--------|------------------|--------------------|-------------------------|------------|-------------------|----|-----------------------|-------------------|
| `run_qwen_image_ocr.sh` | 4 | 4 | 4 | 0 (sync) | 32 | 16 | 3e-5 | 0.510 | 250 |

### Full-Model Training - Zero Standard Deviation Ratio and Reward Curve

<div align="center">
<img width="600" alt="Full Model FlowGRPO OCR training zero standard deviation ratio and reward curve" src="https://github.com/user-attachments/assets/ee5db957-f3b0-44e4-8054-b80ddac02bcb" />
</div>

### Training - Clip Fraction

<div align="center">
<img width="600" alt="Full Model FlowGRPO OCR training Clip Fraction" src="https://github.com/user-attachments/assets/b5d27aae-337b-43bf-8228-1678e71673a5" />
</div>

### Full-Model Validation Reward Curve

<div align="center">
<img width="600" alt="Full Model FlowGRPO OCR validation reward curve" src="https://github.com/user-attachments/assets/5ed8fd76-6f1b-4c80-aa43-af905e58d722" />
</div>

## FlowGRPO non-CFG Full Model: VeOmni vs FSDP1 Backend (same config)

> Apples-to-apples comparison: the **VeOmni** and **FSDP1** actor engines run the *same* FlowGRPO recipe — same algorithm, data, and hyper-parameters — on the *same* hardware (64 × NVIDIA H100), differing only in the training engine. lr 3e-5, clip_ratio 1e-5, optimizer state fp32; other parameters match the LoRA setting.

- **FSDP1** — `run_qwen_image_ocr.sh`
- **VeOmni** — `run_qwen_image_ocr_veomni.sh` (see the [install guide](../start/install.md) "Optional engine backends")

### Settings and Throughput

| Backend | Script | GPU name | # GPUs | # GPUs for Actor | # GPUs for Rollout | # GPUs for Async Reward | Batch Size | Images per Prompt | LR | Throughput (images/GPU/s) | Time per Step (s) |
|---------|--------|--------|--------|------------------|--------------------|-------------------------|------------|-------------------|----|-----------------------|-------------------|
| VeOmni | `run_qwen_image_ocr_veomni.sh` | H100 | 64 | 64 | 64 | 0 (sync) | 32 | 16 | 3e-5 | 0.079 | 100 |
| FSDP1 | `run_qwen_image_ocr.sh` | H100 | 64 | 64 | 64 | 0 (sync) | 32 | 16 | 3e-5 | 0.077 | 105 |

> **Note**: VeOmni and FSDP1 run with `actor_rollout_ref.actor.veomni_config.param_offload=False`, `actor_rollout_ref.actor.veomni_config.optimizer_offload=True`, and `SP=1`.

### Full-Model Training - Zero Standard Deviation Ratio and Reward Curve

<img width="1221" height="465" alt="zero_std_ratio" src="https://github.com/user-attachments/assets/3ba4db3e-ea26-4528-893f-8fb00feb7fad" />

<img width="1221" height="465" alt="reward_mean" src="https://github.com/user-attachments/assets/ac828e51-a99d-4b8a-92fb-2c5d3dcbf08a" />

### Training - Clip Fraction

<img width="1221" height="465" alt="pg_clip" src="https://github.com/user-attachments/assets/467819cf-f7f5-45b4-bf11-1af359900b0d" />

### Full-Model Validation Reward Curve

<img width="1221" height="465" alt="mean" src="https://github.com/user-attachments/assets/2a85d8d8-703e-4975-9b6e-cc6ad3fcda63" />

## FlowDPPO: LoRA Training on Qwen-Image OCR

> All experiments used NVIDIA H200 GPUs, LoRA rank 64, `ppo_micro_batch_size_per_gpu` 16, and the full 1k validation set. Training images per step = batch size × images per prompt = 32 × 16 = 512.

| Script | # GPUs | # GPUs for Actor | # GPUs for Rollout | # GPUs for Async Reward | Batch Size | Images per Prompt | LR | Throughput (images/GPU/s) | Time per Step (s) |
|--------|--------|------------------|--------------------|-------------------------|------------|-------------------|----|-----------------------|-------------------|
| `run_qwen_image_ocr_lora.sh` | 4 | 4 | 4 | 0 (sync) | 32 | 16 | 3e-4 | 0.240 | 540 |

<div align="center">
<img width="600" alt="FlowDPPO LoRA OCR training zero standard deviation ratio and reward curve" src="https://github.com/user-attachments/assets/7e3405bb-d609-42b0-b563-58e81d428c48" />
</div>

### LoRA Validation Reward Curve

<div align="center">
<img width="600" alt="FlowDPPO LoRA OCR training validation curve" src="https://github.com/user-attachments/assets/bd44e0f6-c0f1-4d0d-b5ea-8bade1e9a1c5" />
</div>

## DiffusionNFT: non-CFG LoRA Training on Qwen-Image OCR

> All experiments used NVIDIA H200 GPUs, LoRA rank 64, `ppo_micro_batch_size_per_gpu` 16, and the full 1k validation set. Training images per step = batch size × images per prompt = 32 × 16 = 512.

| Script | # GPUs | # GPUs for Actor | # GPUs for Rollout | # GPUs for Async Reward | Batch Size | Images per Prompt | LR | Throughput (images/GPU/s) | Time per Step (s) |
|--------|--------|------------------|--------------------|-------------------------|------------|-------------------|----|-----------------------|-------------------|
| `run_qwen_image_ocr_lora.sh` | 4 | 4 | 4 | 0 (sync) | 24 | 12 | 3e-4 | 0.175 | 550 |

<div align="center">
<img width="600" alt="DiffusionNFT LoRA OCR training zero standard deviation ratio and reward curve" src="https://github.com/user-attachments/assets/afed8370-c37e-4f7b-9bba-83ef4b28b6c7" />
</div>

### LoRA Validation Reward Curve

<div align="center">
<img width="600" alt="DiffusionNFT LoRA OCR training validation curve" src="https://github.com/user-attachments/assets/9cc0e639-58c7-4ef7-ab8a-ee8e8aef2d53" />
</div>
