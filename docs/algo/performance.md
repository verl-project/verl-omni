(performance)=
# Performance Reference

Last updated: 05/13/2026

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

> Experiments used NVIDIA H200 GPUs, `ppo_micro_batch_size_per_gpu` 8, lr 1e-5, clip_ratio 1e-5, image resolution 384x384, optimizer buffer fp32. The other parameters are consistent with the LoRA setting.

> Note that the initial reward is expected to be low for non-CFG full model training.

### Training - Zero Standard Deviation Ratio and Reward Curve

<div align="center">
<img width="600" alt="Full Model FlowGRPO OCR training zero standard deviation ratio and reward curve" src="https://github.com/user-attachments/assets/573c3ef3-2ab6-478f-b6f5-10a344628d13" />
</div>

### Validation Reward Curve

<div align="center">
<img width="600" alt="Full Model FlowGRPO OCR validation reward curve" src="https://github.com/user-attachments/assets/2cabd94f-2aff-4925-9cab-a8341479e82c" />
</div>
