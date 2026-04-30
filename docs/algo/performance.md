(performance)=
# Performance Reference

Last updated: 04/23/2026

Below are reference benchmark results for VeRL-Omni training runs.

## FlowGRPO: Qwen-Image OCR

> All experiments used NVIDIA H800 GPUs, LoRA rank 64, `ppo_micro_batch_size_per_gpu` 16, and the full 1k validation set. Training images per step = batch size × images per prompt = 32 × 16 = 512.

### Experiment Settings and Throughput

| Script | # GPUs | # GPUs for Actor | # GPUs for Rollout | # GPUs for Async Reward | Batch Size | Images per Prompt | LR | Throughput (images/GPU/s) | Time per Step (s) |
|--------|--------|------------------|--------------------|-------------------------|------------|-------------------|----|-----------------------|-------------------|
| `run_qwen_image_ocr_lora.sh` | 4 | 4 | 4 | 0 (sync) | 32 | 16 | 3e-4 | 0.305 | 420 |
| `run_qwen_image_ocr_lora_async_reward.sh` | 5 | 4 | 4 | 1 | 32 | 16 | 3e-4 | 0.280 | 360 |

### Validation Reward Curve

Evaluated with `trainer.val_before_train=True`:

<div align="center">
<img width="600" alt="FlowGRPO OCR validation reward curve" src="https://github.com/user-attachments/assets/1094beaf-fed9-4661-8a6a-1c3983150648" />
</div>

- `qwen_image_ocr_lora`: sync reward, 4 GPUs (`run_qwen_image_ocr_lora.sh`)
- `qwen_image_ocr_lora_async_reward`: async reward on a dedicated 5th GPU (`run_qwen_image_ocr_lora_async_reward.sh`)

> **Note:** Due to inherent randomness in the training process, your reward curve may differ from the reference above.
