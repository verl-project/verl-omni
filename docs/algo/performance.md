(performance)=
# Performance Reference

Last updated: 09/09/2026

Below are reference benchmark results for VeRL-Omni training runs.

## DAPO Phase 1: LoRA Training on Qwen3-Omni Thinker AVQA

This reference uses the {doc}`Thinker DAPO Phase-1 recipe <../examples/dapo_trainer>`:
vanilla token-level clipping with GRPO advantages, without dynamic sampling or
overlong reward shaping. It is a single-seed run on AVQA, not the full DAPO recipe.

### Experiment Settings

| Setting | Value |
|---------|-------|
| Model | `Qwen3-Omni-30B-A3B-Instruct`, Thinker only |
| Hardware / actor | 4 × NVIDIA A800, FSDP2 LoRA |
| Dataset / reward | AVQA; `naive` reward manager with `choice_reward` |
| LoRA rank / alpha | 32 / 64 |
| Learning rate / train batch size | `3e-6` / 128 prompts |
| Rollout samples / tensor parallel size | `n=16` / 2 |
| Policy loss / clipping / aggregation | `vanilla` / `0.2`–`0.28` / `token-mean` |
| Advantage estimator / KL | GRPO / disabled |
| Data seed | 42 |
| Validation | All 1,911 AVQA examples, before training and every 10 steps |
| Validation decoding | Greedy: `n=1`, `temperature=0`, `top_p=1.0`, `top_k=-1` |

### Long-Run Validation Reward

The run was resumed from step 50 and continued through step 235 without changing
or tuning the training configuration. The recorded full-validation curve extends
through step 220; each point is the mean choice reward over the same validation set.

<div align="center">
<img width="800" alt="Single-seed AVQA validation reward from step 0 to 220: 0.728414 initially, best 0.869702 at step 180, and 0.866039 at step 220" src="https://raw.githubusercontent.com/WenzheWang/verl-omni/e956f8f5f15b0124681ac1f8d2dc0e4cacdc02fb/.github/pr-assets/456/qwen3-omni-thinker-dapo-avqa-validation-long-curve.png" />
</div>

| Checkpoint | Step | Validation reward |
|------------|------|-------------------|
| Initial | 0 | 0.728414 |
| Initial run end | 50 | 0.824176 |
| Best observed | 180 | 0.869702 |
| Latest recorded validation | 220 | 0.866039 |

Validation reward improved by 14.1287 percentage points from the initial to the
best checkpoint. At steps 150–220, it fluctuated between `0.858189` and `0.869702`.
The curve shows observed values without smoothing or interpolation; it does not
reach the GSPO+LoRA reference value of `0.88` in this window.

The GSPO reference uses a different validation sampling configuration
(`temperature=1.0`, `top_p=0.7`, `top_k=-1`). Re-evaluating the retained
step-50/150/200 checkpoints with that tuple gave `0.821559` / `0.856096` /
`0.859759`, versus `0.824176` / `0.860283` / `0.859759` with greedy decoding.
These single-seed, single-decode observations do not establish GSPO performance
parity or a systematic advantage for either validation temperature.

The experiment used implementation commit
[`c4671984`](https://github.com/verl-project/verl-omni/commit/c4671984d6975f0ccbdae77ee6c87c105b1db55e).
See the [long-run evidence and configuration comparison](https://github.com/verl-project/verl-omni/pull/456#issuecomment-5534647748)
for the run record; later documentation updates do not change its tested revision.

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

Reference wandb curve [here](https://wandb.ai/andyzhou/VeRL-Omni-demo/runs/8p8y9olb).

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

Reference wandb curve [here](https://wandb.ai/andyzhou/VeRL-Omni-demo/runs/djrzzibt). 

<div align="center">
<img width="600" alt="DiffusionNFT LoRA OCR training zero standard deviation ratio and reward curve" src="https://github.com/user-attachments/assets/afed8370-c37e-4f7b-9bba-83ef4b28b6c7" />
</div>

### LoRA Validation Reward Curve

<div align="center">
<img width="600" alt="DiffusionNFT LoRA OCR training validation curve" src="https://github.com/user-attachments/assets/9cc0e639-58c7-4ef7-ab8a-ee8e8aef2d53" />
</div>

## GSPO OPD: Qwen3-Omni-30B-A3B training on 32xNPU (2 x Atlas 800T A3)

> Experiments used Atlas 800T A3 NPUs, LoRA rank 32 applied to attention linear modules, `train_batch_size=128` and rollout `n=16` per prompt, and the full 2k validation set.

> We add Gaussian noise (σ = 0.25·∥W∥) to the weights of Qwen3-Omni-30B-A3B-Instruct as the student model, while keeping the original un-noised model as the teacher. Both models are served by vllm_omni in AR mode. We compare the proposed OPD (Offline Preference Distillation) against a standard GSPO baseline, which trains the same noised Qwen3-Omni model directly under identical settings.

| Script | # NPUs | # NPUs for Actor | # NPUs for Rollout | # NPUs for Teacher Model | # NPUs for Async Reward | Batch Size | Rollouts per Prompt | LR | Time per Step (s) |
|--------|--------|------------------|--------------------|--------------------------|-------------------------|------------|-------------------|----|-------------------|
| `run_qwen3_omni_thinker_gspo_lora_mmk12_v1_opd_npu.sh` | 32 | 16 | 16 | 16 | 0 (sync) | 128 | 16 | 3e-6 | 1800 |

### Training Reward Curve (OPD vs GSPO)
<div align="center">
<img width="763" height="261" alt="Training reward curve of GSPO OPD vs GSPO" src="https://github.com/user-attachments/assets/7fd1dc67-85c0-4adc-9f7a-10cdbec005d0" />
</div>

### Validation Reward Curve (OPD vs GSPO)

<div align="center">
<img width="763" height="261" alt="Validation curve of GSPO OPD vs GSPO" src="https://github.com/user-attachments/assets/fe25b199-10e1-4a90-b4e5-18248d40d3fa" />
</div>
