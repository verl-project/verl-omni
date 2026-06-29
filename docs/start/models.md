# Supported Models

Last updated: 06/26/2026.

VeRL-Omni supports RL post-training for generative models across image, video,
audio, and omni modalities. This page catalogues every model with a ready-to-run
example, its architecture and pipeline details, supported trainers, and hardware
requirements.

---

## Diffusion Image Models

### Qwen-Image

| Property | Detail |
|----------|--------|
| **Hugging Face ID** | `Qwen/Qwen-Image` |
| **Architecture** | MM-DiT (Multi-Modal Diffusion Transformer) with joint image-text attention |
| **Modality** | Text → Image |
| **Pipeline** | Flow-matching with True CFG and distilled guidance embedding |
| **Text encoder** | Qwen2-style tokenizer + T5-style encoder |
| **Resolution** | Variable (512×512, 1024×1024) |

**Supported trainers:**

| Trainer | Example script | GPU config |
|---------|---------------|------------|
| Flow-GRPO (LoRA) | `examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora.sh` | 4×GPU |
| Flow-GRPO (full) | `examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr.sh` | 4×H200 |
| Flow-GRPO (async) | `examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_async_reward.sh` | 5×GPU |
| Flow-GRPO (multi-node) | `examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_multi_node.sh` | 2×4 GPU |
| Flow-GRPO (SP=2) | `examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_sp2.sh` | 4×GPU |
| Flow-GRPO (rollout-corr) | `examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_rollout_corr.sh` | 4×GPU |
| Flow-GRPO (VeOmni) | `examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_veomni.sh` | 64×H100 |
| Flow-GRPO (NPU) | `examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_npu.sh` | 8×NPU |
| Flow-DPPO | `examples/flowdppo_trainer/qwen_image/run_qwen_image_ocr_lora.sh` | 4×GPU |
| GRPO-Guard | `examples/grpoguard_trainer/qwen_image/run_qwen_image_ocr_lora.sh` | 4×GPU |
| Mix-GRPO | `examples/mixgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_mixgrpo.sh` | 4×GPU |
| Diffusion-DPO | `examples/dpo_trainer/qwen_image/run_qwen_image_online_dpo_lora.sh` | 4×GPU |
| DiffusionNFT | `examples/diffusionnft_trainer/qwen_image/run_qwen_image_ocr_lora.sh` | 4×GPU |

**Reward model:** `Qwen/Qwen3-VL-8B-Instruct` (OCR VLM judge, TP=4 colocated).

### Stable Diffusion 3.5 Medium

| Property | Detail |
|----------|--------|
| **Hugging Face ID** | `stabilityai/stable-diffusion-3.5-medium` |
| **Architecture** | MM-DiT with dual CLIP + T5 text encoders |
| **Modality** | Text → Image |
| **Pipeline** | Flow-matching (distilled guidance only, no True CFG) |
| **Text encoder** | CLIP-L, CLIP-G, T5-XXL |
| **Default resolution** | 384×384 |
| **Chat template** | Custom — extracts raw user content only (no system prompt) |

**Supported trainers:**

| Trainer | Example script | GPU config |
|---------|---------------|------------|
| Flow-GRPO (LoRA) | `examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora.sh` | 3×GPU (2 actor+rollout, 1 reward) |
| Diffusion-DPO (offline) | `examples/dpo_trainer/sd35/run_sd35_medium_offline_dpo_lora.sh` | 3×GPU |

**Reward model:** `Qwen/Qwen2.5-VL-3B-Instruct` (OCR VLM judge, TP=1, dedicated pool).

### FLUX

| Property | Detail |
|----------|--------|
| **Architecture** | MM-DiT (flow-matching) |
| **Modality** | Text → Image |
| **Pipeline** | Flow-matching with distilled guidance |
| **Text encoder** | T5 + CLIP |

FLUX is supported via the same Flow-GRPO MM-DiT pipeline adapter pattern.
Adapt from the Qwen-Image or SD3.5 configuration — no dedicated example script
is provided, but the framework supports it natively.

---

## Diffusion Video Models

### Wan2.2-TI2V-5B

| Property | Detail |
|----------|--------|
| **Hugging Face ID** | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` |
| **Architecture** | Wan-style DiT with separate self-attention and cross-attention |
| **Modality** | Text → Video |
| **Pipeline** | Flow-matching with spatiotemporal latents |
| **Text encoder** | T5 |
| **Latent stream** | Spatiotemporal video latents |
| **Prompt stream** | Text-encoder tokens (cross-attention KV) |
| **SDE variants** | `dance_sde` (recommended, score-based), `sde` (FlowGRPO), `cps` (consistency-preserving) |

**Supported trainers:**

| Trainer | Example script | GPU config |
|---------|---------------|------------|
| DanceGRPO (HPSv3) | `examples/dancegrpo_trainer/wan22/run_wan22_5b_t2v_hpsv3_npu.sh` | 8×NPU (Ascend 800T A2) |

**Reward model:** HPSv3 (Human Preference Score v3) — local safetensors checkpoint
placed at `$WORKSPACE/CKPT/HPSv3/HPSv3.safetensors`.

The HPSv3 reward is the only validated configuration. Other reward functions
(e.g. OCR, aesthetic score) can be plugged in by changing
`reward.custom_reward_function`.

---

## Unified Multimodal Models

### BAGEL

| Property | Detail |
|----------|--------|
| **Architecture** | Unified multimodal understanding + generation |
| **Modality** | Text + Image (understand and generate) |
| **Deploy config** | `examples/flowgrpo_trainer/bagel/bagel_deploy_config.yaml` |
| **Rollout** | vLLM-Omni with per-stage YAML for engine memory/batching control |

**Supported trainers:**

| Trainer | Example script | GPU config |
|---------|---------------|------------|
| Flow-GRPO (LoRA, OCR) | `examples/flowgrpo_trainer/bagel/run_bagel_ocr_lora.sh` | 4×GPU |
| Flow-GRPO (LoRA, PickScore) | `examples/flowgrpo_trainer/bagel/run_bagel_pickscore_lora.sh` | 4×GPU |

BAGEL uses a per-stage deploy YAML that overrides top-level vLLM engine arguments
— tune `gpu_memory_utilization` and batch sizes directly in the stage config file.

---

## Omni-Modality Models

### Qwen3-Omni-30B-A3B Thinker

| Property | Detail |
|----------|--------|
| **Hugging Face ID** | `Qwen/Qwen3-Omni-30B-A3B-Instruct` |
| **Architecture** | Omni-modality Thinker with Mixture-of-Experts (30B total, 3B active) |
| **Modality** | Text + Image + Audio + Video (understand and generate) |
| **Trainer type** | GSPO — Group Sampling Policy Optimization (verl-native PPO-style) |
| **FSDP** | Full FSDP with LoRA (rank 64), param and optimizer CPU offload |
| **Rollout** | vLLM-Omni TP=4 colocated on the same GPUs as the FSDP actor |
| **Stage config** | `examples/gspo_trainer/qwen3_omni/config/qwen3_omni_thinker_only.yaml` (`gpu_memory_utilization=0.4`) |
| **External module** | `verl_omni.models.transformers.qwen3_omni_thinker` |

For version requirements and detailed setup instructions, see
[`examples/gspo_trainer/README.md`](../../examples/gspo_trainer/README.md).

**Supported trainers:**

| Trainer | Example script | GPU config |
|---------|---------------|------------|
| GSPO (math) | `examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora.sh` | 4×H100/H200 80GB |

The actor (FSDP, 30B + LoRA r=64 with offloading) and vLLM-Omni rollout (TP=4)
colocate on the same 4 GPUs. `gpu_memory_utilization` is kept at `0.4` in the
stage config to leave headroom for the FSDP actor.

---

## Model Architecture Summary

| Model | Architecture | Text encoder |
|-------|-------------|-------------|
| Qwen-Image | MM-DiT | Qwen2 + T5 |
| SD3.5 Medium | MM-DiT | CLIP-L + CLIP-G + T5 |
| Wan2.2-TI2V-5B | Wan DiT | T5 |
| BAGEL | Unified MM | — |
| Qwen3-Omni-30B | Omni MoE | Qwen3 |
| FLUX | MM-DiT | T5 + CLIP |

---

## Reward Models

| Reward model | HF ID / Source | Modality | Used by | Deployment |
|-------------|---------------|----------|---------|------------|
| Qwen3-VL-8B-Instruct | `Qwen/Qwen3-VL-8B-Instruct` | Vision-Language | Qwen-Image (all trainers) | vLLM, TP=4, colocated |
| Qwen2.5-VL-3B-Instruct | `Qwen/Qwen2.5-VL-3B-Instruct` | Vision-Language | SD3.5 (Flow-GRPO) | vLLM, TP=1, dedicated pool |
| HPSv3 | Local `.safetensors` | Vision (aesthetic) | Wan2.2 (DanceGRPO) | Local safetensors load |
| HTTP scorer | External HTTP service | Any | Any model | Gunicorn/Flask, pickle protocol |
| JPEG incompressibility | Rule-based | Image stats | Any diffusion model | No model process needed |

For end-to-end instructions on setting up each reward, see the respective
trainer's README in `examples/`.

---

## Which Trainer for Which Model?

| Algorithm | Qwen-Image | SD3.5 | Wan2.2 | BAGEL | Qwen3-Omni | FLUX |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| Flow-GRPO | ✅ | ✅ | — | ✅ | — | ✅ |
| Flow-DPPO | ✅ | — | — | — | — | — |
| GRPO-Guard | ✅ | — | — | — | — | — |
| Mix-GRPO | ✅ | — | — | — | — | — |
| DanceGRPO | — | — | ✅ | — | — | — |
| Diffusion-DPO | ✅ | ✅ | — | — | — | — |
| DiffusionNFT | ✅ | — | — | — | — | — |
| GSPO | — | — | — | — | ✅ | — |
