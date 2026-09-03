# Supported Models

Last updated: 08/27/2026.

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

### Qwen-Image-Edit

| Property | Detail |
|----------|--------|
| **Hugging Face ID** | `Qwen/Qwen-Image-Edit-2511` |
| **Architecture** | MM-DiT (same family as Qwen-Image) with concat-conditioned I2I — denoise-target latent plus condition-image latent on the image stream |
| **Modality** | Text + Image → Image (I2I / image edit) |
| **Pipeline** | Flow-matching with True CFG; registered as `QwenImageEditPlusPipeline` |
| **Text encoder** | Qwen2.5-VL vision-language encoder (edit instruction + condition image) |
| **Default resolution** | 512×512 (square in the example recipe) |
| **Condition images** | Exactly one per sample; condition aspect ratio must match the target output |

The training adapter is `QwenImageEditPlusFlowGRPO` (`DiffusionI2IModelBase` +
Qwen-Image T2I helpers). Rollout uses vLLM-Omni
`QwenImageEditPlusPipelineWithLogProb`. A replacement checkpoint must set
`model_index.json::_class_name` to `QwenImageEditPlusPipeline`. The older
`QwenImageEditPipeline` architecture is not supported.

For dataset layout, launch overrides, and sequence-parallel constraints, see
[Examples - Qwen-Image-Edit-2511 FlowGRPO training](../../examples/flowgrpo_trainer/qwen_image_edit/README.md).

**Supported trainers:**

| Trainer | Example script | GPU config |
|---------|---------------|------------|
| Flow-GRPO (LoRA) | `examples/flowgrpo_trainer/qwen_image_edit/run_qwen_image_edit_lora.sh` | 8×GPU |

**Reward model:** PickScore (`yuvalkirstain/PickScore_v1`) — CLIP preference
scorer, async workers (default 4). PickScore measures instruction/image
alignment and does not directly enforce source-image preservation.

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
| Flow-GRPO (DiNa-LRM) | `examples/flowgrpo_trainer/sd35/run_sd35_medium_drm_lora.sh` | 8×GPU (7 actor+rollout, 1 latent-reward server) |
| Diffusion-DPO (offline) | `examples/dpo_trainer/sd35/run_sd35_medium_offline_dpo_lora.sh` | 3×GPU |
| [DiffusionOPD](../algo/diffusion_opd.md) (single teacher, OCR) | `examples/diffusionopd_trainer/sd35/run_sd35_medium_ocr_distill.sh` | 3×GPU (2 actor+rollout+teacher, 1 reward) |
| [DiffusionOPD](../algo/diffusion_opd.md) (multi-teacher / MOPD) | `examples/diffusionopd_trainer/sd35/run_sd35_medium_mopd_distill.sh` | 3×GPU (2 actor+rollout+teachers, 1 reward) |

**Reward model:** `Qwen/Qwen2.5-VL-3B-Instruct` (OCR VLM judge, TP=1, dedicated pool) for Flow-GRPO OCR and DiffusionOPD. The DiNa-LRM recipe scores clean latents over HTTP instead of decoding images; see [SD3.5 FlowGRPO with a latent reward model](../examples/flowgrpo_trainer_sd35_drm.md). DiffusionOPD monitors OCR (and PickScore on the mixed-task recipe) but does not put those scores in the loss — see [Diffusion On-Policy Distillation](../algo/diffusion_opd.md).

---

## Diffusion Video and Audio Models

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
| DanceGRPO (HPSv3) | `examples/dancegrpo_trainer/wan22/run_wan22_5b_t2v_hpsv3_auto.sh` | 8×GPU or 16×NPU (auto-detect) |

**Reward model:** HPSv3 (Human Preference Score v3) — local safetensors checkpoint
placed at `$WORKSPACE/CKPT/HPSv3/HPSv3.safetensors`.

The HPSv3 reward is the only validated configuration. Other reward functions
(e.g. OCR, aesthetic score) can be plugged in by changing
`reward.custom_reward_function`.

### LTX-2.3

| Property | Detail |
|----------|--------|
| **Hugging Face ID** | `dg845/LTX-2.3-Diffusers` |
| **Architecture** | LTX-2 DiT; checkpoint `_class_name` is `LTX2Pipeline` (rollout uses vLLM-Omni `LTX23Pipeline`) |
| **Modality** | Text → Video + Audio |
| **Pipeline** | Flow-matching with joint audio-video CPS transitions |
| **Default recipe** | `sde_window_size=3`, `sde_window_range=[0,10]`, `sde_contiguous=False` |

For dataset layout and launch overrides, see
[Examples - LTX-2.3 FlowGRPO](../../examples/flowgrpo_trainer/ltx2/README.md).

**Supported trainers:**

| Trainer | Example script | GPU config |
|---------|---------------|------------|
| Flow-GRPO (LoRA) | `examples/flowgrpo_trainer/ltx2/run_ltx2_3_t2av_lora.sh` | 8×GPU (TP=2) |
| Flow-GRPO (LoRA, NPU) | `examples/flowgrpo_trainer/ltx2/run_ltx2_3_t2av_lora_npu.sh` | 16×NPU (TP=4) |

**Reward models:** CLAP (`laion/larger_clap_general`) and ImageBind (local
`.pth`, CC-BY-NC-SA 4.0) for audio-video alignment.

### MiniMax-H3

| Property | Detail |
|----------|--------|
| **Checkpoint** | Local MiniMax-H3 repo root with `FL2VA/` (vLLM-Omni rollout) and `transformer/` (Diffusers `MiniMaxH3Transformer3DModel` for FSDP) |
| **Architecture** | MiniMax H3 transformer (CFG-distilled; no negative prompts) |
| **Modality** | Text → Video + Audio (T2VA); Text + Image → Video + Audio (FL2VA) |
| **Pipeline** | Online DiffusionNFT with joint video and audio rollouts |
| **Agent loop** | `minimax_h3_diffusion_single_turn_agent` (tokenizes text once for the H3 text encoder) |

FlowGRPO for MiniMax-H3 is still WIP. For checkpoint layout, data prep, and
Diffusers pin, see
[Examples - MiniMax-H3 DiffusionNFT](../../examples/diffusionnft_trainer/minimax_h3/README.md).

**Supported trainers:**

| Trainer | Example script | GPU config |
|---------|---------------|------------|
| DiffusionNFT (T2VA LoRA) | `examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_t2va_lora.sh` | 8×GPU (TP=2) |
| DiffusionNFT (FL2VA LoRA) | `examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_fl2va_lora.sh` | 8×GPU (TP=4) |

**Reward models:** CLAP and ImageBind (audio-video alignment), same pair as LTX-2.3.

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
| **Trainer type** | GSPO (V1 sync via `verl_omni.trainer.main_omni`) and offline DPO (`algorithm.sample_source=offline`) |
| **FSDP** | FSDP2 with LoRA (rank 32 for V1), param and optimizer CPU offload |
| **Rollout** | vLLM-Omni TP=2 colocated on the same GPUs as the FSDP actor |
| **Stage config** | Auto-generated deploy config via `+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="qwen3_omni_moe"` |

For version requirements and detailed setup instructions, see
[Examples - Qwen3-Omni Thinker GSPO Trainer](../../examples/gspo_trainer/README.md).

**Supported trainers:**

| Trainer | Example script | GPU config |
|---------|---------------|------------|
| GSPO (text) | `examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_v1.sh` | 4×H100/H200 80GB |
| GSPO (image) | `examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_mmk12_v1.sh` | 4×H100/H200 80GB |
| GSPO (AVQA, NPU) | `examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_npu_avqa_v1.sh` | 16×NPU (Atlas 800T A3) |
| GSPO (full, NPU) | `examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_npu.sh` | 16×NPU (Atlas 800T A3) |
| Offline DPO (LoRA) | `examples/dpo_trainer/qwen3_omni/qwen3_omni/run_qwen3_omni_omni_preference_lora.sh` | 4×H800 |

The GSPO actor (FSDP2, 30B + LoRA r=32 with offloading) and vLLM-Omni rollout (TP=2)
colocate on the same 4 GPUs. The rollout deploy config is auto-generated from
`pipeline_name=qwen3_omni_moe` — tune rollout memory/batching through standard
verl CLI overrides (e.g. `actor_rollout_ref.rollout.gpu_memory_utilization=0.4`)
rather than a separate per-stage YAML file. Offline DPO reads Omni-Preference
parquet pairs and does not start rollout or reward workers.

---

## Model Architecture Summary

| Model | Architecture | Text encoder |
|-------|-------------|-------------|
| Qwen-Image | MM-DiT | Qwen2 + T5 |
| Qwen-Image-Edit | MM-DiT (I2I concat) | Qwen2.5-VL |
| SD3.5 Medium | MM-DiT | CLIP-L + CLIP-G + T5 |
| Boogu-Image | Double/single-stream DiT | Qwen3-VL |
| Wan2.2-TI2V-5B | Wan DiT | T5 |
| LTX-2.3 | LTX-2 DiT | LTX text encoder |
| MiniMax-H3 | MiniMax H3 transformer | H3 text encoder |
| BAGEL | Unified MM | — |
| Qwen3-Omni-30B | Omni MoE | Qwen3 |

---

## Reward Models

| Reward model | HF ID / Source | Modality | Used by | Deployment |
|-------------|---------------|----------|---------|------------|
| Qwen3-VL-8B-Instruct | `Qwen/Qwen3-VL-8B-Instruct` | Vision-Language | Qwen-Image (all trainers) | vLLM, TP=4, colocated |
| Qwen2.5-VL-3B-Instruct | `Qwen/Qwen2.5-VL-3B-Instruct` | Vision-Language | SD3.5 (Flow-GRPO, DiffusionOPD) | vLLM, TP=1, dedicated pool |
| PickScore | `yuvalkirstain/PickScore_v1` | Vision (preference) | Qwen-Image-Edit (Flow-GRPO), BAGEL (PickScore recipe), SD3.5 (MOPD monitor) | Local CLIP load, async workers |
| HPSv3 | Local `.safetensors` | Vision (aesthetic) | Wan2.2 (DanceGRPO) | Local safetensors load |
| CLAP | `laion/larger_clap_general` | Audio | LTX-2.3 (Flow-GRPO), MiniMax-H3 (DiffusionNFT) | Local transformers load |
| ImageBind | Local `.pth` | Audio + Video | LTX-2.3 (Flow-GRPO), MiniMax-H3 (DiffusionNFT) | Local ImageBind package (CC-BY-NC-SA 4.0) |
| DiNa-LRM | HTTP latent scorer | Diffusion latents | SD3.5 (Flow-GRPO DRM) | Separate `diffusion-rm` process, safetensors HTTP |
| HTTP scorer | External HTTP service | Any | Any model | Gunicorn/Flask, pickle protocol |
| JPEG incompressibility | Rule-based | Image stats | Any diffusion model | No model process needed |

For end-to-end instructions on setting up each reward, see the respective
trainer's README in `examples/`.

---

## Which Trainer for Which Model?

| Algorithm | Qwen-Image | Qwen-Image-Edit | SD3.5 | Wan2.2 | LTX-2.3 | MiniMax-H3 | BAGEL | Qwen3-Omni |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Flow-GRPO | ✅ | ✅ | ✅ | — | ✅ | WIP | ✅ | — |
| Flow-DPPO | ✅ | — | — | — | — | — | — | — |
| GRPO-Guard | ✅ | — | — | — | — | — | — | — |
| Mix-GRPO | ✅ | — | — | — | — | — | — | — |
| DanceGRPO | — | — | — | ✅ | — | — | — | — |
| DPO | ✅ | — | ✅ | — | — | — | — | ✅ |
| DiffusionNFT | ✅ | — | — | — | — | ✅ | — | — |
| [DiffusionOPD](../algo/diffusion_opd.md) (incl. MOPD) | — | — | ✅ | — | — | — | — | — |
| GSPO | — | — | — | — | — | — | — | ✅ |

HunyuanImage-3.0 (MixGRPO / SRPO) and Qwen3-TTS (DPO / GSPO) appear on the
project README as Planned or WIP and do not yet have a ready-to-run recipe, so
they are omitted from the catalogue above.
