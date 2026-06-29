# FlowGRPO Trainer

This example shows how to post-train `Qwen-Image` with FlowGRPO on an OCR-style image generation task using `vllm-omni` rollout and a visual generative reward model (`Qwen3-VL-8B-Instruct` in this example).

For the full installation and quickstart guide, see `docs/start/flowgrpo_quickstart.md`. For algorithm details and rule-based reward training (e.g. JPEG incompressibility), see `docs/algo/flowgrpo.md`.

## Installation

Follow the [installation guide](../../docs/start/install.md) to set up the base environment, then install the FlowGRPO-specific dependency:

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

```bash
bash examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora.sh
```

GPU training defaults to matched FA3 in config (`attn_backend: _flash_3_varlen_hub`; rollout
`DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN` is set by `main_diffusion` when FA3 is available).
Training falls back to native/SDPA if FA3 deps are unavailable.

Optional KL loss tuning:

- `actor_rollout_ref.actor.use_kl_loss=True`
- `actor_rollout_ref.actor.kl_loss_coef=0.001`

The script runs `python3 -m verl_omni.trainer.main_diffusion` with:

- `algorithm.adv_estimator=flow_grpo`
- `actor_rollout_ref.model.path=Qwen/Qwen-Image`
- `actor_rollout_ref.model.lora_rank=64`
- `actor_rollout_ref.model.lora_alpha=128`
- `actor_rollout_ref.rollout.name=vllm_omni`
- `reward.custom_reward_function.name=compute_score_ocr`
- `trainer.n_gpus_per_node=4`

## Logging

W&B logging is enabled by default in the example script:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
```

The script sets:

```bash
trainer.logger='["console", "wandb"]'
trainer.project_name=flow_grpo
trainer.experiment_name=qwen_image_ocr_lora
```

Override these values on the command line if you want to log under a different project or run name.

### Diffusion-specific metrics

See the [Metrics Documentation](../../docs/start/metrics.md) for a full description of all diffusion-specific training metrics.

## Variants

All example scripts in this directory:

### Qwen-Image

| Variant | Script | GPUs | Notes |
|---------|--------|------|-------|
| LoRA (baseline) | `qwen_image/run_qwen_image_ocr_lora.sh` | 4×GPU | Standard LoRA FlowGRPO + OCR reward |
| LoRA + async reward | `qwen_image/run_qwen_image_ocr_lora_async_reward.sh` | 5×GPU | Dedicated GPU pool for reward model |
| LoRA + SP=2 | `qwen_image/run_qwen_image_ocr_lora_sp2.sh` | 4×GPU | Ulysses sequence parallelism |
| LoRA + FA3 | `qwen_image/run_qwen_image_ocr_lora_fsdp2_fa3.sh` | 4×GPU | FSDP2 with FlashAttention 3 |
| LoRA + rollout corr | `qwen_image/run_qwen_image_ocr_lora_rollout_corr.sh` | 4×GPU | Bypass old-log-prob recompute |
| LoRA + multi-node | `qwen_image/run_qwen_image_ocr_lora_multi_node.sh` | 2×4 GPU | Horizontal scaling across nodes |
| LoRA + NPU | `qwen_image/run_qwen_image_ocr_lora_npu.sh` | 8×NPU | Ascend NPU LoRA training |
| Full model | `qwen_image/run_qwen_image_ocr.sh` | 4×H200 | Non-CFG full-weight training |
| Full + FSDP2 64-card | `qwen_image/run_qwen_image_ocr_fsdp2_64cards.sh` | 64×GPU | Large-scale FSDP2 |
| Full + VeOmni | `qwen_image/run_qwen_image_ocr_veomni.sh` | 64×H100 | VeOmni engine backend |
| Full + VeOmni 64-card | `qwen_image/run_qwen_image_ocr_veomni_64cards.sh` | 64×GPU | VeOmni at scale |
| Full + H200 MFU opt | `qwen_image/run_qwen_image_ocr_h200_mfu_optimized.sh` | H200 | MFU-optimized config |
| Full + NPU | `qwen_image/run_qwen_image_ocr_npu.sh` | 16×NPU | Ascend NPU full model |
| HTTP reward server | `qwen_image/run_qwen_image_ocr_reward_server.sh` | 4×GPU | External HTTP OCR reward service |
| Multi-reward | `qwen_image/run_qwen_image_ocr_multi_reward.sh` | 4×GPU | Multiple reward functions combined |

### Stable Diffusion 3.5 Medium

| Variant | Script | GPUs | Notes |
|---------|--------|------|-------|
| LoRA | `sd35/run_sd35_medium_ocr_lora.sh` | 3×GPU | 2 actor+rollout, 1 reward pool |

### BAGEL

| Variant | Script | GPUs | Notes |
|---------|--------|------|-------|
| LoRA | `bagel/run_bagel_flowgrpo_lora.sh` | Configurable | Per-stage deploy YAML |

### Engine backends

The diffusion trainer defaults to **FSDP2**. To use [VeOmni](https://github.com/ByteDance-Seed/VeOmni) instead, follow the [VeOmni install instructions](../../docs/start/install.md#optional-engine-backends), then run the VeOmni counterpart:

```bash
bash examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_veomni.sh
```

`run_qwen_image_ocr_veomni.sh` mirrors `run_qwen_image_ocr.sh` and differs only in engine-selection overrides:

| Override | FSDP2 (default) | VeOmni |
|---|---|---|
| `diffusion/model_engine=` | _(unset; uses `dp_diffusion`)_ | `veomni_diffusion` |
| `actor_rollout_ref.actor.strategy=` | `fsdp2` | `veomni` |
| Actor engine config block | `actor.fsdp_config.*` | `actor.veomni_config.*` |
| Sequence-parallel field | `fsdp_config.ulysses_sequence_parallel_size` | `veomni_config.ulysses_parallel_size` |
| Ref engine config block | `ref.fsdp_config.*` | `ref.veomni_config.*` |

Do not mix `fsdp_config` and `veomni_config` overrides — `diffusion/model_engine=...` selects the schema, and overrides for the other engine are rejected.

## Performance

> All experiments were conducted on *NVIDIA H800* GPUs using the OCR reward.

The experiment settings and throughputs are shown in the table below.

| Script | Model | Algorithm | Hybrid Engine | # Cards | Reward Fn | # GPUs for Actor | # GPUs for Rollout | # GPUs for Async Reward | Batch Size | `rollout.n` | lr   | # Val Samples | Training Samples per Step | `ppo_micro_batch_size_per_gpu` | Throughput (Samples / GPU / Seconds) | Time per Step (Seconds) |
| --- | --- | --- | --- | --- | --- | --- | --- |-------------------------| --- | --- |------| --- | --- | --- |------------------------------| --------------------------------|
| `qwen_image/run_qwen_image_ocr_lora.sh` | Qwen-Image | Flow-GRPO | True | 4 | qwenvl-ocr-vllm | 4 | 4 | 0 (sync)                | 32 | 16 | 3e-4 | 1k (full set) | 32×16=512 | 16 | 0.305                        | 420 |
| `qwen_image/run_qwen_image_ocr_lora_async_reward.sh` | Qwen-Image | Flow-GRPO | True | 5 | qwenvl-ocr-vllm | 4 | 4 | 1                       | 32 | 16 | 3e-4 | 1k (full set) | 32×16=512 | 16 | 0.280                        | 360 |

- Validation reward curve (evaluated with `trainer.val_before_train=True`):

<div align="center">
<img width="600" alt="2p_comparison" src="https://github.com/user-attachments/assets/1094beaf-fed9-4661-8a6a-1c3983150648" />
<br>
qwen_image_ocr_lora: corresponding with the script `qwen_image/run_qwen_image_ocr_lora.sh`; 
<br>
qwen_image_ocr_lora_async_reward: corresponding with the script `run_qwen_image_ocr_lora_async_reward.sh`.
</div>

> **Note:** Reward curves may differ from the references above mainly due to rollout-side stochasticity: diffusion rollouts sample random latents/noise, and the example scripts do not fix the data seed, so prompt ordering can vary between runs.
