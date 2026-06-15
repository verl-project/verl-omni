(bagel_flowgrpo_quickstart)=
# Quickstart: FlowGRPO training on BAGEL-7B-MoT OCR dataset

Last updated: 06/15/2026

Post-train a BAGEL-7B-MoT policy with FlowGRPO for OCR-style image generation tasks.

## Introduction

In this example, we post-train a [BAGEL-7B-MoT](https://github.com/ByteDance-Seed/BAGEL-7B-MoT) policy with FlowGRPO for OCR-style image generation tasks. BAGEL is a **Mixture-of-Transformers (MoT)** model that supports both image understanding and generation. Unlike standard diffusion models (e.g., Qwen-Image), BAGEL:

- Takes **raw token IDs** as input rather than pre-computed prompt embeddings.
- Uses a **3-branch CFG** scheme (text-unconditional, image-unconditional, and full-conditional branches) with global/channel renormalisation.
- Is a **non-diffusers** model: it is a standalone `nn.Module` that does not inherit from `diffusers.ModelMixin`. It manages its own architecture, config format, weight-loading logic, and internal text processing.

The rollout uses `vllm-omni`'s `BagelPipeline` for multimodal generation, wrapped with an SDE scheduler for stochastic denoising and log-probability recording. The reward is computed by a visual generative reward model (*Qwen3-VL-8B-Instruct*) that compares OCR text extracted from generated images against the dataset ground truth.

## Prerequisites

- Install VeRL-Omni and its dependencies following the {doc}`installation guide <install>`. Also install the FlowGRPO-specific reward dependency:

```bash
pip install Levenshtein
```

- Use a machine with `4` GPUs for the provided example script.
- Run the commands below from the repository root.
- Download the BAGEL-7B-MoT model checkpoint from Hugging Face:

```bash
huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT --local-dir ~/models/ByteDance-Seed/BAGEL-7B-MoT
```

> **Note.** BAGEL requires the full checkpoint directory including `ema.safetensors`, `config.json`, `llm_config.json`, `tokenizer.json`, and the `tokenizer_config.json` shipped with the model. The Hugging Face Hub ID `ByteDance-Seed/BAGEL-7B-MoT` provides all of these.

## Dataset Introduction

We use the same OCR dataset as the Qwen-Image FlowGRPO example: [dataset/ocr](https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr). Each sample asks the model to generate an image that contains specific text, and the reward model scores the generated image by reading the rendered text and comparing it with the reference OCR string.

The raw dataset is a plain-text file (`train.txt` / `test.txt`) where each line is one generation prompt. The OCR target — the text the model must render in the image — is enclosed in double quotes within the prompt. A few representative samples:

```text
A close-up of a medicine bottle with a clear, red warning label that reads "Take With Food" prominently displayed, set against a neutral background.
A close-up of a robot's chest panel, with a digital display blinking "System Override Active" in red, set against a dimly lit industrial background.
A detailed textbook diagram labeled "Photosynthesis Process", viewed under a high-powered microscope, showcasing the intricate cellular structures and chemical reactions involved.
```

The BAGEL-specific preprocessing script converts the raw dataset into parquet files that contain:

- the multimodal prompt (system + user messages),
- a negative prompt for CFG unconditional branches,
- **pre-tokenised BAGEL prompt IDs** in `bagel_prompt_ids` — BAGEL uses raw `tokenizer.encode(user_text)` wrapped with `<|im_start|>` / `<|im_end|>` markers, rather than a chat template,
- OCR ground truth stored under `reward_model.ground_truth`,
- auxiliary metadata such as split and sample index.

## Step 1: Prepare the dataset

Set the `WORKSPACE` environment variable to any writable directory you prefer (defaults to `$HOME` if unset):

```bash
export WORKSPACE=${WORKSPACE:-$HOME}
```

Obtain the raw OCR dataset from the original Flow-GRPO repository and place it under `$WORKSPACE/data/ocr`. Then preprocess it into `train.parquet` and `test.parquet` with BAGEL-specific tokenisation:

```bash
python3 examples/flowgrpo_trainer/data_process/bagel_ocr.py \
  --model_path ~/models/ByteDance-Seed/BAGEL-7B-MoT \
  --input_dir $WORKSPACE/data/ocr \
  --output_dir $WORKSPACE/data/ocr/bagel
```

The command above writes:

- `$WORKSPACE/data/ocr/bagel/train.parquet`
- `$WORKSPACE/data/ocr/bagel/test.parquet`

These parquet files are the inputs consumed by the BAGEL FlowGRPO training script.

### Preparing a custom dataset

To train on your own OCR-style data, create `train.txt` and `test.txt` following the same one-prompt-per-line convention. Each prompt must contain the target OCR string enclosed in double quotes — the preprocessing script extracts the text between the first pair of quotes as the ground truth. For example:

```text
A vintage storefront sign above the door reads "Open 24 Hours" in bold neon letters.
A handwritten sticky note on a refrigerator says "Buy milk" in blue ink.
```

Place the files in `$WORKSPACE/data/ocr/` (or any directory you prefer) and run the same preprocessing command, adjusting `--input_dir` and `--output_dir` as needed:

```bash
python3 examples/flowgrpo_trainer/data_process/bagel_ocr.py \
  --model_path ~/models/ByteDance-Seed/BAGEL-7B-MoT \
  --input_dir $WORKSPACE/data/ocr \
  --output_dir $WORKSPACE/data/ocr/bagel
```

For datasets with a different ground-truth extraction scheme (e.g. a CSV with an explicit label column), modify `extract_ocr_solution` and the `make_map_fn` function in `examples/flowgrpo_trainer/data_process/bagel_ocr.py` to match your format, then re-run the script to regenerate the parquet files.

> **Important.** The tokenisation must match BAGEL's upstream pipeline exactly. BAGEL uses `<|im_start|>user_text<|im_end|>` wrapping (no chat template, no vision special tokens). The `bagel_ocr.py` script handles this correctly; if you write a custom data pipeline, ensure you replicate `tokenize_bagel_prompt()` from that script.

## Step 2: Obtain models for RL training

In this example, we train `ByteDance-Seed/BAGEL-7B-MoT` with LoRA and use `Qwen/Qwen3-VL-8B-Instruct` as the OCR reward model.

**Policy model (BAGEL-7B-MoT):** the script defaults to `~/models/ByteDance-Seed/BAGEL-7B-MoT`. Download it as described in [Prerequisites](#prerequisites) or set `model_name` in the script to a local path. BAGEL does **not** use Hugging Face `diffusers` loading — it reads `ema.safetensors` and `config.json` directly.

**Reward model (Qwen3-VL-8B-Instruct):** the script defaults to the Hugging Face Hub ID `Qwen/Qwen3-VL-8B-Instruct`, so no manual download is required — Hugging Face will cache it automatically on first run. To use a local copy instead, edit the `reward_model_name` variable in the script directly.

The run script exposes the following environment variable:

```bash
WORKSPACE              # base directory for data (default: $HOME)
```

## Step 3: Perform FlowGRPO training

The provided example script launches `python3 -m verl_omni.trainer.main_diffusion` with BAGEL-specific config overrides for architecture, deploy config, LoRA targets, FSDP prefixes, and reward function.  See the script for the full command.

Run the training script:

```bash
bash examples/flowgrpo_trainer/run_bagel_flowgrpo_lora.sh
```

Optional KL loss tuning:

- `actor_rollout_ref.actor.use_kl_loss=True`
- `actor_rollout_ref.actor.kl_loss_coef=0.1`

The script uses `$WORKSPACE` (default: `$HOME`) as the base directory. Override any path via the environment variables described in Step 2, or set `WORKSPACE` to point to a volume with enough free space before launching.

You are expected to see training, validation, actor, critic, and reward metrics logged through the configured backends. By default, checkpoints are saved under:

```bash
checkpoints/${trainer.project_name}/${trainer.experiment_name}
```

## BAGEL-specific configuration

BAGEL differs from standard diffusion models in several ways. The key configuration knobs are:

### Architecture override

Unlike diffusers-based models where architecture is auto-detected from `model_index.json`, BAGEL requires an explicit override:

```bash
+actor_rollout_ref.model.architecture=OmniBagelForConditionalGeneration
```

### Deploy config

BAGEL rollout uses a custom vllm-omni deploy YAML that selects the `bagel_single_stage` pipeline topology:

```bash
BAGEL_DEPLOY_CONFIG=${BAGEL_DEPLOY_CONFIG:-"$(dirname "$0")/bagel_deploy_config.yaml"}
+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.deploy_config=$BAGEL_DEPLOY_CONFIG
```

The deploy config at [`examples/flowgrpo_trainer/bagel_deploy_config.yaml`](../../examples/flowgrpo_trainer/bagel_deploy_config.yaml) mirrors vllm-omni's `bagel_single_stage.yaml` and sets:

| Field | Value | Notes |
|---|---|---|
| `pipeline` | `bagel_single_stage` | Single-stage DiT-only topology |
| `max_num_batched_tokens` | `32768` | Max tokens per diffusion batch |
| `max_num_seqs` | `1` | BAGEL diffusion processes one sequence at a time |
| `enforce_eager` | `true` | CUDA graphs not supported for diffusion |
| `trust_remote_code` | `true` | BAGEL uses custom Qwen2-MoT code |

### LoRA target modules

BAGEL uses MoT (Mixture-of-Transformers) with separate projection layers for text and generation pathways. The target modules must include both the standard `*_proj` layers and the `*_moe_gen` layers:

```bash
actor_rollout_ref.model.target_modules="['q_proj_moe_gen','k_proj_moe_gen','v_proj_moe_gen','o_proj_moe_gen','mlp_moe_gen.gate_proj','mlp_moe_gen.up_proj','mlp_moe_gen.down_proj']"
```

### FSDP layer prefixes

BAGEL's transformer layers are named `layers.N` (not `transformer_blocks.N`):

```bash
actor_rollout_ref.model.fsdp_layer_prefixes="['layers.']"
```

### Timestep shift

BAGEL uses a timestep shift of `3.0` (SD3-style), which is built into the sigma schedule. The shift is applied automatically by `setup_bagel_sigmas()` in [`verl_omni/pipelines/bagel_flow_grpo/common.py`](../../verl_omni/pipelines/bagel_flow_grpo/common.py). You do not need to set it manually.

## FAQ: tuning OOM-related parameters

| OOM location | First parameter to tune | What it changes |
| --- | --- | --- |
| Rollout generation OOM | Increase `ROLLOUT_TP` | Sets `actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP` and reduces `actor_rollout_ref.rollout.agent.num_workers` to `NUM_GPUS / ROLLOUT_TP`. |
| Reward-model OOM | Increase `REWARD_TP` | Sets `reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP` and reduces `reward.num_workers` to `NUM_GPUS / REWARD_TP`. |
| Actor loss forward/backward OOM | Decrease `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | Splits each actor mini-batch into smaller per-GPU chunks. |
| Old log-prob recomputation OOM | Decrease `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu` | Splits actor-side log-prob inference into smaller per-GPU chunks. |
| Reference log-prob OOM | Decrease `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu` | Splits reference-policy log-prob inference into smaller per-GPU chunks. |

If rollout OOM persists after increasing `ROLLOUT_TP`, reduce memory-heavy rollout settings such as `actor_rollout_ref.rollout.n`, image `height` / `width`, or `actor_rollout_ref.rollout.pipeline.num_inference_steps`. BAGEL uses `max_num_seqs: 1` by default, so concurrency tuning is less effective — focus on the per-request memory footprint.

## Wandb logging

The provided script already enables:

```bash
trainer.logger='["console", "wandb"]' \
trainer.project_name=flow_grpo \
trainer.experiment_name=bagel_ocr_lora
```

Set your W&B credentials before launching if you want remote tracking:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
```

You can also override `trainer.project_name` and `trainer.experiment_name` from the command line to organize runs under your own project names.

## How BAGEL differs from Qwen-Image

This section helps users migrating from the Qwen-Image FlowGRPO example to BAGEL.

| Dimension | Qwen-Image | BAGEL-7B-MoT |
|---|---|---|
| **Model type** | Diffusers `ModelMixin` | Standalone `nn.Module` (`NonDiffusersModelBase`) |
| **Text input** | Pre-computed prompt embeddings `(B, L, D)` | Raw token IDs `(B, L)` with internal embedding |
| **CFG** | True CFG with negative prompt embeddings | 3-branch CFG (gen / text-uncond / img-uncond) with global/channel renormalisation |
| **Scheduler** | `FlowMatchEulerDiscreteScheduler` → `FlowMatchSDEDiscreteScheduler` | Internal ODE step → `FlowMatchSDEDiscreteScheduler` with `_BagelSchedulerAdapter` (4-arg `step()` convention) |
| **Timestep convention** | `t/1000` | Raw sigma with SD3-style shift of `3.0` |
| **Latent shape** | Packed sequence `(B, seq, patch_dim)` | Packed sequence `(B, seq, patch_dim)` |
| **Position encoding** | 2-D RoPE | 2-D sincos position embedding for latent patches |
| **Architecture registration** | Auto-detected from `model_index.json` | Explicit: `+actor_rollout_ref.model.architecture=OmniBagelForConditionalGeneration` |
| **Weight loading** | `diffusers.AutoModel.from_pretrained` | `BagelForTraining.from_pretrained` reading `ema.safetensors` directly |
| **LoRA targets** | Standard transformer `*_proj` layers | MoT dual-pathway layers (`*_proj` + `*_moe_gen`) |

## Further reading

For the algorithm background, detailed configuration notes, and the BAGEL model integration architecture, see:

- {doc}`../algo/flowgrpo`
- {doc}`../contributing/integrating_a_diffusion_model`
- {doc}`../contributing/integrating_a_non_diffusers_model`
- [vLLM-Omni BAGEL documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/online_serving/bagel/)
