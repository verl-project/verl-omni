# Qwen3-Omni Thinker GSPO Trainer

Last updated: 08/03/2026

This example shows how to post-train the **Qwen3-Omni-30B-A3B Thinker** with
**GSPO** on a math-reasoning task, using FSDP for the actor and `vllm-omni` as
the async rollout backend. Two input modalities are supported: **text → text**
(e.g. `gsm8k`) and **image → text** (e.g. `MMK12`).

Both **GPU** and **NPU** training platforms are supported:

- `examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_v1.sh`
  — **GPU**, **LoRA (r=32)** on a single node with **4 × H800 80GB**.
- `examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_npu.sh`
  — **NPU**, **full-parameter** on a single **Atlas 800T A3** node with **16 NPUs**.

For the base environment setup, see the [installation guide](../../docs/start/install.md).

## Installation

Follow the [installation guide](../../docs/start/install.md) to set up the base
environment. In short:

```bash
git clone https://github.com/verl-project/verl-omni.git && cd verl-omni
uv venv --python 3.12 --seed && source .venv/bin/activate
uv pip install -e ".[gpu]" --torch-backend=auto
uv pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@$(cat .github/vllm_omni_pin.txt)"
uv pip install -e ".[train,dev]"
# flash-attn is required for GPU training
uv pip install flash-attn>=2.8.3
```

> **Tested with** `transformers==5.13.1`, `accelerate==1.14.0`, `peft==0.19.1`.

Verify:

```bash
python -c "import verl, verl_omni, vllm, vllm_omni; print('OK')"
```

## Prepare the model

The GPU V1 scripts default `MODEL_PATH` to `$HOME/models/Qwen/Qwen3-Omni-30B-A3B-Instruct`
(~60 GB). The NPU script defaults to the HuggingFace Hub ID `Qwen/Qwen3-Omni-30B-A3B-Instruct`.
To use a different local copy or Hub ID, set `MODEL_PATH`:

```bash
export MODEL_PATH=/path/to/local/Qwen3-Omni-30B-A3B-Instruct
```

## Training with `gsm8k`

### Prepare the dataset

A parquet dataset of GSM8K math problems, defaulting to
`~/data/gsm8k/{train,test}.parquet`. Use verl's
[`gsm8k.py`](https://github.com/verl-project/verl/blob/main/examples/data_preprocess/gsm8k.py) converter:

```bash
python gsm8k.py --local_save_dir ~/data/gsm8k
ls ~/data/gsm8k/   # train.parquet  test.parquet
```

### Run training

Launch from the repository root — pick the flavor that matches your hardware:

```bash
# GPU, LoRA (r=32), 4 × H800 — V1 trainer (recommended)
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_v1.sh

# NPU, full-parameter, 16 × Atlas 800T A3
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_npu.sh
```

> **Deprecated:** `run_qwen3_omni_thinker_gspo_lora.sh` (old
> `verl.trainer.main_ppo` entrypoint with `external_lib` monkey-patches)
> is kept for backward compatibility but no longer recommended.

The V1 script uses pure CLI overrides on `verl_omni.trainer.main_omni`
(no `--config-path/--config-name`, no recipe YAML). Config precedence,
lowest to highest:

```
verl omni_trainer defaults  →  CLI overrides (run script)  →  "$@" extra args
```

Any field can be overridden from the command line without editing the script:

```bash
MODEL_PATH=/local/Qwen3-Omni-30B-A3B-Instruct \
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_v1.sh \
    trainer.total_epochs=10 \
    actor_rollout_ref.actor.optim.lr=2e-6
```

### What is trained

Only the **Thinker** (`Qwen3OmniMoeThinkerForConditionalGeneration`):

- **GPU (LoRA)** — rank 32, alpha 64, on
  `target_modules="['q_proj','k_proj','v_proj','o_proj']"`
  (the V1 `Qwen3OmniThinkerAdapter.configure_model` handles Thinker-forward
  redirection and `_verl_strip_modules` via `get_strip_modules`,
  so `exclude_modules` only needs to cover the heads/encoders).
- **NPU (full-parameter)** — LoRA is disabled (`lora_rank=0`); all Thinker
  parameters are updated under FSDP.
- `exclude_modules` strips talker / code2wav / code_predictor / visual /
  audio_tower; `freeze_vision_tower=True` keeps the vision encoder cold.
- `configure_model` in the registered adapter
  (`verl_omni/pipelines/qwen3_omni/thinker_training_adapter.py`) redirects
  `module.forward` → `module.thinker.forward` and sets
  `_no_split_modules` after the default base-class stripping.

Reward comes from the `naive` reward manager (math accuracy on parsed answers).

Healthy signals (gsm8k, 4×H800, LoRA r=32):

- `training/rollout_actor_probs_pearson_corr` > 0.995 (actor ↔ rollout agree
  after weight sync) — the primary correctness signal.
- `rollout_corr/log_ppl_diff` ≈ 0.001 (near zero, confirms rollout↔actor
  log-prob consistency).
- `actor/loss` ≈ 1e-5, `actor/grad_norm` ∈ [1e-3, 1e-2], no OOM
  (`actor/perf/max_memory_allocated_gb` < 45).
- `val-core/openai/gsm8k/acc/mean@1` rising with steps.

## Training with `MMK12`

For visual math reasoning we ship an end-to-end pipeline on top of the
[MMK12](https://huggingface.co/datasets/FanqingM/MMK12) dataset (image
input + text output, K12 math). It reuses the same GSPO recipe as the
text-only path — only the data preprocessing and the reward scorer differ.
Use the dedicated V1 GPU/LoRA script:

### Prepare the dataset

Download the raw MMK12 parquet shards (from ModelScope or HuggingFace) into a
local directory — the loader expects filenames like `train-*.parquet` and
`test-*.parquet` — and convert them into the verl RL parquet layout with:

```bash
python examples/gspo_trainer/data_process/mmk12.py \
    --local_dataset_path /path/to/mmk12/ \
    --local_save_dir ~/data/mmk12
```

The converter emits one verl RL row per problem, with
`data_source="math_dapo"`, a system prompt that constrains the model to emit
`<answer>…\boxed{…}…</answer>`, and the image bytes carried inline in the
`images` column so the parquet stays self-contained. Input / kept / dropped
counts and answer-type tallies are printed at the end. See the module docstring
in [`examples/gspo_trainer/data_process/mmk12.py`](https://github.com/verl-project/verl-omni/blob/main/examples/gspo_trainer/data_process/mmk12.py) for the exact output schema.

### Run training

The MMK12 reward scorer grades responses with
[`math_verify`](https://github.com/huggingface/math-verify). Multimodal data
processing also requires [`qwen-vl-utils`](https://github.com/QwenLM/Qwen2.5-VL)
for vision info extraction. Install both explicitly:

```bash
pip install math-verify qwen-vl-utils
```

Then launch the MMK12 V1 training script:

```bash
TRAIN_FILE=$HOME/data/mmk12/train.parquet \
VAL_FILE=$HOME/data/mmk12/test.parquet \
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_mmk12_v1.sh
```

The script registers the custom reward scorer internally (no yaml edits
required). Override LR or other fields via "$@" extras:

```bash
TRAIN_FILE=$HOME/data/mmk12/train.parquet \
VAL_FILE=$HOME/data/mmk12/test.parquet \
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_mmk12_v1.sh \
    actor_rollout_ref.actor.optim.lr=3e-6
```

The scorer combines `math_verify` accuracy with a progressive format reward on
the `<answer>…\boxed{}…</answer>` template; see
[`verl_omni/utils/reward_score/mmk12_reward.py`](https://github.com/verl-project/verl-omni/blob/main/verl_omni/utils/reward_score/mmk12_reward.py)
for the full formula.

## Performance

All GPU results measured on a single node of **4 × H800 80GB**, actor and
rollout colocated, LoRA r=32, GSPO.

| Script | Dataset | # Cards | Batch × `rollout.n` | lr | Steps | val acc@1 / reward@1 | rollout↔actor pearson | GPU memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [`gsm8k (wandb)`](https://wandb.ai/mikecheung/gspo/runs/j5mro1tn) | gsm8k | 4 | 128 × 16 = 2048 | 3e-6 | 578 | acc 0.969 | 0.997 | ~43 GB |
| [`MMK12 (wandb)`](https://wandb.ai/mikecheung/gspo/runs/2j8hxr36) | MMK12 | 4 | 128 × 16 = 2048 | 3e-6 | 456 | reward 0.833 | 0.998 | ~59 GB |

**gsm8k** ([wandb](https://wandb.ai/mikecheung/gspo/runs/j5mro1tn), `naive`
reward, math accuracy): `critic/rewards/mean` rose from ~0.93 to ~0.97,
`val-core/openai/gsm8k/acc/mean@1` reached **0.969**.
`rollout_corr/log_ppl_diff` stayed near zero (~0.002).

**MMK12** ([wandb](https://wandb.ai/mikecheung/gspo/runs/2j8hxr36), composite
reward, `math_verify` + format): `critic/rewards/mean` reached 0.842,
`val-core/mmk12/reward/mean@1` reached **0.833** (still training at
step 456). `rollout_corr/log_ppl_diff` stayed near zero (~0.002).

## Logging

W&B logging is enabled by default:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
# trainer.project_name / experiment_name are already set in the script
```

## File map

```
examples/gspo_trainer/
├── qwen3_omni/
│   ├── run_qwen3_omni_thinker_gspo_lora_v1.sh       ← V1 launch script (GPU, LoRA r=32, text)
│   ├── run_qwen3_omni_thinker_gspo_lora_mmk12_v1.sh  ← V1 launch script (GPU, LoRA r=32, image)
│   ├── run_qwen3_omni_thinker_gspo_lora.sh           ← deprecated (old main_ppo entrypoint)
│   ├── run_qwen3_omni_thinker_gspo_npu.sh            ← launch script (NPU, full-parameter)
│   ├── config/
│   │   └── qwen3_omni_thinker_gspo.yaml              ← old recipe config (deprecated path only)
│   ├── qwen3_omni_thinker_only.yaml                  ← old vllm-omni stage config (deprecated path only)
│   └── qwen3_omni_thinker_only_npu.yaml              ← old vllm-omni stage config (deprecated path only)
├── data_process/
│   └── mmk12.py                                      ← MMK12 → verl RL parquet converter
└── README.md                                         ← (this file)
```
