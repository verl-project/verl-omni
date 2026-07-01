# Qwen3-Omni Thinker GSPO + LoRA Trainer

This example shows how to post-train the **Qwen3-Omni-30B-A3B Thinker** with
**GSPO + LoRA** on a math-reasoning task, using FSDP for the actor and
`vllm-omni` as the async rollout backend.

For the base environment setup, see the [installation guide](../../docs/start/install.md).

## Installation

Follow the [installation guide](../../docs/start/install.md) to set up the base
environment (vLLM + vLLM-Omni). This recipe was last validated end-to-end on the
following stack (rollout↔actor pearson ≈ 0.993):

| Component | Version |
| --- | --- |
| vLLM | `0.22.0` |
| vLLM-Omni | `0.22.0` (the `v0.22.0` release tag) |
| transformers | `4.57.6` (see warning below) |
| torch | `2.11.0+cu130` |
| flash-attn | `2.8.3` |
| accelerate | `1.12.0` |
| verl | commit: `8a69493027` (w/ FSDP LoRA fixes) |

This pins to the upcoming `vllm 0.22.0` + `vllm-omni 0.22.0` release so the recipe
stays aligned with what maintainers ship.

```bash
# vLLM + vLLM-Omni rollout backend
pip install vllm==0.22.0
pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@v0.22.0"

# vllm-omni's CLI entrypoints import pydub at startup but don't declare it
pip install pydub

# verl (must include the FSDP layered-summon fix)
pip install "verl @ git+https://github.com/verl-project/verl.git@8a69493027"

# verl-omni (this repo)
pip install -e .
```

> **Pin transformers to 4.x.** transformers `5.x` rewrote the weight-loading
> path (`core_model_loading.py`); combined with `accelerate`'s meta-device init
> it raises `TypeError: Parameter.__new__() got an unexpected keyword argument
> '_is_hf_initialized'` when loading the 30B checkpoint. Use `transformers<5`
> (validated on `4.57.6`); vLLM's transformers floor is satisfied either way.

> **flash-attn needs a CUDA toolkit matching torch.** torch `2.11.0+cu130` is
> built against CUDA 13; building flash-attn from source against a system CUDA 12.x
> `nvcc` fails the version check. If you hit this, point `CUDA_HOME` at the
> pip-installed `nvidia/cu13` toolkit (`export CUDA_HOME=$(python -c "import os,
> nvidia; print(os.path.join(os.path.dirname(nvidia.__file__), 'cu13'))")`)
> before `pip install flash-attn==2.8.3 --no-build-isolation`.

Verify:

```bash
python -c "import verl, verl_omni, vllm, vllm_omni; print('OK')"
```

The run scripts set
`export VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni.models.transformers.qwen3_omni_thinker`,
so verl loads both on the driver via its `VERL_USE_EXTERNAL_MODULES` hook:
`verl_omni` registers the `vllm_omni` rollout adapter, and the
`qwen3_omni_thinker` submodule applies the Qwen3-Omni processor / automodel
patches before the driver's dataset loader runs. The GPU workers load the same
model patch via `actor_rollout_ref.model.external_lib`.

The provided script is configured for a single node with **4 × H100/H200 80GB**:
the actor (FSDP, 30B + LoRA r=64 with param/optimizer offload) and the
`vllm-omni` rollout (TP=4) colocate on the same 4 GPUs. Multi-node is not yet
validated.

> **Where the rollout engine's memory/batching is set.** When
> `stage_configs_path` is provided, vLLM-Omni **ignores** the top-level engine
> args verl passes (`gpu_memory_utilization`, `max_num_seqs`, `load_format`,
> `dtype`, LoRA, …) — the per-stage YAML takes precedence. So the rollout engine
> runs with the values in
> [`qwen3_omni/qwen3_omni_thinker_only.yaml`](qwen3_omni/qwen3_omni_thinker_only.yaml)
> (e.g. `gpu_memory_utilization: 0.4`), kept low because the engine **shares each
> GPU with the FSDP actor**. To change rollout memory/batching, edit that stage
> file, not the verl rollout config.

> `vllm==0.22` pulls `numpy>=2.x` while verl/verl-omni still pin `numpy<2.0.0`;
> the codepaths used here are numpy-2 compatible, so the pip resolver warning is
> safe to ignore.

## Prepare the dataset

A parquet dataset of math problems with `prompt` and `answer` fields, defaulting
to `~/data/math/{train,test}.parquet`. The example was tested on
`MATH-lighteval`; any standard RL math dataset works. To convert HuggingFace
datasets into verl's parquet format, see
[`verl/examples/data_preprocess/`](https://github.com/verl-project/verl/tree/main/examples/data_preprocess).

```bash
mkdir -p ~/data/math
# … place train.parquet and test.parquet here …
ls ~/data/math/   # train.parquet  test.parquet
```

## Prepare the model

The script uses the HuggingFace Hub ID `Qwen/Qwen3-Omni-30B-A3B-Instruct`
(~60 GB), cached automatically on first run. To use a local copy, set
`MODEL_PATH`:

```bash
export MODEL_PATH=/path/to/local/Qwen3-Omni-30B-A3B-Instruct
```

> **Use the Instruct variant.** The base checkpoint ships no
> `tokenizer.chat_template`; verl's dataset loader calls
> `tokenizer.apply_chat_template(...)` and fails without it.

## Run training

Launch from the repository root:

```bash
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora.sh
```

The recipe config lives in
[`qwen3_omni/config/qwen3_omni_thinker_gspo.yaml`](qwen3_omni/config/qwen3_omni_thinker_gspo.yaml),
which inherits verl's default `ppo_trainer` config and overrides the GSPO/LoRA
fields. The launch script passes it via `--config-name` and only sets volatile
values (data/model paths, GPU/node counts, the vLLM-Omni stage config path) on
the command line. Config precedence, lowest to highest:

```
verl ppo_trainer defaults  →  config/qwen3_omni_thinker_gspo.yaml  →  CLI overrides
```

So any field can be overridden from the command line without editing the yaml:

```bash
MODEL_PATH=/local/Qwen3-Omni-30B-A3B-Instruct \
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora.sh \
    trainer.total_epochs=10 \
    actor_rollout_ref.actor.optim.lr=2e-6
```

To verify the wiring before a full run, use the end-to-end GSPO smoke test
[`tests/special_e2e/run_gspo_qwen3_omni_thinker_lora_smoke.sh`](../../tests/special_e2e/run_gspo_qwen3_omni_thinker_lora_smoke.sh),
which trains on a tiny random-weight model built by
[`build_qwen3_omni_tiny_random.py`](../../tests/special_e2e/build_qwen3_omni_tiny_random.py)
(no 60 GB download). It runs under `transformers<5`; it will be wired into the
`tests/gpu_smoke` CI suite once the recipe supports transformers 5.x.

## Logging

W&B logging is enabled by default:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
# trainer.project_name / experiment_name are already set in the script
```

## What is trained

Only the **Thinker** (`Qwen3OmniMoeThinkerForConditionalGeneration`):

- LoRA rank 64, alpha 32, on `target_modules="all-linear"`.
- `exclude_modules` strips talker / code2wav / code_predictor / visual /
  audio_tower; `freeze_vision_tower=True` keeps the vision encoder cold.
- The non-Thinker heads are dropped at FSDP-wrap time via `_verl_strip_modules`.

Reward comes from the `dapo` reward manager (math accuracy on parsed answers).

Healthy signals after one full step (~22 min on 4×H100):

- `training/rollout_actor_probs_pearson_corr` > 0.95 (actor ↔ rollout agree
  after weight sync) — the primary correctness signal.
- `actor/loss` ≈ 1e-4…1e-3, `actor/grad_norm` ∈ [1e-3, 1], no OOM
  (`actor/perf/max_memory_allocated_gb` < 60).
- `val-core/.../acc/mean@1` rising with steps.

## Performance

> Measured on a single node of **4 × H100/H200 80GB**, actor and rollout
> colocated, MATH-lighteval, `dapo` reward.

| Script | Model | Algorithm | # Cards (colocate) | Batch × `rollout.n` | lr | Throughput (tok/gpu/s) | Time / Step (s) | val acc/mean@1 | rollout↔actor pearson |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `run_qwen3_omni_thinker_gspo_lora.sh` | Qwen3-Omni-30B-A3B Thinker | GSPO + LoRA (r=64) | 4 | 8 × 8 = 64 | 1e-6 | 38.2 | ~1350 | 0.90 | 0.993 |

The step time is dominated by rollout generation (~1010 s of ~1350 s) because
`max_response_length=8192` with `rollout.n=8`; `actor/perf/max_memory_allocated`
peaks at ~57 GB.

## Preliminary results

Validation accuracy on MATH-lighteval sits around **0.90** with the default
config. Treat this as a plumbing-correctness signal (finite loss, reasonable
grad norm, rollout↔actor pearson ≈ 0.99, no OOM) rather than evidence the recipe
is tuned — gains are slow because the Instruct base is already a strong
zero-shot solver, LoRA r=64 has limited capacity against a 30B base, and the
binary math reward yields low-variance advantages on a high-baseline policy.

![training reward](reward.png)

## File map

```
examples/gspo_trainer/
├── qwen3_omni/
│   ├── run_qwen3_omni_thinker_gspo_lora.sh   ← launch script (volatile overrides only)
│   ├── config/
│   │   └── qwen3_omni_thinker_gspo.yaml      ← recipe config (inherits verl ppo_trainer)
│   └── qwen3_omni_thinker_only.yaml          ← vllm-omni stage config
├── reward.png                            ← preliminary reward curve
└── README.md                             ← (this file)
```
