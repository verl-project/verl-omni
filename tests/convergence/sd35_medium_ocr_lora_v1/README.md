# SD3.5 OCR LoRA v1 L4 precision job

Runs the production SD3.5-Medium LoRA OCR recipe on **real** SD3.5 / Qwen2.5-VL
weights and the real OCR parquet for 100 steps on **8 devices** (6
actor+rollout, 2 reward). There is **no baseline** compare. Timing is written
into the report only.

Runs on NVIDIA GPU by default and on Ascend NPU via `DEVICE=npu` (sources the
CANN env, attention switches to `_native_npu`, artifacts land under the
`sd35_medium_ocr_lora_v1_npu` case name).

This L4 case uses **Route A (memory-first sync)**:

- `ROLLOUT_TP=2` and `REWARD_TP=2` to lower per-GPU VRAM
- FSDP param/optimizer offload and smaller micro-batches
- Larger `train_batch_size=16` and `rollout.n=16` for faster convergence

Attention is pinned to ordinary PyTorch SDPA (`attn_backend=native` on GPU /
`_native_npu` on NPU, `rollout_attn_backend=TORCH_SDPA` in both cases), not
Flash Attention.

Reference example:
`examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh`

## Gates

| Check | Metric | Fail when |
| --- | --- | --- |
| Train-infer gap | `training/rollout_probs_diff_mean` if logged, else `rollout_corr/logprob_abs_diff_mean` | any post-warmup step **> 1e-4**, missing, or non-finite |
| Grad | `actor/grad_norm` | NaN / Inf |
| Val reward at step 100 | `val-core/*/reward/mean@*` | missing, or any source **< `VAL_REWARD_MIN`** |

`calculate_log_probs=true` and rollout-correction bypass off are required so
the train-infer gap exists.

Default thresholds (printed at job start and stored in `report.json` under
`thresholds`):

| Env var | Default | Meaning |
| --- | --- | --- |
| `SKIP_STEPS` | `2` | Warmup steps excluded from train-infer / grad gates |
| `MIN_TRAIN_STEPS` | `100` | Minimum logged actor steps |
| `ROLLOUT_PROB_DIFF_MEAN_MAX` | `1e-4` | Max allowed train-infer log-prob gap per step |
| `VAL_REWARD_MIN` | `0.8` | Floor for step-100 validation OCR reward |

`VAL_REWARD_MIN` defaults to **0.8**. Override it if a recipe needs a different
floor:

```bash
VAL_REWARD_MIN=0.85 bash tests/convergence/sd35_medium_ocr_lora_v1/run_sd35_medium_ocr_lora_v1.sh
```

## Route A training defaults

| Env var | Default | Notes |
| --- | --- | --- |
| `NUM_GPUS_ACTOR_ROLLOUT` | `6` | Colocated actor + rollout |
| `NUM_GPUS_REWARD` | `2` | Dedicated GenRM pool |
| `ROLLOUT_TP` / `REWARD_TP` | `2` / `2` | Shard rollout and reward models |
| `TRAIN_BATCH_SIZE` | `16` | Up from the 3-GPU example (`8`) |
| `ROLLOUT_N` | `16` | Up from the 3-GPU example (`8`) |
| `PPO_MINI_BATCH_SIZE` | `8` | Must divide `TRAIN_BATCH_SIZE` |
| `PPO_MICRO_BATCH_SIZE` | `4` | Per-GPU actor micro-batch |
| `LOG_PROB_MICRO_BATCH_SIZE` | `4` | Old/ref log-prob micro-batch |
| `MAX_NUM_SEQS` | `8` | Request-level rollout concurrency cap; lower it to reduce the VAE-decode peak that caused NPU OOM in CI |
| `REWARD_GPU_MEMORY_UTILIZATION` | `0.85` | Reward vLLM memory fraction |

## Recorded, not gated

`perf/time_per_step`, `timing_s/{step,gen,old_log_prob,reward,update_actor}`,
`perf/throughput` — summarized in `report.json` under `"perf"`.

Per-step trainer metrics are written to `metrics.jsonl`. The gate still uses the
step-100 val floor only.

## Requirements

- 8 accelerators (`NUM_GPUS_ACTOR_ROLLOUT=6`, `NUM_GPUS_REWARD=2`) — NVIDIA
  GPUs or Ascend NPUs (CANN at `${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}`
  plus `/usr/local/Ascend/nnal/atb`)
- `stabilityai/stable-diffusion-3.5-medium` (`MODEL_PATH`) or a local copy
- `Qwen/Qwen2.5-VL-3B-Instruct` (`REWARD_MODEL_PATH`) or a local copy
- `${OCR_DATA_DIR}/{train,test}.parquet` where `OCR_DATA_DIR=${OCR_RAW_DIR}/sd3`

If the parquet files are missing, the script converts the raw one-prompt-per-line
`${OCR_RAW_DIR}/{train,test}.txt` automatically via `sd3_ocr.py`; otherwise it
fails closed. `OCR_RAW_DIR` defaults to `$WORKSPACE/data/ocr` when the case
script runs standalone; the launcher
(`tests/convergence/run_l4_convergence.sh`) exports
`OCR_RAW_DIR=$HOME/.cache/huggingface/hub/verl-omni` instead. Manual
preparation:

```bash
python3 examples/flowgrpo_trainer/data_process/sd3_ocr.py \
  --input_dir "${OCR_RAW_DIR:-$WORKSPACE/data/ocr}" \
  --output_dir "${OCR_RAW_DIR:-$WORKSPACE/data/ocr}/sd3"
```

`WORKSPACE` defaults to `$HOME`.

## Run

Preferred — via the `tests/convergence` launcher (case list, weight snapshots
`MODEL_PATH`/`REWARD_MODEL_PATH`, and `OCR_RAW_DIR` are wired in there):

```bash
bash tests/convergence/run_l4_convergence.sh sd35_medium_ocr_lora_v1
bash tests/convergence/run_l4_convergence.sh --device npu sd35_medium_ocr_lora_v1
```

GPU:

```bash
bash tests/convergence/sd35_medium_ocr_lora_v1/run_sd35_medium_ocr_lora_v1.sh
```

NPU (Ascend; sources CANN env and sets `DEVICE=npu`; artifacts are isolated
under `sd35_medium_ocr_lora_v1_npu`):

```bash
DEVICE=npu bash tests/convergence/sd35_medium_ocr_lora_v1/run_sd35_medium_ocr_lora_v1.sh
```

NPU notes (aligned with `tests/special_e2e/*_npu.sh` / npu_smoke CI):

- CANN is sourced from `${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh`
  plus the absolute `/usr/local/Ascend/nnal/atb/set_env.sh`; `DEVICE=npu` fails
  closed when they are missing.
- The script prints the sourced Ascend env and `npu-smi info` for diagnostics,
  and defaults `ASCEND_RT_VISIBLE_DEVICES=0..7` (`NUM_NPUS` overrides).
- An Ascend teardown abort with exit code 134 after training is tolerated: the
  pass/fail decision is deferred to the `collect_report.py` gates. Any other
  nonzero trainer exit fails the job.

Hydra overrides can be appended after the script. Validation runs every
`TEST_FREQ` steps (default 20) and always on the last step.

## CI

GitHub Actions workflow: `.github/workflows/l4_convergence.yml`

- Manual (GPU): **Actions → l4_convergence → Run workflow** → choose this
  case or `all`; enable **run_npu** to also run it on Ascend NPU.
- Pull request: apply the `L4-npu-ci` label → NPU job (runs every case
  registered in `tests/convergence/run_l4_convergence.sh` on the static 8-NPU
  runner `linux-aarch64-a2b4-8`, image
  `swr.cn-north-4.myhuaweicloud.com/mindspeed/pr-verl-omni-a2:v010-vllm030-omni030`,
  `DEVICE=npu`, timeout 720 min).

Artifacts: `tests/convergence/outputs/l4_convergence/**/*.jsonl` and `**/*.json`
(NPU job uploads under the `l4-convergence-npu-outputs-*` artifact name).

## Outputs

```text
tests/convergence/outputs/l4_convergence/
|-- sd35_medium_ocr_lora_v1/metrics.jsonl
|-- sd35_medium_ocr_lora_v1/report.json
|-- sd35_medium_ocr_lora_v1_npu/metrics.jsonl    # NPU run only
|-- sd35_medium_ocr_lora_v1_npu/report.json      # NPU run only
`-- logs/sd35_medium_ocr_lora_v1/sd35_medium_ocr_lora_v1.log
```
