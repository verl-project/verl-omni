# SD3.5 OCR LoRA v1 L4 precision job

Runs the production SD3.5-Medium LoRA OCR recipe on **real** SD3.5 / Qwen2.5-VL
weights and the real OCR parquet for 100 steps on **8 GPUs** (6 actor+rollout, 2
reward). There is **no baseline** compare. Timing is written into the report only.

This L4 case uses **Route A (memory-first sync)**:

- `ROLLOUT_TP=2` and `REWARD_TP=2` to lower per-GPU VRAM
- FSDP param/optimizer offload and smaller micro-batches
- Larger `train_batch_size=16` and `rollout.n=16` for faster convergence

Attention is pinned to ordinary PyTorch SDPA (`attn_backend=native`,
`rollout_attn_backend=TORCH_SDPA`), not Flash Attention.

Reference example:
`examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh`

## Gates

| Check | Metric | Fail when |
| --- | --- | --- |
| Train-infer gap | `training/rollout_probs_diff_mean` if logged, else `rollout_corr/logprob_abs_diff_mean` | any post-warmup step **> 0.01**, missing, or non-finite |
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
| `ROLLOUT_PROB_DIFF_MEAN_MAX` | `0.01` | Max allowed train-infer log-prob gap per step |
| `VAL_REWARD_MIN` | `0.6` | Floor for step-100 validation OCR reward (placeholder until calibrated) |

`VAL_REWARD_MIN` defaults to **0.6** as an initial placeholder. Tighten after
reviewing a full 100-step run on your cluster:

```bash
VAL_REWARD_MIN=0.65 bash tests/convergence/sd35_medium_ocr_lora_v1/run_sd35_medium_ocr_lora_v1.sh
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
| `MAX_NUM_SEQS` | `128` | Request-level rollout depth cap |
| `REWARD_GPU_MEMORY_UTILIZATION` | `0.85` | Reward vLLM memory fraction |

## Recorded, not gated

`perf/time_per_step`, `timing_s/{step,gen,old_log_prob,reward,update_actor}`,
`perf/throughput` — summarized in `report.json` under `"perf"`.

Per-step trainer metrics are written to `metrics.jsonl`. The gate still uses the
step-100 val floor only.

## Requirements

- 8 GPUs (`NUM_GPUS_ACTOR_ROLLOUT=6`, `NUM_GPUS_REWARD=2`)
- `stabilityai/stable-diffusion-3.5-medium` (`MODEL_PATH`) or a local copy
- `Qwen/Qwen2.5-VL-3B-Instruct` (`REWARD_MODEL_PATH`) or a local copy
- `$WORKSPACE/data/ocr/sd3/{train,test}.parquet`

Prepare OCR parquet with:

```bash
python3 examples/flowgrpo_trainer/data_process/sd3_ocr.py \
  --output_dir "$WORKSPACE/data/ocr/sd3"
```

`WORKSPACE` defaults to `$HOME`.

## Run

```bash
bash tests/convergence/sd35_medium_ocr_lora_v1/run_sd35_medium_ocr_lora_v1.sh
```

Hydra overrides can be appended after the script. Validation runs every
`TEST_FREQ` steps (default 20) and always on the last step.

## Outputs

```text
tests/convergence/outputs/l4_convergence/
|-- sd35_medium_ocr_lora_v1/metrics.jsonl
|-- sd35_medium_ocr_lora_v1/report.json
`-- logs/sd35_medium_ocr_lora_v1/sd35_medium_ocr_lora_v1.log
```
