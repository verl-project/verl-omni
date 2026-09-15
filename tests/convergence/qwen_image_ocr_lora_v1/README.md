# Qwen-Image OCR LoRA v1 L4 precision job

Runs the production LoRA OCR recipe on **real** Qwen-Image / Qwen3-VL weights
and the real OCR parquet for 100 steps on 8 GPUs. There is **no baseline**
compare. Timing is written into the report only.

Attention is pinned to ordinary PyTorch SDPA (`attn_backend=native`,
`rollout_attn_backend=TORCH_SDPA`), not Flash Attention.

## Gates

| Check | Metric | Fail when |
| --- | --- | --- |
| Train-infer gap | `training/rollout_probs_diff_mean` if logged, else `rollout_corr/logprob_abs_diff_mean` | any post-warmup step **> 0.01**, missing, or non-finite |
| Grad | `actor/grad_norm` | NaN / Inf |
| Val reward at step 100 | `val-core/*/reward/mean@*` | missing, or any source **< `VAL_REWARD_MIN`** |

`calculate_log_probs=true` and rollout-correction bypass off are required so
the train-infer gap exists.

`VAL_REWARD_MIN` defaults to **0.9**. Override it if a recipe needs a different
floor:

```bash
VAL_REWARD_MIN=0.85 bash tests/convergence/qwen_image_ocr_lora_v1/run_qwen_image_ocr_lora_v1.sh
```

## Recorded, not gated

`perf/time_per_step`, `timing_s/{step,gen,old_log_prob,reward,update_actor}`,
`perf/throughput` — summarized in `report.json` under `"perf"`.

Per-step `actor/loss` (or `actor/loss/mean`) and validation reward at every
`TEST_FREQ` step (default **20**) are written to `metrics.json`. The gate still
uses the step-100 val floor only.

## Requirements

- 8 GPUs (`NUM_GPUS`)
- `$WORKSPACE/models/Qwen-Image` (`MODEL_PATH`)
- `$WORKSPACE/models/Qwen3-VL-8B-Instruct` (`REWARD_MODEL_PATH`)
- `$WORKSPACE/data/ocr/qwen_image/{train,test}.parquet`

`WORKSPACE` defaults to `$HOME`.

## Run

```bash
bash tests/convergence/qwen_image_ocr_lora_v1/run_qwen_image_ocr_lora_v1.sh
```

Hydra overrides can be appended after the script. Validation runs every
`TEST_FREQ` steps (default 20) and always on the last step.

## Outputs

```text
outputs/l4_convergence/
|-- current/qwen_image_ocr_lora_v1/metrics.jsonl
|-- current/qwen_image_ocr_lora_v1/metrics.json
|-- current/qwen_image_ocr_lora_v1/report.json
`-- logs/qwen_image_ocr_lora_v1/qwen_image_ocr_lora_v1.log
```
