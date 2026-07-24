# Qwen-Image FlowGRPO Single-Sample Nightly

This nightly regression runs 20 FlowGRPO training steps on local tiny random
`Qwen-Image` weights with one deterministic OCR prompt repeated across the
batch. It also uses a local tiny random Qwen-VL reward model and performs two
checks in the same job:

- Steps in `DEBUG_DUMP_STEPS` default to `1,2` and write driver, actor-forward,
  and LoRA-gradient debug dumps for precision comparison.
- Steps after `PERF_SKIP_STEPS` default to `2` and are used for timing,
  throughput, and memory metric comparison.

All implementation lives in this directory and is enabled through test-side
hooks. No `verl_omni/` production code is modified.

## CI Status

This directory contains a runnable L3 nightly regression, but the repository
does not currently include a stable GitHub scheduled workflow for it. Run it
manually on a fixed GPU runner, or call the script from an external nightly
orchestration job.

Use this test as a regression signal for numerical drift and performance
changes. It is not part of the fast pull-request CI loop.

## Requirements

The runner must have:

- 4 GPUs by default, or set `NUM_GPUS` to match the runner.
- An installed `verl_omni` GPU environment with the rollout and training
  dependencies needed by Qwen-Image FlowGRPO.
- Local tiny-random policy and reward model directories.
- Enough local disk space for debug dumps, metrics, and logs under
  `OUTPUT_ROOT`.

## Run

```bash
bash tests/nightly/qwen_image_flowgrpo_single_sample/run_qwen_image_flowgrpo_single_sample.sh
```

Expected local model defaults:

- Policy: `~/models/tiny-random/Qwen-Image`
- Reward: `~/models/tiny-random/qwen3-vl`

Useful overrides:

```bash
NUM_GPUS=4 \
MODEL_PATH=/path/to/tiny-random/Qwen-Image \
REWARD_MODEL_PATH=/path/to/tiny-random/qwen3-vl \
OUTPUT_ROOT=/path/to/debug_dumps \
bash tests/nightly/qwen_image_flowgrpo_single_sample/run_qwen_image_flowgrpo_single_sample.sh
```

Useful nightly mode:

```bash
BOOTSTRAP_MISSING_BASELINE=0 \
bash tests/nightly/qwen_image_flowgrpo_single_sample/run_qwen_image_flowgrpo_single_sample.sh
```

## Baselines

The default layout is:

```text
outputs/debug_dumps/
|-- current/
`-- baseline/
```

`BOOTSTRAP_MISSING_BASELINE=1` is the default, so a first run with no baseline
will copy current artifacts into `baseline/` and pass. Set
`BOOTSTRAP_MISSING_BASELINE=0` for strict nightly comparison.

Use bootstrap mode only when intentionally creating or refreshing a reviewed
baseline. A real scheduled nightly should use strict mode so missing baselines,
missing dump files, or metric regressions fail the job.

The comparison outputs are written under `current/`:

- `metrics.json` contains aggregated timing, throughput, and memory metrics.
- `dump_compare.json` contains precision comparison results.
- `metrics.jsonl` contains step-level debug metrics from the run.
- `logs/qwen_image_flowgrpo_single_sample.log` contains the console log.

Precision thresholds:

- `PRECISION_ATOL`, default `1e-4`
- `PRECISION_RTOL`, default `1e-3`
- `PRECISION_MIN_COS_SIM`, default `0.999`

Performance threshold:

- `PERF_THRESHOLD`, default `0.10`

## Failure Triage

When the job fails:

1. Check `logs/qwen_image_flowgrpo_single_sample.log` first for environment,
   model-loading, Ray, CUDA, or OOM failures.
2. Check `dump_compare.json` for tensor-level precision drift on the configured
   `DEBUG_DUMP_STEPS`.
3. Check `metrics.json` for post-warmup performance regressions after
   `PERF_SKIP_STEPS`.
4. If the change is expected, rerun once on the same fixed runner, review the
   new `current/` artifacts, and refresh `baseline/` only after confirming the
   drift is intentional.
