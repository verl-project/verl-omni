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

This directory is wired to
`.github/workflows/l3_qwen_image_flowgrpo_nightly.yml`. The workflow runs strict
nightly mode every day at 22:00 Asia/Shanghai and can also be triggered
manually.

Use this test as a regression signal for numerical drift and performance
changes. It is not part of the fast pull-request CI loop.

Workflow trigger modes:

- Scheduled runs use `nightly` mode and require an existing baseline artifact.
- `workflow_dispatch` supports `mode=nightly` or `mode=baseline`.
- Adding the `L3-baseline` label to a pull request triggers `baseline` mode for
  that PR.

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

In GitHub Actions, baseline mode uploads:

- `l3-qwen-image-flowgrpo-single-sample-baseline`, retained for 30 days.
- `l3-qwen-image-flowgrpo-single-sample-reports-<run_id>`, retained for 14 days.

Nightly mode downloads the latest non-expired
`l3-qwen-image-flowgrpo-single-sample-baseline` artifact for the configured
`baseline_branch` input, then runs strict comparison. If no matching baseline is
found, the job fails before training starts.

To refresh the production baseline:

1. Review the expected change and choose the branch whose baseline should be
   refreshed.
2. Trigger `l3_qwen_image_flowgrpo_nightly` manually with `mode=baseline`, or
   apply `L3-baseline` to the target pull request.
3. Inspect the uploaded reports and baseline artifact before relying on later
   nightly runs.

The comparison outputs are written under `current/`:

- `metrics.json` contains aggregated timing, throughput, and memory metrics.
- `dump_compare.json` contains precision comparison results.
- `metrics.jsonl` contains step-level debug metrics from the run.
- `logs/qwen_image_flowgrpo_single_sample.log` contains the console log.

Nightly mode uploads all current intermediate outputs as
`l3-qwen-image-flowgrpo-single-sample-current-<run_id>` and always uploads logs
and reports as `l3-qwen-image-flowgrpo-single-sample-reports-<run_id>`.

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
