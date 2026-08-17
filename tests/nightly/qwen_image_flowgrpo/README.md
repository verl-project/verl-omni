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
`.github/workflows/l3_nightly.yml`. The workflow runs strict
nightly mode every day at 22:00 Asia/Shanghai and can also be triggered
manually.

Use this test as a regression signal for numerical drift and performance
changes. It is not part of the fast pull-request CI loop.

Workflow trigger modes:

- Scheduled runs use `nightly` mode and require an existing baseline artifact.
- `workflow_dispatch` supports `mode=nightly` or `mode=baseline`.
- To trigger a baseline run, open the workflow in GitHub Actions, choose
  **Run workflow**, and set `mode=baseline`.
- Adding the `L3-nightly-ci` label to a pull request triggers strict `nightly`
  mode for that PR.

## Requirements

The runner must have:

- 1 GPU by default, or set `NUM_GPUS` to match the runner.
- An installed `verl_omni` GPU environment with the rollout and training
  dependencies needed by Qwen-Image FlowGRPO.
- Local tiny-random policy and reward model directories.
- Enough local disk space for debug dumps, metrics, and logs under
  `OUTPUT_ROOT`.

## Run

Expected local model defaults:

- Policy: `~/models/tiny-random/Qwen-Image`
- Reward: `~/models/tiny-random/qwen3-vl`

Generate a local baseline first. This is the default behavior because
`BOOTSTRAP_MISSING_BASELINE=1` copies the current artifacts into `baseline/`
when no baseline exists:

```bash
NUM_GPUS=1 \
MODEL_PATH=/path/to/tiny-random/Qwen-Image \
REWARD_MODEL_PATH=/path/to/tiny-random/qwen3-vl \
OUTPUT_ROOT=/path/to/debug_dumps \
bash tests/nightly/qwen_image_flowgrpo/run_qwen_image_flowgrpo.sh
```

Then run strict nightly mode against the generated baseline:

```bash
NUM_GPUS=1 \
MODEL_PATH=/path/to/tiny-random/Qwen-Image \
REWARD_MODEL_PATH=/path/to/tiny-random/qwen3-vl \
OUTPUT_ROOT=/path/to/debug_dumps \
BOOTSTRAP_MISSING_BASELINE=0 \
bash tests/nightly/qwen_image_flowgrpo/run_qwen_image_flowgrpo.sh
```

## Baselines

The default layout is:

```text
outputs/debug_dumps/
|-- baseline/
|   `-- qwen_image_flowgrpo/
|-- current/
|   `-- qwen_image_flowgrpo/
`-- logs/
    `-- qwen_image_flowgrpo/
```

Each L3 test case uses its own subdirectory under `baseline/` and `current/`.
GitHub Actions stores every case inside one shared `l3-nightly-baseline`
artifact.

`BOOTSTRAP_MISSING_BASELINE=1` is the default, so a first run with no baseline
will copy current artifacts into `baseline/qwen_image_flowgrpo/`
and pass. Set `BOOTSTRAP_MISSING_BASELINE=0` for strict nightly comparison.

Use bootstrap mode only when intentionally creating or refreshing a reviewed
baseline. A real scheduled nightly should use strict mode so missing baselines,
missing dump files, or metric regressions fail the job.

In GitHub Actions, baseline mode uploads:

- `l3-nightly-baseline`, retained for 90 days.
- `l3-nightly-reports-<run_id>`, retained for 30 days.

Nightly mode downloads the latest non-expired `l3-nightly-baseline` artifact for
the configured `baseline_branch` input, then runs strict comparison. If no
matching baseline is found, the job fails before training starts.

Baseline refresh downloads the existing unified baseline when available, updates
only this test's subdirectory, and re-uploads the full baseline tree.

To refresh the production baseline:

1. Review the expected change and choose the branch whose baseline should be
   refreshed.
2. Trigger `l3_nightly` manually with `mode=baseline`.
3. Inspect the uploaded reports and baseline artifact before relying on later
   nightly runs.

The comparison outputs are written under
`current/qwen_image_flowgrpo/`:

- `metrics.json` contains aggregated timing, throughput, and memory metrics.
- `dump_compare.json` contains precision comparison results.
- `metrics.jsonl` contains step-level debug metrics from the run.
- `logs/qwen_image_flowgrpo.log` contains the console log.

Nightly mode uploads all current intermediate outputs as
`l3-nightly-current-<run_id>` and always uploads logs and reports as
`l3-nightly-reports-<run_id>`.

## Comparison Details

Performance comparison reads step-level metrics from `metrics.jsonl`, or falls
back to parsing the console log. It compares only steps after `PERF_SKIP_STEPS`,
which defaults to `2`, and compares the mean value for each comparable metric.

Performance tolerance:

- `PERF_THRESHOLD`, default `0.15`, meaning a 15% relative-change threshold.
- `timing_s/update_weights` uses a fixed 20% threshold. Weight synchronization is
  sensitive to vLLM-Omni, LoRA, IPC, free-cache, and layered summon details, and
  is usually a small part of total step time, so this wider tolerance avoids
  noisy failures while still catching material regressions.

Performance metrics:

- Lower is better: `perf/time_per_step`, `timing_s/gen`,
  `timing_s/old_log_prob`, `timing_s/reward`, `timing_s/update_actor`, and
  `timing_s/update_weights`.
- Higher is better: `perf/mfu/actor`, `perf/mfu/actor_infer`, and
  `perf/throughput`.

A lower-is-better metric fails when current is more than the metric threshold
above baseline. A higher-is-better metric fails when current is more than the
metric threshold below baseline. Other collected metrics, including
`perf/total_num_images`, `timing_s/step`, and derived `timing_per_image_ms/*`
values, are kept in the report but are not compared against the baseline.

Precision comparison recursively compares all tensors in matching `payload.pt`
debug dumps under `current/qwen_image_flowgrpo/` and
`baseline/qwen_image_flowgrpo/`. Missing files, missing tensors, and
shape mismatches fail the comparison. The default dumps cover driver-forward,
actor-forward, and LoRA-gradient payloads for `DEBUG_DUMP_STEPS`, which defaults
to `1,2`.

Precision tensors:

- batch tensors: `responses`, `log_probs`, `old_log_probs`,
  `advantages`, `sample_level_scores`, `sample_level_rewards`, `latents`,
  `all_latents`, and `all_timesteps`.
- Actor-forward tensors: every tensor returned by the diffusion FSDP engine
  forward/backward batch output, recursively flattened from the actor-forward
  `payload.pt`.
- LoRA-gradient tensors: every available LoRA parameter gradient under
  `gradients.<parameter_name>`.
- Optional actor-forward-step tensors are included only when
  `DEBUG_DUMP_FORWARD_STEPS=1`.

Precision thresholds:

- `PRECISION_ATOL`, default `1e-4`, maximum allowed absolute error.
- `PRECISION_RTOL`, default `1e-3`, maximum allowed relative error.
- `PRECISION_MIN_COS_SIM`, default `0.999`, minimum allowed cosine similarity.
- `PRECISION_IMAGE_ATOL`, default `2/255` (~`0.00784`), for decoded image
  tensors (`batch.responses`). One 8-bit LSB is `1/255` ≈ `0.00392` abs.
- `PRECISION_IMAGE_RTOL`, default `2e-2`, relative tolerance for
  `batch.responses`.
- `PRECISION_IMAGE_MIN_COS_SIM`, default `0.999`, cosine similarity floor for
  `batch.responses`.

Each tensor fails if `max_abs_err > atol`, `max_rel_err > rtol`, or
`cos_sim < min_cos_sim` for the threshold profile that applies to that key.

## Failure Triage

When the job fails:

1. Check `logs/qwen_image_flowgrpo.log` first for environment,
   model-loading, Ray, CUDA, or OOM failures.
2. Check `dump_compare.json` for tensor-level precision drift on the configured
   `DEBUG_DUMP_STEPS`.
3. Check `metrics.json` for post-warmup performance regressions after
   `PERF_SKIP_STEPS`.
4. If the change is expected, rerun once on the same fixed runner, review the
   new `current/` artifacts, and refresh `baseline/` only after confirming the
   drift is intentional.
