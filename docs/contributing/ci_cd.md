# CI/CD Layers

Last updated: 09/14/2026.

VeRL-Omni uses layered CI/CD checks so fast CPU feedback and expensive GPU or convergence validation can evolve independently.

| Layer | Purpose | Trigger | Hardware | Blocking scope | Output |
| --- | --- | --- | --- | --- | --- |
| L1 CPU API tests | Validate CPU-only APIs, configs, data utilities, adapters, rewards, and unit behavior | Pull requests with `ready-for-ci`, pushes to `main` and release branches | CPU | Required before merge | Pass/fail and coverage artifacts |
| L2 GPU smoke tests | Validate tiny-random GPU end-to-end training paths | Pull requests with `ready-for-ci`, usually after L1 is green | GPU | Required before merge for GPU-touching changes | Smoke logs and summaries |
| L3 nightly regression | Detect numerical drift and performance regressions | Scheduled or manual | Fixed GPU runners | Nightly regression signal | Metrics and baseline comparisons |
| L4 convergence tests | Real-recipe precision: train-infer gap, finite grad, 100-step val reward floor | Manual on a production GPU node; weekly / RC workflow is not wired yet | 8-GPU node with real weights and datasets | Release-readiness signal, not a PR merge gate | `report.json` (gates + recorded perf), `metrics.json` (per-step loss + val every `test_freq`), `metrics.jsonl`, console logs |

L3 has one runnable scheduled workflow for the Qwen-Image FlowGRPO
single-sample regression. L4 has runnable local/cluster scripts for the
Qwen-Image and SD3.5 OCR LoRA v1 recipes; none are wired to GitHub Actions yet.

## L1 CPU API Tests

L1 is the default merge gate for code that can be validated without GPU hardware. Tests must run on CPU, avoid Ray clusters, avoid real checkpoint downloads, and use mocks or tiny in-memory fixtures when model boundaries need to be exercised.

The L1 workflow is `[.github/workflows/cpu_unit_tests.yml](../../.github/workflows/cpu_unit_tests.yml)`. It selects files ending in `_on_cpu.py`, runs `pytest` with coverage enabled for `verl_omni`, uploads coverage artifacts, and writes a short summary to the GitHub job summary.

Add or update L1 tests when changing:

- Config dataclasses or Hydra config wiring.
- Dataset loading, collation, and data utility behavior that can run on CPU.
- Reward managers, rule rewards, and reward-score adapters that do not require model inference.
- Trainer math, loss registries, and utility functions.
- Pipeline adapter boundaries that can be covered with mocks instead of real model weights.

## L2 GPU Smoke Tests

L2 covers tiny-random end-to-end training paths that need GPU runtime, rollout engines, Ray, or backend-specific kernels. These checks should prove the trainer reaches the configured smoke-test steps without exceptions, OOMs, or Ray failures. They are not accuracy or convergence checks.

Use L2 for changes that affect GPU rollout, trainer entrypoints, backend integration, or full scripts under `tests/special_e2e/`.

## L3 Nightly Regression

L3 is intended for scheduled numerical and performance regression tracking. These tests should run fixed-seed, short training windows and compare key metrics against reviewed baselines, such as loss, reward, KL, log probability, gradient norm, throughput, step time, and memory peak.

The current runnable L3 case is
`tests/nightly/qwen_image_flowgrpo/`. It runs a deterministic
20-step Qwen-Image FlowGRPO LoRA training window on local tiny-random policy and
reward models, then compares:

- debug dumps from selected steps for precision regressions; and
- post-warmup timing, throughput, and memory metrics for performance regressions.

Run it manually with:

```bash
bash tests/nightly/qwen_image_flowgrpo/run_qwen_image_flowgrpo.sh
```

The GitHub workflow is `.github/workflows/l3_nightly.yml`. The current job runs
strict nightly
mode every day at 22:00 Asia/Shanghai and can also be triggered manually with
`workflow_dispatch`.

Nightly jobs run with `BOOTSTRAP_MISSING_BASELINE=0` so missing or stale
baselines fail closed. Baseline creation is manual only: run the workflow with
`mode=baseline`. Baseline mode uses `BOOTSTRAP_MISSING_BASELINE=1`, downloads the
existing unified baseline when available, updates each test's subdirectory, and
uploads the reviewed baseline as the `l3-nightly-baseline` artifact.

Strict nightly mode downloads the latest non-expired baseline artifact for the
configured baseline branch, runs the comparison, and uploads current debug
dumps, metrics, reports, and logs as artifacts even when the regression fails.

Until the baseline policy, artifact retention, ownership, and fixed runner
capacity are stable, L3 should remain outside the fast pull-request loop and
should be treated as a regression signal rather than a required merge gate.

## L4 Convergence Tests

L4 validates production-like recipes with real weights and real datasets. It
is release-focused and is **not** a replacement for L1 or L2.

L4 does **not** compare against a stored baseline and does **not** fail on
throughput or step time. Those numbers are recorded for humans to inspect.

### Current runnable cases

#### Qwen-Image OCR LoRA v1

`tests/convergence/qwen_image_ocr_lora_v1/` wraps
`examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_v1.sh` for an
8-GPU, 100-step v1 sync LoRA OCR job. It enables
`actor_rollout_ref.rollout.calculate_log_probs=true` and keeps rollout-correction
bypass mode off so the actor recomputes `old_log_probs`. Validation runs every
20 steps (`trainer.test_freq=20`) and on the last step.

```bash
bash tests/convergence/qwen_image_ocr_lora_v1/run_qwen_image_ocr_lora_v1.sh
```

Defaults:

- Policy: `$WORKSPACE/models/Qwen-Image` (`MODEL_PATH`)
- Reward: `$WORKSPACE/models/Qwen3-VL-8B-Instruct` (`REWARD_MODEL_PATH`)
- Data: `$WORKSPACE/data/ocr/qwen_image/{train,test}.parquet`
- `WORKSPACE` defaults to `$HOME`. `NUM_GPUS` defaults to `8`.
- `VAL_REWARD_MIN` defaults to `0.9`.

#### SD3.5 OCR LoRA v1

`tests/convergence/sd35_medium_ocr_lora_v1/` wraps
`examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh` for an 8-GPU
(6 actor+rollout, 2 reward) Route-A sync LoRA OCR job: TP-sharded rollout/reward,
FSDP offload, `train_batch_size=16`, `rollout.n=16`, and the same train-infer
and grad gates. Validation runs every 20 steps and on the last step.

```bash
bash tests/convergence/sd35_medium_ocr_lora_v1/run_sd35_medium_ocr_lora_v1.sh
```

Defaults:

- Policy: `stabilityai/stable-diffusion-3.5-medium` (`MODEL_PATH`)
- Reward: `Qwen/Qwen2.5-VL-3B-Instruct` (`REWARD_MODEL_PATH`)
- Data: `$WORKSPACE/data/ocr/sd3/{train,test}.parquet`
- `WORKSPACE` defaults to `$HOME`.
- `VAL_REWARD_MIN` defaults to `0.6` (placeholder until calibrated on cluster).

Gate logic lives in `collect_report.py`. The collector's own tests are L1
(`tests/convergence/test_collect_report_on_cpu.py`).

### What is gated

| Gate | Metric | Fail when |
| --- | --- | --- |
| Train-infer consistency | `training/rollout_probs_diff_mean` if present, else `rollout_corr/logprob_abs_diff_mean` | missing, non-finite, or any post-warmup step **> 0.01** |
| Grad | `actor/grad_norm` | NaN / Inf |
| Val reward at step 100 | `val-core/*/reward/mean@*` | missing, or any source **< `VAL_REWARD_MIN`** (default **0.9**) |

`perf/time_per_step` and related `timing_s/*` keys are copied into
`report.json` under `"perf"` with no pass/fail.

`metrics.json` records per-step training loss (`actor/loss/mean` or `actor/loss`)
and `val-core/*/reward/mean@*` at each `test_freq` step. `metrics.jsonl` is the
raw trainer log dump used to build both files.

There is no `.github/workflows/l4_*.yml` yet. Keep L4 off the default
pull-request merge path.

## Contributor Expectations

When submitting changes, pick the lowest layer that can catch the regression:

- Prefer L1 for pure Python behavior and CPU-testable APIs.
- Add L2 when the behavior only exists in GPU runtime paths.
- Reserve L3/L4 changes for benchmark, regression, dashboard, and release-readiness work.

For test placement and naming rules, see `[testing_guide.md](testing_guide.md)`.
