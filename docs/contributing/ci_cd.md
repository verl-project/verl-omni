# CI/CD Layers

Last updated: 10/30/2026.

VeRL-Omni uses layered CI/CD checks so fast CPU feedback and expensive GPU or convergence validation can evolve independently.

| Layer | Purpose | Trigger | Hardware | Blocking scope | Output |
| --- | --- | --- | --- | --- | --- |
| L1 CPU API tests | Validate CPU-only APIs, configs, data utilities, adapters, rewards, and unit behavior | Pull requests with `ready-for-ci`, pushes to `main` and release branches | CPU | Required before merge | Pass/fail and coverage artifacts |
| L2 GPU smoke tests | Validate tiny-random GPU end-to-end training paths | Pull requests with `ready-for-ci`, usually after L1 is green | GPU | Required before merge for GPU-touching changes | Smoke logs and summaries |
| L3 nightly regression | Detect numerical drift and performance regressions | Scheduled or manual | Fixed GPU runners | Nightly regression signal | Metrics, baseline comparisons, and the dashboard |
| L4 convergence tests | Validate real recipe convergence | Weekly, release candidate, or manual | Production GPU cluster | Release readiness gate | Reward/loss curves and convergence reports |

L3 has one runnable scheduled workflow. It runs different algorithm cases on the v1 sync trainer. L4 remains a planned layer.

## L1 CPU API Tests

L1 is the default merge gate for code that can be validated without GPU hardware. Tests must run on CPU, avoid Ray clusters, avoid real checkpoint downloads, and use mocks or tiny in-memory fixtures when model boundaries need to be exercised.

The L1 workflow is `[.github/workflows/cpu_unit_tests.yml](../../.github/workflows/cpu_unit_tests.yml)`. It selects files ending in `_on_cpu.py`, runs `pytest` with coverage enabled for `verl_omni`, uploads coverage artifacts, and writes a short summary to the GitHub job summary. Tests run in parallel via `pytest-xdist` (one worker per runner core, test files distributed whole with `--dist loadfile`). L1 tests must not use Ray: worker processes are shared across files and Ray state is process-global. The single exception is `tests/reward_loop/test_cpu_native_reward_ray_on_cpu.py`, which runs as its own sequential pytest invocation in a fresh process.

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

The runnable L3 cases live under `tests/nightly/`. Shared debug hooks, metric
collection, dump comparison, and `run_l3_case.sh` live in that directory next
to the case folders. They run a
deterministic 20-step LoRA training window on local tiny-random policy and
reward models, then compare:

- debug dumps from selected steps for precision regressions; and
- post-warmup timing, throughput, and memory metrics for performance regressions.

Supported cases:
| Case | Script | GPUs | Algorithm |
| --- | --- | --- | --- |
| `qwen_image_flowgrpo` | `tests/nightly/qwen_image_flowgrpo/run_qwen_image_flowgrpo.sh` | 1 | FlowGRPO, OCR reward |
| `qwen_image_diffusion_nft` | `tests/nightly/qwen_image_diffusion_nft/run_qwen_image_diffusion_nft.sh` | 1 | DiffusionNFT, OCR reward, old-adapter rollout |
| `qwen_image_online_dpo` | `tests/nightly/qwen_image_online_dpo/run_qwen_image_online_dpo.sh` | 1 | Online DPO, OCR reward |

The workflow runs the ids in `ALL_L3_CASES`. Add a case there and to the `case`
input options.

Run one case manually with:

```bash
bash tests/nightly/model_name_my_algo/run_model_name_my_algo.sh
```

The GitHub workflow is `.github/workflows/l3_nightly.yml`. The job runs the
three cases sequentially in strict nightly mode every day at 22:00
Asia/Shanghai and can also be triggered manually with `workflow_dispatch`.

Nightly jobs run with `BOOTSTRAP_MISSING_BASELINE=0` so missing or stale
baselines fail closed. Baseline creation is manual only: run the workflow with
`mode=baseline`. Baseline mode downloads the existing unified baseline when one
exists, sets `L3_REFRESH_BASELINE=1` so each successful case replaces only its
own subdirectory, and uploads the reviewed baseline as the
`l3-nightly-baseline` artifact. The `case` input selects `all` or one case.
A single-case baseline run refreshes that subdirectory and uploads the other
downloaded cases unchanged. Publish that artifact only from a green run: a
failed baseline job can refresh the cases that finished and leave later cases
unchanged.

Strict nightly mode downloads the latest non-expired baseline artifact for the
configured baseline branch, checks that every case it is about to run has
`payload.pt` and `metrics.json`, and fails before creating the GPU runner when
one is missing. It then runs the comparison and uploads current debug dumps,
metrics, reports, and logs as artifacts even when the regression fails.

Until the baseline policy, artifact retention, ownership, and fixed runner
capacity are stable, L3 should remain outside the fast pull-request loop and
should be treated as a regression signal rather than a required merge gate.


### L3 Dashboard

The dashboard is a static page of performance and precision trends for nightly cases (view the [dashboard Github Pages](https://verl-project.github.io/verl-omni/dashboard/pages)). 

[`build_dashboard.py`](https://github.com/verl-project/verl-omni/blob/dashboard/build_dashboard.py) on **dashboard** branch downloads the shared `l3-nightly-reports-*` and `l3-nightly-baseline` artifacts, reads each case's `metrics.json` and `dump_compare.json`, and merges those points with the previously published [`history.json`](https://verl-project.github.io/verl-omni/dashboard/pages/history.json). The history file is what keeps runs visible after the Actions artifacts expire. 

[`.github/workflows/publish_dispatch.yml`](../../.github/workflows/publish_dispatch.yml) triggers publish workflow on dashboard branch every day at 08:00 Asia/Shanghai and rebuilds the page on pushes to the `dashboard` branch and publishes it to [GitHub Pages](https://verl-project.github.io/verl-omni/dashboard/pages/).

#### Register a New Nightly Case
Register every new nightly case in that same change. The report upload already picks up `current/**/metrics.json` and `current/**/dump_compare.json`, but the page charts a run only when the case id is listed in `EXPERIMENTS`. The id is the parent directory of those two files. The runner sets it with `L3_TEST_CASE`, which selects:

```text
${OUTPUT_ROOT}/current/${L3_TEST_CASE}/metrics.json
${OUTPUT_ROOT}/current/${L3_TEST_CASE}/dump_compare.json
${OUTPUT_ROOT}/baseline/${L3_TEST_CASE}/metrics.json
```

After adding a new test case, add the case object to `EXPERIMENTS` in [`build_dashboard.py`](https://github.com/verl-project/verl-omni/blob/dashboard/build_dashboard.py). Keep `layer` as `l3-nightly` for a case on the current nightly workflow. `label` is the experiment button text. A new case follows the same three fields. For a case whose directory is `model_name_my_algo`, the new entry is:

```python
EXPERIMENTS = [
    {
        "id": "qwen_image_flowgrpo",
        "label": "Qwen-Image + FlowGRPO",
        "layer": "l3-nightly",
    },
    ...
    { # new entry
        "id": "model_name_my_algo",
        "label": "Model Name + My Algo",
        "layer": "l3-nightly",
    },
]
```

`id` must equal `L3_TEST_CASE` and the directory name under both `current/` and `baseline/`. The builder stores that directory name as `case` and keeps the row only when `case` equals `experiment["id"]`.

Two further edits are needed only in these cases:

- Add a `{ "id", "label" }` entry to `LAYERS` when the case belongs to a new CI layer, and set the experiment `layer` to that id. Another L3 case stays on the existing `l3-nightly` layer.
- Add a metric key to `PERF_CHART_KEYS` when the case should plot a timing series that is not already listed. Keys under `timing_per_image_ms/` are drawn in milliseconds. Every other listed key is drawn in seconds. `metrics.json` still has to expose that key as `summary_for_compare.<key>.mean`. Precision charts need no key list: each numeric `mean_abs_err` in `dump_compare.json` becomes one series.

Leave `history.json` and `publish.yml` unchanged for a new case. The next dashboard build writes `experiments.<id>.rows` and `experiments.<id>.baseline` from the report and baseline artifacts. The baseline line comes from `baseline/<id>/metrics.json` inside `l3-nightly-baseline`.

## L4 Convergence Tests

L4 validates production-like recipes with real weights and real datasets. These checks are release-focused and should compare convergence curves against reviewed baselines.

L4 is not a replacement for L1 or L2. A recipe can converge while an API regression still exists, and a unit test can pass while a long-running recipe no longer converges.

## Contributor Expectations

When submitting changes, pick the lowest layer that can catch the regression:

- Prefer L1 for pure Python behavior and CPU-testable APIs.
- Add L2 when the behavior only exists in GPU runtime paths.
- Reserve L3/L4 changes for benchmark, regression, dashboard, and release-readiness work.

For test placement and naming rules, see `[testing_guide.md](testing_guide.md)`.
