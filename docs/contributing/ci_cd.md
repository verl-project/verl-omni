# CI/CD Layers

Last updated: 07/14/2026.

VeRL-Omni uses layered CI/CD checks so fast CPU feedback and expensive GPU or convergence validation can evolve independently.

| Layer | Purpose | Trigger | Hardware | Blocking scope | Output |
| --- | --- | --- | --- | --- | --- |
| L1 CPU API tests | Validate CPU-only APIs, configs, data utilities, adapters, rewards, and unit behavior | Pull requests with `ready-for-ci`, pushes to `main` and release branches | CPU | Required before merge | Pass/fail and coverage artifacts |
| L2 GPU smoke tests | Validate tiny-random GPU end-to-end training paths | Pull requests with `ready-for-ci`, usually after L1 is green | GPU | Required before merge for GPU-touching changes | Smoke logs and summaries |
| L3 nightly regression | Detect numerical drift and performance regressions | Scheduled or manual | Fixed GPU runners | Nightly regression gate | Metrics and baseline comparisons |
| L4 convergence tests | Validate real recipe convergence | Weekly, release candidate, or manual | Production GPU cluster | Release readiness gate | Reward/loss curves and convergence reports |

L3 and L4 are planned layers and are not supported by runnable CI tests yet.

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

L3 should not be part of the fast pull-request loop until its baseline policy and runner capacity are stable.

## L4 Convergence Tests

L4 validates production-like recipes with real weights and real datasets. These checks are release-focused and should compare convergence curves against reviewed baselines.

L4 is not a replacement for L1 or L2. A recipe can converge while an API regression still exists, and a unit test can pass while a long-running recipe no longer converges.

## Contributor Expectations

When submitting changes, pick the lowest layer that can catch the regression:

- Prefer L1 for pure Python behavior and CPU-testable APIs.
- Add L2 when the behavior only exists in GPU runtime paths.
- Reserve L3/L4 changes for benchmark, regression, dashboard, and release-readiness work.

For test placement and naming rules, see `[testing_guide.md](testing_guide.md)`.
