# Testing Guide

Last updated: 07/14/2026.

This guide explains the test hierarchy for `verl_omni`, starting with L1 CPU tests and leaving room for higher layers such as L2 GPU smoke tests.

## Table of Contents

- [Test Hierarchy](#test-hierarchy)
- [Choosing a Layer](#choosing-a-layer)
- [L1 CPU Tests](#l1-cpu-tests)
  - [Scope](#scope)
  - [File Naming](#file-naming)
  - [Placement](#placement)
  - [Coverage](#coverage)
  - [Local Commands](#local-commands)
  - [Adding a New L1 Test](#adding-a-new-l1-test)
- [Future Layers](#future-layers)

## Test Hierarchy

The project uses a layered testing model. Lower layers should be cheaper, faster, and more deterministic. Choose the lowest layer that can catch the regression.

| Layer | Purpose | Typical Environment |
| --- | --- | --- |
| L1 | CPU-only unit and lightweight integration tests | GitHub-hosted CPU runner |
| L2 | GPU smoke tests for tiny end-to-end paths | GPU runner with tiny-random models |
| L3 | Backend or numerical comparison tests | GPU runner with fixed seeds |
| L4 | Real model and dataset validation | Scheduled or manually triggered jobs |

## Choosing a Layer

| Question | Layer |
| --- | --- |
| Does it run fully on CPU and validate API, config, data, reward, adapter, or utility behavior? | L1 |
| Does it require GPU and run a tiny-random model for one or two end-to-end training steps? | L2 |
| Does it compare numerical metrics or performance across backends for short fixed-seed runs? | L3 |
| Does it use real model weights and real datasets to validate convergence curves? | L4 |


## L1 CPU Tests

L1 is the current primary CI test layer. Use it for lightweight CPU coverage that can run quickly and deterministically on every pull request.

### Scope

Use L1 for lightweight coverage of:

- Config defaults, validation, and Hydra composition.
- Dataset parsing, tensor conversion, collation helpers, and data-source metadata.
- Reward-score functions and reward managers that run without model inference.
- Loss functions, advantage computation, registries, and metric utilities.
- Pipeline or adapter boundary behavior that can be tested with mocks.

Do not use L1 for GPU kernels, Ray clusters, rollout engines, real checkpoints, or full trainer smoke scripts. Put those in L2 or above.

### File Naming

L1 tests must end with `_on_cpu.py`. The CPU workflow writes a temporary `pytest.ini` that sets:

```ini
[pytest]
python_files = *_on_cpu.py
```

### Placement

Place tests under the top-level module they cover. For example:

- `verl_omni/trainer/...` -> `tests/trainer/...`
- `verl_omni/workers/...` -> `tests/workers/...`
- `verl_omni/utils/...` -> `tests/utils/...`
- `verl_omni/pipelines/...` -> `tests/pipelines/...`
- `verl_omni/reward_loop/...` -> `tests/reward_loop/...`

Special workflow folders such as `tests/special_e2e/` and `tests/special_sanity/` are reserved for non-L1 checks.

### Coverage

L1 reports line and branch coverage for `verl_omni`. The workflow produces:

- A terminal `term-missing` report in the job log.
- `coverage.xml` for tooling and artifact download.
- `pytest-coverage.txt` for the GitHub summary and artifact download.

Coverage should help identify untested API surfaces, but avoid adding brittle tests just to raise a number. Prefer tests that describe stable behavior and would catch a real regression.

Diff coverage thresholds should be introduced only after the baseline and exception policy are agreed by maintainers. Until then, contributors should use the report to inspect their changed modules.

### Local Commands

To run only L1-style tests locally:

```bash
printf '[pytest]\npython_files = *_on_cpu.py\n' > pytest.ini
pytest -s -x --asyncio-mode=auto --cov=verl_omni --cov-report=term-missing --cov-report=xml:coverage.xml tests/
```

To run a single new L1 test file while iterating:

```bash
pytest -s -x --asyncio-mode=auto tests/path/to/test_file_on_cpu.py
```

Delete the temporary `pytest.ini` if it is not part of your intended change.

### Adding a New L1 Test

1. Confirm the behavior can run fully on CPU.
2. Place the file under the matching `tests/<module>/` directory.
3. Name the file `test_<behavior>_on_cpu.py`.
4. Keep fixtures small and deterministic.
5. Mock model loading and external services.
6. Run the L1 command above before opening a PR.

## Future Layers

This guide currently defines L1 in detail because L1 is the main pull-request test layer. Add dedicated sections for L2, L3, or L4 when those workflows have stable ownership, trigger rules, naming conventions, and local commands.

When adding a new layer section, include:

1. Scope and examples.
2. Required environment.
3. File naming or folder conventions.
4. CI workflow trigger rules.
5. Local or manual run commands.
