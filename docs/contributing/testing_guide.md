# Testing Guide

Last updated: 07/14/2026.

This guide explains how to choose the right CI layer and how to add tests that run in the expected workflow.

## Test Taxonomy

| Question | Layer |
| --- | --- |
| Does it run fully on CPU and validate API, config, data, reward, adapter, or utility behavior? | L1 |
| Does it require GPU and run a tiny-random model for one or two end-to-end training steps? | L2 |
| Does it compare numerical metrics or performance across backends for short fixed-seed runs? | L3 |
| Does it use real model weights and real datasets to validate convergence curves? | L4 |

## L1 CPU Tests

L1 tests must end with `_on_cpu.py`. The CPU workflow writes a temporary `pytest.ini` that sets:

```ini
[pytest]
python_files = *_on_cpu.py
```

Use L1 for lightweight coverage of:

- Config defaults, validation, and Hydra composition.
- Dataset parsing, tensor conversion, collation helpers, and data-source metadata.
- Reward-score functions and reward managers that run without model inference.
- Loss functions, advantage computation, registries, and metric utilities.
- Pipeline or adapter boundary behavior that can be tested with mocks.

Do not use L1 for GPU kernels, Ray clusters, rollout engines, real checkpoints, or full trainer smoke scripts. Put those in L2.

## Placement Rules

Place tests under the top-level module they cover. For example:

- `verl_omni/trainer/...` -> `tests/trainer/...`
- `verl_omni/workers/...` -> `tests/workers/...`
- `verl_omni/utils/...` -> `tests/utils/...`
- `verl_omni/pipelines/...` -> `tests/pipelines/...`
- `verl_omni/reward_loop/...` -> `tests/reward_loop/...`

Special workflow folders such as `tests/special_e2e/` and `tests/special_sanity/` are reserved for non-L1 checks.

## Coverage

L1 reports line and branch coverage for `verl_omni`. The workflow produces:

- A terminal `term-missing` report in the job log.
- `coverage.xml` for tooling and artifact download.
- `pytest-coverage.txt` for the GitHub summary and artifact download.

Coverage should help identify untested API surfaces, but avoid adding brittle tests just to raise a number. Prefer tests that describe stable behavior and would catch a real regression.

Diff coverage thresholds should be introduced only after the baseline and exception policy are agreed by maintainers. Until then, contributors should use the report to inspect their changed modules.

## Local Commands

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

## Adding a New Test

1. Choose the lowest CI layer that can catch the regression.
2. Place the file under the matching `tests/<module>/` directory.
3. Use `_on_cpu.py` for L1 tests.
4. Keep fixtures small and deterministic.
5. Mock model loading and external services unless the test explicitly belongs in L2 or above.
6. Update workflow path filters if the new file lives outside existing trigger patterns.
