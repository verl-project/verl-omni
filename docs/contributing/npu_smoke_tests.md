# NPU Smoke Tests

Last updated: 10/08/2026.

This guide explains how to add CI test cases for Huawei Ascend devices in verl-omni.

The GitHub Actions entry point is
[`.github/workflows/npu_smoke.yml`](../../.github/workflows/npu_smoke.yml).
The common local runner is
[`tests/npu_smoke/run_npu_smoke_tests.sh`](../../tests/npu_smoke/run_npu_smoke_tests.sh).

## Test Cases

| ID | Name | Test entry point | Default NPUs | Status |
|---|---|---|---:|---|
| 0 | vLLM-Omni rollout + sleep/wake-up | `tests/workers/rollout/rollout_vllm/test_vllm_omni_generate_npu.py` | 8 | Enabled |
| 1 | Qwen-Image FlowGRPO trainer e2e | `tests/special_e2e/run_flowgrpo_qwen_image_npu.sh` | 8 | Temporarily skipped |

Test 1 remains registered in both the runner and workflow. However,
`run_npu_smoke_tests.sh` currently forces `RUN_TEST[1]=0`, so the test is
reported as `SKIP`. Remove this temporary override after the FlowGRPO runtime
issue is resolved.

## Add a New NPU Smoke Test

### 1. Add the smallest useful test entry point

Choose the location based on the scope of the test:

| Test type | Recommended location |
|---|---|
| Rollout, worker, engine, or sleep/wake-up | A pytest file under `tests/workers/` |
| Trainer end-to-end path | A shell script under `tests/special_e2e/` |
| Shared NPU smoke logic | `tests/npu_smoke/` |

Keep the test small and reproducible:

- Use tiny-random checkpoints or model weights already cached on the CI runner.
- Use the smallest practical dataset, batch size, and number of training steps.
- Do not require model or dataset downloads from the public internet at runtime.
- Return exit code 0 on success and preserve the original nonzero code on failure.
- Do not reuse Ray clusters or worker processes left by another job.

For example, add a pytest test:

```python
def test_my_npu_feature():
    # Arrange the smallest NPU workload.
    ...
```

Or add an end-to-end script:

```bash
#!/usr/bin/env bash
set -euo pipefail

export DEVICE_NAME=npu
python -m verl_omni.trainer.main_ppo \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1
```

### 2. Register the test in the common runner

Open
[`run_npu_smoke_tests.sh`](../../tests/npu_smoke/run_npu_smoke_tests.sh)
and assign the next unused numeric ID. For example, use ID 2 for a new test.

First register the ID in `RUN_TEST`:

```bash
declare -A RUN_TEST=([0]=1 [1]=1 [2]=1)
```

Then add a `run_selected_test` call in the execution section:

```bash
cleanup_runtime
run_selected_test 2 "my NPU smoke test" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \
        NUM_NPUS="${NUM_NPUS}" \
    pytest -s tests/workers/test_my_npu_feature.py
```

For an end-to-end script:

```bash
cleanup_runtime
run_selected_test 2 "my trainer e2e" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \
        NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_my_npu_smoke.sh
```

Also update the script's `--help` output:

```text
Tests:
  0  vllm-omni rollout + sleep/wake_up
  1  FlowGRPO trainer e2e
  2  my NPU smoke test
```

Do not reuse an existing ID. Use a test name that clearly identifies the
component and behavior being covered.

### 3. Configure the NPU count and visible devices

The common runner uses eight NPUs by default and builds
`ASCEND_RT_VISIBLE_DEVICES` from `--num-npus`. Each test command must pass:

```bash
env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \
    NUM_NPUS="${NUM_NPUS}" \
    <test-command>
```

If a test requires a fixed number of devices, validate the value in the test
script and fail with a clear message instead of silently using an invalid
parallel configuration.

### 4. Update the workflow test groups

To run the new test under an existing label, add its ID to the
`resolve-groups` step in
[`npu_smoke.yml`](../../.github/workflows/npu_smoke.yml).
For example, include Test 2 in the full mode:

```yaml
if [[ "${mode}" == "all" ]]; then
  plan_json='{
    "test_ids": ["0", "1", "2"],
    "group_count": 3,
    "num_npus": 8,
    "runner_size": 8
  }'
fi
```

To run the test independently, add a mode and pull-request label, then update
both `Resolve NPU CI mode` and `resolve-groups`.

If the new test lives outside the current workflow path filters, add its path.
The current filters include:

- `verl_omni/**`
- `tests/npu_smoke/**`
- `tests/workers/**`
- `tests/special_e2e/**`
- `pyproject.toml`
- `.github/workflows/npu_smoke.yml`
- `.github/vllm_omni_pin.txt`

### 5. Handle models and datasets

The default Qwen-Image tiny-random checkpoint path is:

```text
${HOME}/.cache/modelscope/hub/models/tiny-random/Qwen-Image
```

The workflow builds the checkpoint only when the directory is absent:

```bash
MODEL_PATH="${HOME}/.cache/modelscope/hub/models/tiny-random/Qwen-Image"
if [[ ! -d "${MODEL_PATH}" ]]; then
  python tests/special_e2e/build_qwen_image_tiny_random.py \
    --output-dir "${MODEL_PATH}"
fi
```

Prefer weights already cached on the runner. If a tiny checkpoint must be
generated:

- Generate it only when the target directory does not exist.
- Allow the default path to be overridden with an environment variable.
- Do not overwrite a shared CI cache.
- Remove only temporary files created by the current test.

### 6. Validate locally

Run tests from the repository root in an Ascend environment with CANN, vLLM,
vLLM-Ascend, and vLLM-Omni installed. Check device availability and stop any
stale Ray runtime first:

```bash
npu-smi info
ray stop --force
```

Run the default test set:

```bash
bash tests/npu_smoke/run_npu_smoke_tests.sh
```

Run Test 0 only:

```bash
bash tests/npu_smoke/run_npu_smoke_tests.sh --num-npus 8 0
```

Run Test 0 with a different NPU count:

```bash
bash tests/npu_smoke/run_npu_smoke_tests.sh --num-npus 4 0
```

Select devices explicitly:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,2,4,6 NUM_NPUS=4 \
  bash tests/npu_smoke/run_npu_smoke_tests.sh 0
```

While Test 1 is disabled, requesting it reports `SKIP` without starting
FlowGRPO:

```bash
bash tests/npu_smoke/run_npu_smoke_tests.sh --num-npus 8 1
```

After registering Test 2, run it independently:

```bash
bash tests/npu_smoke/run_npu_smoke_tests.sh --num-npus 8 2
```

Logs are written to `logs/npu_smoke/<timestamp>/`. Check `summary.log`
first, then inspect the corresponding `test_<id>.log` for a failed test.

Before committing, run the relevant static checks:

```bash
bash -n tests/npu_smoke/run_npu_smoke_tests.sh
pre-commit run --all-files
```

The first command performs a Bash syntax check only. It does not start Ray,
allocate NPUs, or execute any smoke test. No output and exit code 0 mean that
the script syntax is valid.

The second command runs every hook configured by the repository against all
tracked files. Depending on the repository configuration, this can include
Ruff formatting and linting, type checks, documentation checks, license checks,
Python compilation, and generated-configuration consistency checks. Some hooks
may modify files automatically; review those changes and rerun pre-commit until
all relevant hooks pass.

When a change only touches this guide, use the narrower check instead:

```bash
pre-commit run --files docs/contributing/npu_smoke_tests.md
```

The full `--all-files` check is recommended before submitting a pull request,
especially when the runner, workflow, Python tests, or training scripts also
changed.

## CI Triggers

The workflow responds to relevant file changes for:

- Pushes to `main` or `v0.*`, which run the full test set.
- Pull requests targeting `main` or `v0.*` when a CI label is applied.
  Opening, synchronizing, or reopening a pull request does not start an NPU
  runner. New commits automatically remove labels whose names contain `ci`,
  so the required label must be applied again.

Without one of the labels below, a pull request does not consume an NPU runner.
The pull request must also modify a path covered by the workflow filters, such
as `verl_omni/**`, `tests/npu_smoke/**`, `tests/workers/**`, or
`tests/special_e2e/**`.

Pull-request labels select the requested test scope:

| Label | Mode | Requested tests |
|---|---|---|
| `ready-for-ci` | all | Test 0 and Test 1 |
| `ci-npu` | all | Test 0 and Test 1 |
| `ci-npu-rollout` | rollout | Test 0 |
| `ci-npu-flowgrpo` | flowgrpo | Test 1 |

While Test 1 is disabled, the runner reports it as `SKIP` even when the
workflow requests it.

## Runtime Environment

The current CI configuration uses:

- Runner: `linux-aarch64-a2b4-8`
- NPU count: 8
- Timeout: 120 minutes
- Shared memory: 16 GiB
- Container image:
  `swr.cn-north-4.myhuaweicloud.com/mindspeed/pr-verl-omni-a2:latest`

Before running the tests, the workflow prints the CANN installation details,
`npu-smi info`, relevant Ascend environment variables, and the installed
vLLM, vLLM-Ascend, and vLLM-Omni versions.

## Runtime Cleanup and Logs

Before each test, the common runner:

1. Runs `ray stop --force`.
2. Terminates stale `DiffusionWorker`, `VLLMWorker`, and
   `vLLMOmniHttpServer` processes.
3. Waits five seconds and force-kills any matching processes that remain.
4. Prints `npu-smi info`.

Do not run unrelated workloads with matching process names on the same machine
while the smoke suite is running.

Logs are written to:

```text
logs/npu_smoke/<timestamp>/
```

Each executed test writes `test_<id>.log`. The `summary.log` file records
the `PASS`, `FAIL`, or `SKIP` result and elapsed time. When a test fails,
check its log, NPU memory usage, the CANN environment, and the installed vLLM
component versions.
