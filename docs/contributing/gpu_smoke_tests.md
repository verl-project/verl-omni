# GPU Smoke Tests

Last updated: 07/06/2026.

GPU smoke tests validate GPU-only trainer, rollout, worker, reward, and
agent-loop paths with small workloads. They are intentionally lightweight: use
tiny checkpoints, dummy data, low step counts, and short timeouts.

For CPU-only tests, see [`cpu_unit_tests.yml`](../../.github/workflows/cpu_unit_tests.yml)
and the `_on_cpu.py` naming convention described in [Adding CI tests](../index.md#adding-ci-tests).

## Test Groups

The smoke suite is split by code-change coverage under `tests/gpu_smoke/`:

| Group script | CI label | Default GPUs | What it covers |
|---|---|---|---|
| [`run_gpu_smoke_core.sh`](../../tests/gpu_smoke/run_gpu_smoke_core.sh) | `ci-core` | 2 | Rollout, engines, agent loop, reward loop |
| [`run_gpu_smoke_omni_e2e.sh`](../../tests/gpu_smoke/run_gpu_smoke_omni_e2e.sh) | `ci-e2e-omni` | 2 | Qwen3-Omni end-to-end training (GSPO + LoRA) |
| [`run_gpu_smoke_diffusion_e2e.sh`](../../tests/gpu_smoke/run_gpu_smoke_diffusion_e2e.sh) | `ci-e2e-diffusion` | 4 | Diffusion end-to-end training (FlowGRPO, online DPO, DiffusionNFT) |

[`run_gpu_smoke_tests.sh`](../../tests/gpu_smoke/run_gpu_smoke_tests.sh) runs all
groups **sequentially** for local use. 


In CI, `.github/workflows/gpu_smoke.yml`
runs all groups **in parallel** on one 8-GPU runner when the full suite is
requested.

Shared helpers live in [`lib_gpu_smoke.sh`](../../tests/gpu_smoke/lib_gpu_smoke.sh).

## Add A New Smoke Test

1. Add the smallest useful test or launch script.

   Use regular pytest tests for worker, rollout, reward, and agent-loop coverage.
   Put trainer end-to-end scripts under `tests/special_e2e/` and keep them close
   to existing scripts such as `run_flowgrpo_qwen_image.sh`,
   `run_diffusionnft_qwen_image.sh`, or `run_online_dpo_qwen_image.sh`.

2. Pick the right GPU smoke group.

   | Change type | Register in |
   |---|---|
   | Trainer config/algo, rollout, reward managers, worker engines, agent loops, LoRA sync | `run_gpu_smoke_core.sh` |
   | Omni trainer end-to-end paths | `run_gpu_smoke_omni_e2e.sh` |
   | Diffusion trainer end-to-end paths | `run_gpu_smoke_diffusion_e2e.sh` |

3. Register the test with `run_test`.

   Each group script sources `lib_gpu_smoke.sh`, calls `gpu_smoke_init`, then
   registers commands through `run_test`. Use the next numeric test id in that
   group and pass `CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}"` to commands that
   need GPUs.

   ```bash
   run_test 3 "my trainer e2e" \
       env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" \
       bash tests/special_e2e/run_my_trainer_smoke.sh
   ```

   For omni e2e tests that require exactly 2 GPUs (tensor parallel size is pinned
   in the smoke config), keep `NUM_GPUS=2` even when overriding device ids.

4. Keep the test self-contained.

   Generate any dummy data inside the script, use tiny-random model checkpoints
   where possible, avoid network-only assumptions, and assert that the command
   exits successfully. The shared smoke helper already sets
   `PYTHONUNBUFFERED=1`, `RAY_DEDUP_LOGS=0`, creates a log directory, and stops
   Ray between tests unless `GPU_SMOKE_SKIP_RAY_STOP=1` is set.

5. Update workflow paths if needed.

   If the new test lives outside the existing path filters in
   `.github/workflows/gpu_smoke.yml`, add the path so PR updates trigger the GPU
   smoke workflow. Current filters include `verl_omni/**`, `tests/gpu_smoke/**`,
   `tests/agent_loop/**`, `tests/reward_loop/**`, `tests/workers/**`,
   `tests/special_e2e/**`, and `pyproject.toml`.

## Run Locally

### Prerequisites

Install the GPU stack following [`docs/start/install.md`](../start/install.md).
Smoke tests need a working CUDA environment, Ray, and the pinned `verl` /
`vllm-omni` dependencies. From the repo root:

```bash
uv pip install -e ".[gpu]" --torch-backend=auto
uv pip install -e ".[vllm-omni,train,dev]"
```

Run `ray stop --force` before starting if a previous Ray session is still active.

### Run all groups (sequential)

Requires at least 4 GPUs total because the diffusion group defaults to 4 GPUs.
Groups run one after another and reuse the same visible devices:

```bash
bash tests/gpu_smoke/run_gpu_smoke_tests.sh
```

### Run one group

Use this when you only changed code covered by a single group, or when you have
fewer GPUs available:

```bash
bash tests/gpu_smoke/run_gpu_smoke_core.sh
bash tests/gpu_smoke/run_gpu_smoke_omni_e2e.sh
bash tests/gpu_smoke/run_gpu_smoke_diffusion_e2e.sh
```

Each group script accepts `--help` for its options.

### Select GPUs and GPU count

Override the GPU count for any group. `--num-gpus` accepts any positive integer:

```bash
bash tests/gpu_smoke/run_gpu_smoke_core.sh --num-gpus 2
bash tests/gpu_smoke/run_gpu_smoke_diffusion_e2e.sh --num-gpus 4
```

Select local CUDA devices with environment variables:

```bash
CUDA_VISIBLE_DEVICES=0,2 NUM_GPUS=2 bash tests/gpu_smoke/run_gpu_smoke_omni_e2e.sh
```

Or pass device ids directly to a group script:

```bash
bash tests/gpu_smoke/run_gpu_smoke_diffusion_e2e.sh \
    --num-gpus 4 \
    --cuda-visible-devices 0,2,4,5
```

Keep `NUM_GPUS` aligned with the number of visible devices unless you are
intentionally testing a specific placement behavior.

### Mirror CI parallel layout (8 GPUs)

When you have 8 GPUs and want to reproduce the CI `ready-for-ci` layout, run
the three groups in parallel on disjoint device sets:

```bash
mkdir -p logs/gpu_smoke/parallel

GPU_SMOKE_SKIP_RAY_STOP=1 bash tests/gpu_smoke/run_gpu_smoke_core.sh \
    --num-gpus 2 --cuda-visible-devices 0,1 \
    > logs/gpu_smoke/parallel/ci-core.log 2>&1 &

GPU_SMOKE_SKIP_RAY_STOP=1 bash tests/gpu_smoke/run_gpu_smoke_omni_e2e.sh \
    --num-gpus 2 --cuda-visible-devices 2,3 \
    > logs/gpu_smoke/parallel/ci-e2e-omni.log 2>&1 &

GPU_SMOKE_SKIP_RAY_STOP=1 bash tests/gpu_smoke/run_gpu_smoke_diffusion_e2e.sh \
    --num-gpus 4 --cuda-visible-devices 4,5,6,7 \
    > logs/gpu_smoke/parallel/ci-e2e-diffusion.log 2>&1 &

wait
ray stop --force || true
```

Set `GPU_SMOKE_SKIP_RAY_STOP=1` so parallel groups do not stop each other's Ray
clusters mid-run. CI uses the same pattern in `.github/workflows/gpu_smoke.yml`.

### Logs

Logs are written under `logs/gpu_smoke/<group>/<timestamp>/`. Each test gets its
own log file and the group writes a `summary.log`. Check `summary.log` first when
a group fails locally.

## Run In CI

Pull-request GPU smoke runs are **label-driven**. 

### PR labels for GPU smoke

Apply one label to the pull request to trigger the matching GPU smoke job:

| Label | Runner | What runs |
|---|---|---|
| `ready-for-ci` | Up to 8 GPUs (L20x2/L20x4/L20x8) | Automatically selected groups based on the PR diff |
| `ci-core` | 2 GPUs (L20x2) | `run_gpu_smoke_core.sh` only |
| `ci-e2e-omni` | 2 GPUs (L20x2) | `run_gpu_smoke_omni_e2e.sh` only |
| `ci-e2e-diffusion` | 4 GPUs (L20x4) | `run_gpu_smoke_diffusion_e2e.sh` only |

Pick the **smallest label that covers your change** during development. Use
`ready-for-ci` before merge when you want CI to choose all required GPU smoke
groups for the PR.

### Automatic `ready-for-ci` selection

For pull requests, `ready-for-ci` reads the changed file list and selects the
smallest known group set that covers those paths. The selector also assigns
non-overlapping `CUDA_VISIBLE_DEVICES` ranges so all selected groups fit within
8 GPUs:

| Selected groups | Device assignment |
|---|---|
| `ci-core` | `0,1` |
| `ci-e2e-omni` | `0,1` |
| `ci-e2e-diffusion` | `0,1,2,3` |
| `ci-core` + `ci-e2e-omni` | `0,1` and `2,3` |
| all groups | `0,1`, `2,3`, and `4,5,6,7` |

The mapping is intentionally conservative:

| Changed paths | Selected group |
|---|---|
| `verl_omni/workers/**`, `verl_omni/agent_loop/**`, `verl_omni/reward_loop/**`, matching GPU tests | `ci-core` |
| `verl_omni/trainer/omni/**`, `verl_omni/trainer/config/omni/**`, `qwen3_omni_thinker.py`, omni e2e scripts | `ci-e2e-omni` |
| `verl_omni/trainer/diffusion/**`, `verl_omni/trainer/config/diffusion/**`, `verl_omni/pipelines/**`, `verl_omni/models/diffusers/**`, diffusion e2e scripts | `ci-e2e-diffusion` |
| shared CI/test helpers, package metadata, shared trainer config, or unknown GPU-smoke paths | all groups |

Pushes to `main` and `v0.*` still run all groups. If future group definitions
would require more than 8 GPUs in one `ready-for-ci` run, CI fails closed instead
of oversubscribing devices.

### Other CI workflows on PRs

Several non-GPU workflows also run when a PR carries any label whose name
contains `ci` (including the labels above):

- `pre-commit.yml`
- `cpu_unit_tests.yml`
- `sanity.yml`
- `doc.yml`
- `check-pr-title.yml`

So adding `ci-core` triggers both the core GPU smoke group and the CPU checks.

### Label auto-removal

When new commits are pushed or a PR is reopened, `.github/workflows/drop-ci-labels.yml`
removes all labels whose names contain `ci`. Re-apply the label you need after
each push.

### Debugging CI failures

The workflow uploads `logs/gpu_smoke/**` as an artifact. For the full suite,
parallel group logs are also under `logs/gpu_smoke/parallel/`.
