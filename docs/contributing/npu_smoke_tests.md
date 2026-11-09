# NPU Smoke Tests

Last updated: 10/29/2026.

This guide explains how to add CI test cases for Huawei Ascend devices in verl-omni.

The workflow entry point is .github/workflows/npu_smoke.yml. The suite is split
into these runners:

- tests/npu_smoke/run_npu_smoke_core.sh runs rollout, worker, and engine checks.
- tests/npu_smoke/run_npu_smoke_e2e.sh runs end-to-end trainer checks.
- tests/npu_smoke/run_npu_smoke_tests.sh runs both groups sequentially for local use.
- tests/npu_smoke/lib_npu_smoke.sh contains shared cleanup, logging, and summaries.

## Test Cases

| ID | Group | Name | Test entry point | Default NPUs |
|---|---|---|---|---:|
| 0 | core | vLLM-Omni rollout + sleep/wake-up | tests/workers/rollout/rollout_vllm/test_vllm_omni_generate_npu.py | 8 |
| 1 | e2e | Qwen-Image FlowGRPO trainer | tests/special_e2e/run_flowgrpo_qwen_image_npu.sh | 8 |
| 2 | e2e | Wan2.2 DanceGRPO trainer with Qwen2.5-VL reward | tests/special_e2e/run_dancegrpo_wan22_npu.sh | 8 |
| 3 | core | Diffusers FSDP engine inference and training | tests/workers/test_diffusers_fsdp_engine_npu.py | 8 |
| 4 | core | Diffusion agent loop, request and step execution | tests/agent_loop/test_diffusion_agent_loop_npu.py::test_diffusion_single_turn_npu | 8 |
| 5 | core | Unseeded multi-worker rollout diversity | tests/agent_loop/test_diffusion_agent_loop_npu.py::test_rollout_without_seed_npu | 8 |
| 6 | core | Seeded multi-worker rollout reproducibility | tests/agent_loop/test_diffusion_agent_loop_npu.py::test_rollout_seed_reproducibility_npu | 8 |
| 7 | core | Standalone vLLM reward engine and OCR scoring | tests/reward_loop/test_reward_engine_npu.py | 8 |

The combined runner has eight entries: six core (0, 3, 4, 5, 6, 7) and two e2e
(1, 2). The core group collects seven pytest cases because entry 4 is
parametrized into request and step execution. The two e2e entries each run a
trainer workload; they are not pytest cases.

Test 3 ports the `fsdp` case of `tests/workers/test_diffusers_fsdp_engine.py`
(GPU core ID 2). It reuses the GPU test's synthetic trajectories and engine
inference/training exercise, with `_native_npu` attention and sequence parallel
size 1. It requires at least two visible NPUs and a local tiny Qwen-Image model.
FSDP2 and sequence-parallel execution are not covered by this NPU case.
The test checks finite inference log probabilities, training loss, and gradient
norm; it does not establish GPU/NPU numerical parity or model convergence.

Run only the migrated case from the repository root after configuring the
[runtime environment](#runtime-environment):

```bash
DEVICE_NAME=npu NUM_NPUS=2 \
MODEL_PATH="${HOME}/.cache/modelscope/hub/models/tiny-random/Qwen-Image" \
python3 -m pytest -v -s tests/workers/test_diffusers_fsdp_engine_npu.py
```

The same case runs through `run_npu_smoke_core.sh` and the combined
`run_npu_smoke_tests.sh` entry point. Missing hardware or model files fail
explicitly rather than producing an all-skipped smoke success.

### Agent-loop execution and rollout seeds

Tests 4-6 reuse the GPU cases to check request/step execution, unseeded rollout
diversity, and seeded reproducibility. They use tiny Qwen-Image, rollout TP=2,
and two agent-loop workers. `NUM_NPUS` must be even and at least 2.

The workflow prepares the local checkpoint. For a local run, prepare it first:

```bash
python3 tests/special_e2e/build_qwen_image_tiny_random.py \
    --output-dir "${HOME}/.cache/modelscope/hub/models/tiny-random/Qwen-Image"
```

After configuring the [runtime environment](#runtime-environment), run these
cases from the repository root:

```bash
export DEVICE_NAME=npu NUM_NPUS=8
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HCCL_NPU_SOCKET_PORT_RANGE=auto
python3 -m pytest -v -s tests/agent_loop/test_diffusion_agent_loop_npu.py
```

For a two-device local check, set both visibility variables to `0,1` and
`NUM_NPUS=2`. Missing hardware or model files fail the tests.

### Standalone reward engine

Test 7 checks standalone vLLM reward-server startup and OCR scoring with a tiny
random Qwen2.5-VL model generated offline. Eight NPUs run four TP=2 reward
replicas, without an actor rollout server. This is an execution smoke test.

After configuring the [runtime environment](#runtime-environment), run:

```bash
NUM_NPUS=8 python3 -m pytest -v -s tests/reward_loop/test_reward_engine_npu.py
```

All nonzero exits, including native abort 134, fail the smoke group.

### Wan2.2 trainer with a reward model

Test 2 runs one DanceGRPO training step with six actor NPUs and a dedicated
TP=2 Qwen2.5-VL reward engine on two NPUs. Prepare the Qwen2.5-VL-3B-Instruct
checkpoint before an offline run; set `REWARD_MODEL_PATH` to its local directory
when needed. The test must complete training and exit 0.

## Add a New NPU Smoke Test

### 1. Add the smallest useful entry point

Use a pytest file under the relevant tests module for core integration coverage,
such as tests/workers, tests/agent_loop, or tests/reward_loop.
Use a shell script under tests/special_e2e for a trainer end-to-end path.

Keep tests small and reproducible:

- Use tiny-random checkpoints or weights cached on the CI runner.
- Use the smallest practical dataset, batch size, and training-step count.
- Do not require public model or dataset downloads at runtime.
- Preserve nonzero exit codes on failure.
- Do not reuse Ray processes left by another test.

### 2. Register the test in its group runner

Add core tests to run_npu_smoke_core.sh and trainer tests to
run_npu_smoke_e2e.sh. Assign an unused numeric ID and call run_test:

```bash
run_test 8 "my NPU smoke test" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \
        NUM_NPUS="${NUM_NPUS}" \
    python3 -m pytest -s tests/workers/test_my_npu_feature.py
```

Do not reuse an existing ID. Adding a test to an existing group needs no
workflow mapping change.

### 3. Configure devices

Group runners use eight NPUs by default and derive
ASCEND_RT_VISIBLE_DEVICES from --num-npus. Each command should receive both
ASCEND_RT_VISIBLE_DEVICES and NUM_NPUS. If a test requires a fixed number of
devices, validate it in the test entry point and fail with a clear message.

### 4. Update workflow groups and paths

The Resolve NPU CI groups step maps labels to groups as listed in
[CI Triggers](#ci-triggers). Multiple labels produce the union of their groups.
Selected groups run in separate containers on independent eight-NPU runners,
in parallel when runner capacity is available. The setup_runner check succeeds
only when all selected groups pass.

If the new test lives outside current path filters, add its path. Current
filters include verl_omni/**, tests/npu_smoke/**, tests/agent_loop/**,
tests/reward_loop/**, tests/workers/**, tests/special_e2e/**, pyproject.toml,
.github/workflows/npu_smoke.yml,
and .github/vllm_omni_pin.txt.

### 5. Handle models and datasets

Prefer tiny checkpoints that can be generated and validated offline. Prepare
shared checkpoints in the workflow, model-specific core checkpoints in pytest
fixtures, and trainer checkpoints in their end-to-end launchers.
The Wan2.2 launcher prepares and validates its checkpoint
before training, so selecting only the core group does not prepare Wan2.2
weights.
Builders should reuse a valid cached checkpoint, rebuild missing or stale
components, and reload saved assets when file-presence checks are not
sufficient. Keep checkpoint paths overridable and do not overwrite shared
caches.

### 6. Validate locally

Run from the repository root in an environment with CANN, vLLM, vLLM-Ascend,
and vLLM-Omni installed.

Run all groups:

```bash
bash tests/npu_smoke/run_npu_smoke_tests.sh --num-npus 8
```

Run one group:

```bash
bash tests/npu_smoke/run_npu_smoke_core.sh --num-npus 8
bash tests/npu_smoke/run_npu_smoke_e2e.sh --num-npus 8
```

Select devices explicitly:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,2,4,6 NUM_NPUS=4 \
    bash tests/npu_smoke/run_npu_smoke_core.sh
```

Before committing:

```bash
bash -n tests/npu_smoke/*.sh
pre-commit run --all-files
```

bash -n checks syntax only; it does not start Ray or allocate NPUs. Pre-commit
may reformat files, so review changes and rerun it until all hooks pass.

## CI Triggers

Pushes to main or v0.* run both groups. Pull requests targeting those branches
run only after a supported label is applied:

| Label | Requested groups |
|---|---|
| ready-for-ci | core e2e |
| ci-npu | core e2e |
| ci-npu-core | core |
| ci-npu-e2e | e2e |

Labels are not mutually exclusive. Applying both ci-npu-core and ci-npu-e2e
runs both groups.

## Runtime Environment

Each group job uses eight NPUs on linux-aarch64-a2b4-8, a 120 minute timeout,
16 GiB shared memory, and this image:

```text
swr.cn-north-4.myhuaweicloud.com/mindspeed/pr-verl-omni-a2:v010-vllm030-omni030
```

The workflow prints CANN details, npu-smi output, relevant Ascend variables,
and installed vLLM component versions.

The FlowGRPO and DanceGRPO end-to-end launchers source CANN and ATB before
starting the workload:

```bash
source "${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}/set_env.sh"
source /usr/local/Ascend/nnal/atb/set_env.sh
```

Set `ASCEND_HOME_PATH` to the toolkit directory for a custom installation.
If it is unset or empty, the launchers use the `latest` symlink. The selected
CANN version must remain compatible with the image's PyTorch and torch-npu.

Both group runners source `lib_npu_smoke.sh`, which defaults `VLLM_PLUGINS`
to `ascend` and preloads jemalloc before starting the tests. If jemalloc is
not already in `LD_PRELOAD`, the library uses `JEMALLOC_PATH` or discovers a
readable library through `ldconfig`; a missing library fails the run. Set
`JEMALLOC_PATH` for a custom installation.

For direct pytest execution outside a group runner, export `VLLM_PLUGINS=ascend`
and preload the installed jemalloc library before launching Python.

The diagnostics and artifact-upload steps are defined directly in the workflow.
The NPU runner executes the job in a container, so repository-local composite
actions may otherwise be resolved against a different host workspace.

## Runtime Cleanup, Results, and Artifacts

Before each test, the shared library stops Ray, terminates stale
DiffusionWorker, VLLMWorker, and vLLMOmniHttpServer processes, waits five
seconds, force-kills remaining matches, and prints npu-smi information.

Per-test logs are stored under:

```text
logs/npu_smoke/<group>/<timestamp>/
```

Each test writes test_<id>.log and each group writes summary.log. The workflow
also writes group console logs under logs/npu_smoke/parallel/.

Each group's actions/upload-artifact step runs even after failure and uploads
logs/npu_smoke/** as npu-smoke-logs-core or npu-smoke-logs-e2e. A failed group
does not cancel the other selected group.

Every nonzero test exit, including native abort 134 after a pytest or training
completion message, remains a failure. Keep the raw exit code and full log
when reporting a failed run.
