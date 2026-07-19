#!/usr/bin/env bash
# tests/gpu_smoke/run_gpu_smoke_tests.sh
#
# Offline GPU smoke-test suite for verl-omni.
# Runs a curated set of GPU-dependent tests and produces a structured
# pass/fail summary with per-test log capture.
#
# Usage:
#   bash tests/gpu_smoke/run_gpu_smoke_tests.sh [--num-gpus N] [TEST_IDs...]
#
#   With no arguments, runs all enabled tests.
#   Pass specific test IDs to run only those:
#     bash tests/gpu_smoke/run_gpu_smoke_tests.sh 0 3 4
#   Select GPU count (allowed: 1, 2, 4, 8):
#     bash tests/gpu_smoke/run_gpu_smoke_tests.sh --num-gpus 2
#
# Optional environment overrides:
#   LOG_DIR   Directory for per-test log files  (default: logs/gpu_smoke/<timestamp>)
#   NUM_GPUS  Number of GPUs to run with        (default: 4)

set -euo pipefail

# ── Repo root ──────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# ── Logging helpers ──────────────────────────────────────────────────────────
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
pass() { echo "[PASS] $*"; }
fail() { echo "[FAIL] $*"; }
warn() { echo "[WARN] $*"; }
sep()  { printf '%0.s-' {1..78}; echo; }

# ── Timestamp / log directory ──────────────────────────────────────────────────
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/gpu_smoke/${TIMESTAMP}}"
mkdir -p "${LOG_DIR}"
SUMMARY_LOG="${LOG_DIR}/summary.log"

# ── Shared environment setup ───────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
# Ensure CUDA compat libs are visible when running inside a conda env
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/cuda-compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# ── GPU selection ──────────────────────────────────────────────────────────────
REQUESTED_NUM_GPUS="${NUM_GPUS:-4}"
declare -a CLI_TEST_IDS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--num-gpus)
            if [[ $# -lt 2 ]]; then
                fail "Missing value for $1 (expected one of: 1, 2, 4, 8)"
                exit 2
            fi
            REQUESTED_NUM_GPUS="$2"
            shift 2
            ;;
        --num-gpus=*)
            REQUESTED_NUM_GPUS="${1#*=}"
            shift
            ;;
        -h|--help)
            cat <<'EOF'
Usage:
  bash tests/gpu_smoke/run_gpu_smoke_tests.sh [--num-gpus N] [TEST_IDs...]

Options:
  -g, --num-gpus N    GPU count to run with (allowed: 1, 2, 4, 8)
  -h, --help          Show this help message
EOF
            exit 0
            ;;
        *)
            CLI_TEST_IDS+=("$1")
            shift
            ;;
    esac
done

if [[ -n "${REQUESTED_NUM_GPUS}" ]]; then
    if ! [[ "${REQUESTED_NUM_GPUS}" =~ ^[0-9]+$ ]]; then
        fail "Invalid --num-gpus value '${REQUESTED_NUM_GPUS}' (must be 1, 2, 4, or 8)"
        exit 2
    fi
    case "${REQUESTED_NUM_GPUS}" in
        1|2|4|8) ;;
        *)
            fail "Unsupported --num-gpus value '${REQUESTED_NUM_GPUS}' (must be 1, 2, 4, or 8)"
            exit 2
            ;;
    esac

    NUM_GPUS="${REQUESTED_NUM_GPUS}"
fi

export NUM_GPUS

build_cuda_device_list() {
    local n="$1"
    local devices=()
    local i
    for (( i=0; i<n; i++ )); do
        devices+=("${i}")
    done
    local IFS=,
    echo "${devices[*]}"
}

if [[ "${NUM_GPUS}" -gt 0 ]]; then
    CUDA_DEVICE_LIST="$(build_cuda_device_list "${NUM_GPUS}")"
else
    CUDA_DEVICE_LIST=""
fi

# ── Internal result tracking ───────────────────────────────────────────────────
declare -a TEST_NAMES=()
declare -a TEST_RESULTS=()   # "PASS" | "FAIL" | "SKIP"
declare -a TEST_DURATIONS=()
declare -a TEST_LOG_FILES=()

# ── run_test <id> <name> <cmd...> ─────────────────────────────────────────────
# Runs a command, tees output to a per-test log, and records the outcome.
run_test() {
    local id="$1"; local name="$2"; shift 2
    local logfile="${LOG_DIR}/test_${id}.log"

    ray stop --force 2>/dev/null || true

    sep
    log "Starting  [${id}] ${name}"
    log "Command : $*"
    log "Log file: ${logfile}"
    sep

    local start_ts; start_ts="$(date +%s)"

    # Run command; tee stdout+stderr to log file and also to the terminal.
    set +e
    "$@" 2>&1 | tee "${logfile}"
    local rc="${PIPESTATUS[0]}"
    set -e

    local end_ts; end_ts="$(date +%s)"
    local elapsed=$(( end_ts - start_ts ))

    TEST_NAMES+=("${name}")
    TEST_DURATIONS+=("${elapsed}s")
    TEST_LOG_FILES+=("${logfile}")

    if [[ "${rc}" -eq 0 ]]; then
        TEST_RESULTS+=("PASS")
        pass "[${id}] ${name}  (${elapsed}s)"
    else
        TEST_RESULTS+=("FAIL")
        fail "[${id}] ${name}  (${elapsed}s)  exit=${rc}"
    fi

    echo ""
}

# ── skip_test <id> <name> <reason> ────────────────────────────────────────────
skip_test() {
    local id="$1"; local name="$2"; local reason="$3"
    warn "Skipping  [${id}] ${name}  — ${reason}"
    TEST_NAMES+=("${name}")
    TEST_RESULTS+=("SKIP")
    TEST_DURATIONS+=("-")
    TEST_LOG_FILES+=("-")
}

# ── run_selected_test <id> <name> <cmd...> ────────────────────────────────────
run_selected_test() {
    local id="$1"; local name="$2"; shift 2
    if [[ "${RUN_TEST[$id]}" == "1" ]]; then
        run_test "${id}" "${name}" "$@"
    else
        skip_test "${id}" "${name}" "not selected"
    fi
}

# ── Determine which tests to run ───────────────────────────────────────────────
declare -A RUN_TEST=(
    [0]=1 [1]=1 [2]=1 [3]=1 [4]=1 [5]=1 [6]=1 [7]=1 [8]=1 [9]=1
)

# If explicit IDs were passed on the CLI, override to run only those.
if [[ "${#CLI_TEST_IDS[@]}" -gt 0 ]]; then
    for k in "${!RUN_TEST[@]}"; do RUN_TEST[$k]=0; done
    for id in "${CLI_TEST_IDS[@]}"; do
        if [[ -n "${RUN_TEST[$id]+x}" ]]; then
            RUN_TEST[$id]=1
        else
            warn "Unknown test id '${id}' — ignored"
        fi
    done
fi

# ── Print header ───────────────────────────────────────────────────────────────
sep
echo "  verl-omni GPU Smoke Test Suite"
echo -e "  Date      : $(date '+%Y-%m-%d %H:%M:%S')"
echo -e "  Repo root : ${REPO_ROOT}"
echo -e "  Log dir   : ${LOG_DIR}"
echo -e "  NUM_GPUS  : ${NUM_GPUS}"
if [[ -n "${CUDA_DEVICE_LIST}" ]]; then
    echo -e "  CUDA_VISIBLE_DEVICES : ${CUDA_DEVICE_LIST}"
fi
sep
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# TEST DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── Test 0: vllm-omni rollout ─────────────────────────────────────────────────
run_selected_test 0 "vllm-omni rollout" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" \
    pytest -s tests/workers/rollout/rollout_vllm/test_vllm_omni_generate.py

# ── Test 1: diffusion agent loop ──────────────────────────────────────────────
run_selected_test 1 "diffusion agent loop" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" \
    pytest -s tests/agent_loop/test_diffusion_agent_loop.py \
        tests/agent_loop/test_diffusion_rollout_seed_gpu.py

# ── Test 2: visual reward manager ─────────────────────────────────────────────
run_selected_test 2 "visual reward manager" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" \
    pytest -s tests/reward_loop/test_visual_reward_manager.py

# ── Test 3: diffusers FSDP engine ─────────────────────────────────────────────
run_selected_test 3 "diffusers FSDP engine" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" \
    pytest -s tests/workers/test_diffusers_fsdp_engine.py

# ── Test 4: FlowGRPO trainer e2e (vllm_omni rollout) ─────────────────────────
run_selected_test 4 "FlowGRPO trainer e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" \
    bash tests/special_e2e/run_flowgrpo_qwen_image.sh

# ── Test 5: DiffusionNFT trainer e2e (vllm_omni rollout) ─────────────────────
run_selected_test 5 "DiffusionNFT trainer e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" \
    bash tests/special_e2e/run_diffusionnft_qwen_image.sh

# ── Test 6: Qwen-Image online DPO trainer e2e (vllm_omni rollout) ─────────────
run_selected_test 6 "Qwen-Image online DPO trainer e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" \
    bash tests/special_e2e/run_online_dpo_qwen_image.sh

# ── Test 7: diffusers VeOmni engine ───────────────────────────────────────────
# Skips itself if the optional `veomni` backend is not installed (importorskip).
run_selected_test 7 "diffusers VeOmni engine" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" \
    pytest -s tests/workers/test_diffusers_veomni_engine.py

# ── Test 8: Qwen3-Omni Thinker GSPO LoRA e2e (vllm_omni AR rollout) ────────────
# Fixed at 2 GPUs: the smoke stage config pins tensor_parallel_size=2 and FSDP
# needs >1 GPU to shard (NO_SHARD can't run the offload_to_cpu LoRA-sync summon).
run_selected_test 8 "Qwen3-Omni Thinker GSPO LoRA e2e" \
    env CUDA_VISIBLE_DEVICES="0,1" NUM_GPUS=2 \
    bash tests/special_e2e/run_gspo_qwen3_omni_thinker_lora_smoke.sh

# ── Test 9: LTX-2.3 FlowGRPO trainer e2e (vllm_omni rollout) ──────────────
run_selected_test 9 "LTX-2.3 FlowGRPO trainer e2e" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" NUM_GPUS="${NUM_GPUS}" \
    bash tests/special_e2e/run_flowgrpo_ltx2_3.sh

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

sep
echo "  SMOKE TEST SUMMARY"
sep

passed=0; failed=0; skipped=0
{
    echo "Test Results  —  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Repo: ${REPO_ROOT}"
    echo ""
    printf "%-4s  %-7s  %-8s  %s\n" "ID" "RESULT" "ELAPSED" "NAME"
    printf "%-4s  %-7s  %-8s  %s\n" "----" "-------" "--------" "----"
} | tee "${SUMMARY_LOG}"

for i in "${!TEST_NAMES[@]}"; do
    result="${TEST_RESULTS[$i]}"
    name="${TEST_NAMES[$i]}"
    elapsed="${TEST_DURATIONS[$i]}"
    logfile="${TEST_LOG_FILES[$i]}"

    case "${result}" in
        PASS) (( ++passed  )) ;;
        FAIL) (( ++failed  )) ;;
        SKIP) (( ++skipped )) ;;
    esac

    printf "%-4s  %-7s  %-8s  %s\n" \
        "${i}" "${result}" "${elapsed}" "${name}" | tee -a "${SUMMARY_LOG}"

    if [[ "${result}" == "FAIL" && "${logfile}" != "-" ]]; then
        echo "            └─ log: ${logfile}" | tee -a "${SUMMARY_LOG}"
    fi
done

sep | tee -a "${SUMMARY_LOG}"

total=$(( passed + failed + skipped ))
echo "  Total: ${total}  |  Passed: ${passed}  |  Failed: ${failed}  |  Skipped: ${skipped}" \
    | tee -a "${SUMMARY_LOG}"
echo "  Full logs: ${LOG_DIR}" | tee -a "${SUMMARY_LOG}"
sep | tee -a "${SUMMARY_LOG}"

# Exit non-zero if any test failed
if [[ "${failed}" -gt 0 ]]; then
    exit 1
fi
