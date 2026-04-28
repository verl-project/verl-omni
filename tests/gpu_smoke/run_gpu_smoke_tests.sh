#!/usr/bin/env bash
# tests/gpu_smoke/run_gpu_smoke_tests.sh (offline version, to be dropped when online GPU test is available)
#
# Offline GPU smoke-test suite for verl-omni.
# Runs a curated set of GPU-dependent tests and produces a structured
# pass/fail summary with per-test log capture.
#
# Usage:
#   bash tests/gpu_smoke/run_gpu_smoke_tests.sh [TEST_IDs...]
#
#   With no arguments, runs all enabled tests.
#   Pass specific test IDs to run only those:
#     bash tests/gpu_smoke/run_gpu_smoke_tests.sh 0 3 6
#
# Optional environment overrides:
#   LOG_DIR   Directory for per-test log files  (default: logs/gpu_smoke/<timestamp>)
#   NUM_GPUS  Number of GPUs available          (default: auto-detected)

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

# ── GPU detection ──────────────────────────────────────────────────────────────
if [[ -z "${NUM_GPUS:-}" ]]; then
    if command -v nvidia-smi &>/dev/null; then
        NUM_GPUS="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
    else
        NUM_GPUS=0
    fi
fi
export NUM_GPUS

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

# ── Determine which tests to run ───────────────────────────────────────────────
declare -A RUN_TEST=(
    [0]=1 [1]=1 [2]=1 [3]=1 [4]=1
)

# If explicit IDs were passed on the CLI, override to run only those.
if [[ $# -gt 0 ]]; then
    for k in "${!RUN_TEST[@]}"; do RUN_TEST[$k]=0; done
    for id in "$@"; do
        if [[ -v RUN_TEST[$id] ]]; then
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
sep
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# TEST DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── Test 0: vllm-omni rollout ─────────────────────────────────────────────────
if [[ "${RUN_TEST[0]}" == "1" ]]; then
    if [[ "${NUM_GPUS}" -lt 1 ]]; then
        skip_test 0 "vllm-omni rollout" "requires at least 1 GPU"
    else
        run_test 0 "vllm-omni rollout" \
            pytest -s tests/workers/rollout/rollout_vllm/test_vllm_omni_generate.py
    fi
else
    skip_test 0 "vllm-omni rollout" "not selected"
fi

# ── Test 1: diffusion agent loop ──────────────────────────────────────────────
if [[ "${RUN_TEST[1]}" == "1" ]]; then
    run_test 1 "diffusion agent loop" \
        pytest -s tests/agent_loop/test_diffusion_agent_loop.py
else
    skip_test 1 "diffusion agent loop" "not selected"
fi

# ── Test 2: visual reward manager ─────────────────────────────────────────────
if [[ "${RUN_TEST[2]}" == "1" ]]; then
    run_test 2 "visual reward manager" \
        pytest -s tests/reward_loop/test_visual_reward_manager.py
else
    skip_test 2 "visual reward manager" "not selected"
fi

# ── Test 3: diffusers FSDP engine (4 GPUs) ────────────────────────────────────
if [[ "${RUN_TEST[3]}" == "1" ]]; then
    if [[ "${NUM_GPUS}" -lt 4 ]]; then
        skip_test 3 "diffusers FSDP engine" "requires 4 GPUs, only ${NUM_GPUS} available"
    else
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        run_test 3 "diffusers FSDP engine" \
            pytest -s tests/workers/test_diffusers_fsdp_engine.py
    fi
else
    skip_test 3 "diffusers FSDP engine" "not selected"
fi

# ── Test 4: FlowGRPO trainer e2e (vllm_omni rollout) ─────────────────────────
if [[ "${RUN_TEST[4]}" == "1" ]]; then
    if [[ "${NUM_GPUS}" -lt 4 ]]; then
        skip_test 4 "FlowGRPO trainer e2e" "requires 4 GPUs, only ${NUM_GPUS} available"
    else
        run_test 4 "FlowGRPO trainer e2e" \
            bash tests/special_e2e/run_flowgrpo_trainer_diffusers.sh
    fi
else
    skip_test 4 "FlowGRPO trainer e2e" "not selected"
fi

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
