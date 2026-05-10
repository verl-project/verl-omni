#!/usr/bin/env bash
# tests/npu_smoke/run_npu_smoke_tests.sh
#
# Offline Ascend NPU smoke-test suite for verl-omni.
#
# Usage:
#   bash tests/npu_smoke/run_npu_smoke_tests.sh [--num-npus N] [TEST_IDs...]
#
# Optional environment overrides:
#   LOG_DIR   Directory for per-test log files  (default: logs/npu_smoke/<timestamp>)
#   NUM_NPUS  Number of NPUs to run with        (default: 8)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
pass() { echo "[PASS] $*"; }
fail() { echo "[FAIL] $*"; }
warn() { echo "[WARN] $*"; }
sep()  { printf '%0.s-' {1..78}; echo; }

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/npu_smoke/${TIMESTAMP}}"
mkdir -p "${LOG_DIR}"
SUMMARY_LOG="${LOG_DIR}/summary.log"

export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export DEVICE_NAME=npu
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

REQUESTED_NUM_NPUS="${NUM_NPUS:-8}"
declare -a CLI_TEST_IDS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--num-npus)
            if [[ $# -lt 2 ]]; then
                fail "Missing value for $1"
                exit 2
            fi
            REQUESTED_NUM_NPUS="$2"
            shift 2
            ;;
        --num-npus=*)
            REQUESTED_NUM_NPUS="${1#*=}"
            shift
            ;;
        -h|--help)
            cat <<'EOF'
Usage:
  bash tests/npu_smoke/run_npu_smoke_tests.sh [--num-npus N] [TEST_IDs...]

Tests:
  0  vllm-omni rollout + sleep/wake_up
  1  FlowGRPO trainer e2e
EOF
            exit 0
            ;;
        *)
            CLI_TEST_IDS+=("$1")
            shift
            ;;
    esac
done

if ! [[ "${REQUESTED_NUM_NPUS}" =~ ^[0-9]+$ ]]; then
    fail "Invalid --num-npus value '${REQUESTED_NUM_NPUS}'"
    exit 2
fi

NUM_NPUS="${REQUESTED_NUM_NPUS}"
export NUM_NPUS

build_npu_device_list() {
    local n="$1"
    local devices=()
    local i
    for (( i=0; i<n; i++ )); do
        devices+=("${i}")
    done
    local IFS=,
    echo "${devices[*]}"
}

NPU_DEVICE_LIST="$(build_npu_device_list "${NUM_NPUS}")"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-${NPU_DEVICE_LIST}}"

declare -a TEST_NAMES=()
declare -a TEST_RESULTS=()
declare -a TEST_DURATIONS=()
declare -a TEST_LOG_FILES=()

run_test() {
    local id="$1"; local name="$2"; shift 2
    local logfile="${LOG_DIR}/test_${id}.log"

    sep
    log "Starting  [${id}] ${name}"
    log "Command : $*"
    log "Log file: ${logfile}"
    sep

    local start_ts; start_ts="$(date +%s)"
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
}

skip_test() {
    local id="$1"; local name="$2"; local reason="$3"
    warn "Skipping  [${id}] ${name}  - ${reason}"
    TEST_NAMES+=("${name}")
    TEST_RESULTS+=("SKIP")
    TEST_DURATIONS+=("-")
    TEST_LOG_FILES+=("-")
}

run_selected_test() {
    local id="$1"; local name="$2"; shift 2
    if [[ "${RUN_TEST[$id]}" == "1" ]]; then
        run_test "${id}" "${name}" "$@"
    else
        skip_test "${id}" "${name}" "not selected"
    fi
}

declare -A RUN_TEST=([0]=1 [1]=1)

if [[ "${#CLI_TEST_IDS[@]}" -gt 0 ]]; then
    for k in "${!RUN_TEST[@]}"; do RUN_TEST[$k]=0; done
    for id in "${CLI_TEST_IDS[@]}"; do
        if [[ -n "${RUN_TEST[$id]+x}" ]]; then
            RUN_TEST[$id]=1
        else
            warn "Unknown test id '${id}' - ignored"
        fi
    done
fi

sep
echo "  verl-omni NPU Smoke Test Suite"
echo -e "  Date      : $(date '+%Y-%m-%d %H:%M:%S')"
echo -e "  Repo root : ${REPO_ROOT}"
echo -e "  Log dir   : ${LOG_DIR}"
echo -e "  NUM_NPUS  : ${NUM_NPUS}"
echo -e "  ASCEND_RT_VISIBLE_DEVICES : ${ASCEND_RT_VISIBLE_DEVICES}"
sep
echo ""

run_selected_test 0 "vllm-omni rollout + sleep/wake_up" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    pytest -s tests/workers/rollout/rollout_vllm/test_vllm_omni_generate_npu.py

run_selected_test 1 "FlowGRPO trainer e2e" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    bash tests/special_e2e/run_flowgrpo_qwen_image_npu.sh

sep | tee "${SUMMARY_LOG}"
echo "  SMOKE TEST SUMMARY" | tee -a "${SUMMARY_LOG}"
sep | tee -a "${SUMMARY_LOG}"

passed=0; failed=0; skipped=0
for i in "${!TEST_NAMES[@]}"; do
    result="${TEST_RESULTS[$i]}"
    case "${result}" in
        PASS) (( ++passed  )) ;;
        FAIL) (( ++failed  )) ;;
        SKIP) (( ++skipped )) ;;
    esac
    printf "%-4s  %-7s  %-8s  %s\n" \
        "${i}" "${result}" "${TEST_DURATIONS[$i]}" "${TEST_NAMES[$i]}" | tee -a "${SUMMARY_LOG}"
done

sep | tee -a "${SUMMARY_LOG}"
echo "Passed : ${passed}" | tee -a "${SUMMARY_LOG}"
echo "Failed : ${failed}" | tee -a "${SUMMARY_LOG}"
echo "Skipped: ${skipped}" | tee -a "${SUMMARY_LOG}"

if [[ "${failed}" -gt 0 ]]; then
    exit 1
fi