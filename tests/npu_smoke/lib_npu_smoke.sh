#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
pass() { echo "[PASS] $*"; }
fail() { echo "[FAIL] $*"; }
warn() { echo "[WARN] $*"; }
sep() { printf '%0.s-' {1..78}; echo; }

NPU_SMOKE_GROUP="${NPU_SMOKE_GROUP:?NPU_SMOKE_GROUP must be set}"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/npu_smoke/${NPU_SMOKE_GROUP}/${TIMESTAMP}}"
mkdir -p "${LOG_DIR}"
SUMMARY_LOG="${LOG_DIR}/summary.log"

export PYTHONUNBUFFERED=1
export RAY_DEDUP_LOGS=0
export DEVICE_NAME=npu
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VLLM_PLUGINS="${VLLM_PLUGINS:-ascend}"

if [[ "${LD_PRELOAD:-}" != *libjemalloc* ]]; then
    if [[ -z "${JEMALLOC_PATH:-}" ]] && ! JEMALLOC_PATH=$(ldconfig -p 2>/dev/null \
        | awk '/libjemalloc\.so/ && !path {path = $NF} END {print path}'); then
        JEMALLOC_PATH=
    fi
    if [[ -z "${JEMALLOC_PATH:-}" || ! -r "${JEMALLOC_PATH}" ]]; then
        fail "Unable to locate a readable jemalloc library for NPU smoke tests."
        exit 1
    fi
    export LD_PRELOAD="${JEMALLOC_PATH}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

log "NPU smoke tests will use jemalloc via LD_PRELOAD=${LD_PRELOAD}"

REQUESTED_NUM_NPUS="${NUM_NPUS:-8}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--num-npus)
            [[ $# -ge 2 ]] || { fail "Missing value for $1"; exit 2; }
            REQUESTED_NUM_NPUS="$2"
            shift 2
            ;;
        --num-npus=*)
            REQUESTED_NUM_NPUS="${1#*=}"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--num-npus N]"
            exit 0
            ;;
        *)
            fail "Unknown argument: $1"
            exit 2
            ;;
    esac
done

[[ "${REQUESTED_NUM_NPUS}" =~ ^[1-9][0-9]*$ ]] || {
    fail "Invalid --num-npus value '${REQUESTED_NUM_NPUS}'"
    exit 2
}
NUM_NPUS="${REQUESTED_NUM_NPUS}"
export NUM_NPUS

build_npu_device_list() {
    local devices=()
    local i
    for ((i = 0; i < $1; i++)); do
        devices+=("${i}")
    done
    local IFS=,
    echo "${devices[*]}"
}
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-$(build_npu_device_list "${NUM_NPUS}")}"

declare -a TEST_IDS=()
declare -a TEST_NAMES=()
declare -a TEST_RESULTS=()
declare -a TEST_DURATIONS=()
declare -a TEST_LOG_FILES=()

cleanup_runtime() {
    ray stop --force || true
    pkill -TERM -f DiffusionWorker || true
    pkill -TERM -f VLLMWorker || true
    pkill -TERM -f vLLMOmniHttpServer || true
    sleep 5
    pkill -KILL -f DiffusionWorker || true
    pkill -KILL -f VLLMWorker || true
    pkill -KILL -f vLLMOmniHttpServer || true
    npu-smi info || true
}

run_test() {
    local id="$1"
    local name="$2"
    shift 2
    local logfile="${LOG_DIR}/test_${id}.log"

    cleanup_runtime
    sep
    log "Starting  [${id}] ${name}"
    log "Command : $*"
    log "Log file: ${logfile}"
    sep

    local start_ts end_ts elapsed rc
    start_ts="$(date +%s)"
    set +e
    "$@" 2>&1 | tee "${logfile}"
    rc="${PIPESTATUS[0]}"
    set -e

    end_ts="$(date +%s)"
    elapsed=$((end_ts - start_ts))
    TEST_IDS+=("${id}")
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

    return 0
}

print_summary() {
    sep | tee "${SUMMARY_LOG}"
    echo "  ${NPU_SMOKE_GROUP^^} NPU SMOKE TEST SUMMARY" | tee -a "${SUMMARY_LOG}"
    sep | tee -a "${SUMMARY_LOG}"

    local passed=0 failed=0 i result
    for i in "${!TEST_NAMES[@]}"; do
        result="${TEST_RESULTS[$i]}"
        case "${result}" in
            PASS) ((++passed)) ;;
            FAIL) ((++failed)) ;;
        esac
        printf "%-4s  %-7s  %-8s  %s\n"             "${TEST_IDS[$i]}" "${result}" "${TEST_DURATIONS[$i]}" "${TEST_NAMES[$i]}"             | tee -a "${SUMMARY_LOG}"
    done

    sep | tee -a "${SUMMARY_LOG}"
    echo "Passed: ${passed}" | tee -a "${SUMMARY_LOG}"
    echo "Failed: ${failed}" | tee -a "${SUMMARY_LOG}"
    echo "Logs  : ${LOG_DIR}" | tee -a "${SUMMARY_LOG}"
    [[ "${failed}" -eq 0 ]]
}

trap cleanup_runtime EXIT

print_header() {
    sep
    echo "  verl-omni NPU Smoke Test: ${NPU_SMOKE_GROUP}"
    echo "  Date      : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Repo root : ${REPO_ROOT}"
    echo "  Log dir   : ${LOG_DIR}"
    echo "  NUM_NPUS  : ${NUM_NPUS}"
    echo "  ASCEND_RT_VISIBLE_DEVICES : ${ASCEND_RT_VISIBLE_DEVICES}"
    sep
}
