#!/usr/bin/env bash
# Shared helpers for verl-omni GPU smoke test groups.

set -euo pipefail

GPU_SMOKE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${GPU_SMOKE_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
pass() { echo "[PASS] $*"; }
fail() { echo "[FAIL] $*"; }
sep()  { printf '%0.s-' {1..78}; echo; }

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

gpu_smoke_init() {
    GPU_SMOKE_GROUP="$1"
    local default_num_gpus="$2"
    shift 2

    REQUESTED_NUM_GPUS="${NUM_GPUS:-${default_num_gpus}}"
    REQUESTED_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -g|--num-gpus)
                if [[ $# -lt 2 ]]; then
                    fail "Missing value for $1 (expected a positive integer)"
                    exit 2
                fi
                REQUESTED_NUM_GPUS="$2"
                shift 2
                ;;
            --num-gpus=*)
                REQUESTED_NUM_GPUS="${1#*=}"
                shift
                ;;
            --cuda-visible-devices)
                if [[ $# -lt 2 ]]; then
                    fail "Missing value for $1 (example: 0,1)"
                    exit 2
                fi
                REQUESTED_CUDA_VISIBLE_DEVICES="$2"
                shift 2
                ;;
            --cuda-visible-devices=*)
                REQUESTED_CUDA_VISIBLE_DEVICES="${1#*=}"
                shift
                ;;
            -h|--help)
                cat <<EOF
Usage:
  bash ${BASH_SOURCE[1]:-$0} [--num-gpus N] [--cuda-visible-devices DEVICES]

Options:
  -g, --num-gpus N              GPU count to run with (positive integer)
      --cuda-visible-devices    Comma-separated GPU IDs to expose
  -h, --help                    Show this help message
EOF
                exit 0
                ;;
            *)
                fail "Unknown argument '$1'"
                exit 2
                ;;
        esac
    done

    if ! [[ "${REQUESTED_NUM_GPUS}" =~ ^[0-9]+$ ]]; then
        fail "Invalid --num-gpus value '${REQUESTED_NUM_GPUS}' (must be a positive integer)"
        exit 2
    fi
    if [[ "${REQUESTED_NUM_GPUS}" -lt 1 ]]; then
        fail "Invalid --num-gpus value '${REQUESTED_NUM_GPUS}' (must be a positive integer)"
        exit 2
    fi

    export NUM_GPUS="${REQUESTED_NUM_GPUS}"
    if [[ -n "${REQUESTED_CUDA_VISIBLE_DEVICES}" ]]; then
        CUDA_DEVICE_LIST="${REQUESTED_CUDA_VISIBLE_DEVICES}"
    elif [[ "${NUM_GPUS}" -gt 0 ]]; then
        CUDA_DEVICE_LIST="$(build_cuda_device_list "${NUM_GPUS}")"
    else
        CUDA_DEVICE_LIST=""
    fi
    export CUDA_DEVICE_LIST

    TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
    LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/gpu_smoke/${GPU_SMOKE_GROUP}/${TIMESTAMP}}"
    mkdir -p "${LOG_DIR}"
    SUMMARY_LOG="${LOG_DIR}/summary.log"

    export PYTHONUNBUFFERED=1
    export RAY_DEDUP_LOGS=0
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        export LD_LIBRARY_PATH="${CONDA_PREFIX}/cuda-compat${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi

    TEST_IDS=()
    TEST_NAMES=()
    TEST_RESULTS=()
    TEST_DURATIONS=()
    TEST_LOG_FILES=()

    sep
    echo "  verl-omni GPU Smoke Test Group"
    echo -e "  Date      : $(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "  Repo root : ${REPO_ROOT}"
    echo -e "  Log dir   : ${LOG_DIR}"
    echo -e "  Group     : ${GPU_SMOKE_GROUP}"
    echo -e "  NUM_GPUS  : ${NUM_GPUS}"
    if [[ -n "${CUDA_DEVICE_LIST}" ]]; then
        echo -e "  CUDA_VISIBLE_DEVICES : ${CUDA_DEVICE_LIST}"
    fi
    sep
    echo ""
}

run_test() {
    local id="$1"; local name="$2"; shift 2
    local logfile="${LOG_DIR}/test_${id}.log"

    if [[ "${GPU_SMOKE_SKIP_RAY_STOP:-0}" != "1" ]]; then
        ray stop --force 2>/dev/null || true
    fi

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

    echo ""
}

gpu_smoke_summary() {
    sep
    echo "  SMOKE TEST SUMMARY"
    sep

    local passed=0 failed=0 skipped=0
    {
        echo "Test Results  -  $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Repo: ${REPO_ROOT}"
        echo "Group: ${GPU_SMOKE_GROUP}"
        echo ""
        printf "%-4s  %-7s  %-8s  %s\n" "ID" "RESULT" "ELAPSED" "NAME"
        printf "%-4s  %-7s  %-8s  %s\n" "----" "-------" "--------" "----"
    } | tee "${SUMMARY_LOG}"

    local i result name elapsed logfile id
    for i in "${!TEST_NAMES[@]}"; do
        id="${TEST_IDS[$i]}"
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
            "${id}" "${result}" "${elapsed}" "${name}" | tee -a "${SUMMARY_LOG}"

        if [[ "${result}" == "FAIL" && "${logfile}" != "-" ]]; then
            echo "            log: ${logfile}" | tee -a "${SUMMARY_LOG}"
        fi
    done

    sep | tee -a "${SUMMARY_LOG}"

    local total=$(( passed + failed + skipped ))
    echo "  Total: ${total}  |  Passed: ${passed}  |  Failed: ${failed}  |  Skipped: ${skipped}" \
        | tee -a "${SUMMARY_LOG}"
    echo "  Full logs: ${LOG_DIR}" | tee -a "${SUMMARY_LOG}"
    sep | tee -a "${SUMMARY_LOG}"

    if [[ "${failed}" -gt 0 ]]; then
        exit 1
    fi
}
