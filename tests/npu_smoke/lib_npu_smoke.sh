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
if [[ -z "${LOG_DIR:-}" ]]; then
    NPU_SMOKE_LOG_ROOT="${REPO_ROOT}/logs/npu_smoke/${NPU_SMOKE_GROUP}"
    mkdir -p "${NPU_SMOKE_LOG_ROOT}"
    LOG_DIR="$(mktemp -d "${NPU_SMOKE_LOG_ROOT}/${TIMESTAMP}.XXXXXX")"
fi
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
NPU_SMOKE_FULL_TOPOLOGY="${NPU_SMOKE_FULL_TOPOLOGY:-0}"
[[ "${NPU_SMOKE_FULL_TOPOLOGY}" == 0 || "${NPU_SMOKE_FULL_TOPOLOGY}" == 1 ]] || {
    fail "NPU_SMOKE_FULL_TOPOLOGY must be 0 or 1."
    exit 2
}
NPU_SMOKE_PARALLEL="${NPU_SMOKE_PARALLEL:-0}"
NPU_SMOKE_PARALLEL_CHILD="${NPU_SMOKE_PARALLEL_CHILD:-0}"
for npu_smoke_flag in NPU_SMOKE_PARALLEL NPU_SMOKE_PARALLEL_CHILD; do
    [[ "${!npu_smoke_flag}" == 0 || "${!npu_smoke_flag}" == 1 ]] || {
        fail "${npu_smoke_flag} must be 0 or 1."
        exit 2
    }
done
NPU_SMOKE_TEST_IDS="${NPU_SMOKE_TEST_IDS:-}"
if [[ -n "${NPU_SMOKE_TEST_IDS}" ]]; then
    [[ "${NPU_SMOKE_TEST_IDS}" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]] || {
        fail "NPU_SMOKE_TEST_IDS must be comma-separated test IDs."
        exit 2
    }
    IFS=, read -r -a NPU_SMOKE_REQUESTED_IDS <<< "${NPU_SMOKE_TEST_IDS}"
    npu_smoke_seen_ids=,
    for npu_smoke_id in "${NPU_SMOKE_REQUESTED_IDS[@]}"; do
        if [[ "${npu_smoke_seen_ids}" == *",${npu_smoke_id},"* ]]; then
            fail "NPU_SMOKE_TEST_IDS must not repeat test IDs."
            exit 2
        fi
        npu_smoke_seen_ids+="${npu_smoke_id},"
    done
elif [[ "${NPU_SMOKE_PARALLEL_CHILD}" == 1 ]]; then
    fail "Parallel smoke children require a nonempty NPU_SMOKE_TEST_IDS selection."
    exit 2
fi

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
export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-${ASCEND_RT_VISIBLE_DEVICES}}"

validate_npu_devices() {
    local name="$1" value="$2" device seen=,
    local -a devices
    [[ "${value}" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
        fail "${name} must be a comma-separated list of NPU IDs."
        return 2
    }
    IFS=, read -r -a devices <<< "${value}"
    [[ "${#devices[@]}" -eq "${NUM_NPUS}" ]] || {
        fail "${name} must list exactly ${NUM_NPUS} NPU IDs."
        return 2
    }
    for device in "${devices[@]}"; do
        device="${device#"${device%%[!0]*}"}"
        device="${device:-0}"
        if [[ "${seen}" == *",${device},"* ]]; then
            fail "${name} must not repeat NPU IDs."
            return 2
        fi
        seen+="${device},"
    done
}
for npu_smoke_variable in ASCEND_RT_VISIBLE_DEVICES ASCEND_VISIBLE_DEVICES; do
    validate_npu_devices "${npu_smoke_variable}" "${!npu_smoke_variable}" || exit 2
done
IFS=, read -r -a NPU_SMOKE_DEVICES <<< "${ASCEND_RT_VISIBLE_DEVICES}"
IFS=, read -r -a NPU_SMOKE_VISIBLE_DEVICES <<< "${ASCEND_VISIBLE_DEVICES}"

declare -a TEST_IDS=()
declare -a TEST_NAMES=()
declare -a TEST_RESULTS=()
declare -a TEST_DURATIONS=()
declare -a TEST_LOG_FILES=()
NPU_SMOKE_REGISTERED_IDS=,

cleanup_runtime() {
    if [[ "${NPU_SMOKE_PARALLEL_CHILD}" == 1 ]]; then
        python3 tests/npu_smoke/run_npu_smoke_processes.py cleanup
        return $?
    fi
    ray stop --force || true
    pkill -TERM -f DiffusionWorker || true
    pkill -TERM -f VLLMWorker || true
    pkill -TERM -f 'VLLM::' || true
    pkill -TERM -f vLLMOmniHttpServer || true
    sleep 5
    pkill -KILL -f DiffusionWorker || true
    pkill -KILL -f VLLMWorker || true
    pkill -KILL -f 'VLLM::' || true
    pkill -KILL -f vLLMOmniHttpServer || true
    npu-smi info || true
}

run_test() {
    local id="$1"
    local name="$2"
    shift 2
    if [[ "${NPU_SMOKE_REGISTERED_IDS}" == *",${id},"* ]]; then
        fail "Test ${id} is registered more than once."
        return 2
    fi
    NPU_SMOKE_REGISTERED_IDS+="${id},"
    if [[ -n "${NPU_SMOKE_TEST_IDS}" && ",${NPU_SMOKE_TEST_IDS}," != *",${id},"* ]]; then
        return 0
    fi
    local test_num_npus="${NUM_NPUS}" max_npus test_devices test_visible_devices
    if [[ "${1:-}" == --max-npus ]]; then
        [[ $# -ge 2 && "${2:-}" =~ ^[1-9][0-9]*$ ]] || {
            fail "Test ${id}: --max-npus requires a positive integer."
            return 2
        }
        max_npus="$2"
        shift 2
        if [[ "${NPU_SMOKE_FULL_TOPOLOGY}" == 0 ]] && (( max_npus < test_num_npus )); then
            test_num_npus="${max_npus}"
        fi
    fi
    [[ $# -gt 0 ]] || { fail "Test ${id}: missing command."; return 2; }
    test_devices="$(IFS=,; echo "${NPU_SMOKE_DEVICES[*]:0:test_num_npus}")"
    test_visible_devices="$(IFS=,; echo "${NPU_SMOKE_VISIBLE_DEVICES[*]:0:test_num_npus}")"
    local logfile="${LOG_DIR}/test_${id}.log"

    cleanup_runtime
    sep
    log "Starting  [${id}] ${name}"
    log "NPU budget: ${test_num_npus}; ASCEND_RT_VISIBLE_DEVICES=${test_devices}; ASCEND_VISIBLE_DEVICES=${test_visible_devices}"
    log "Command : $*"
    log "Log file: ${logfile}"
    sep

    local start_ts end_ts elapsed rc tee_rc
    local -a pipeline_status
    start_ts="$(date +%s)"
    set +e
    env NUM_NPUS="${test_num_npus}" \
        ASCEND_RT_VISIBLE_DEVICES="${test_devices}" ASCEND_VISIBLE_DEVICES="${test_visible_devices}" \
        "$@" 2>&1 | tee "${logfile}"
    pipeline_status=("${PIPESTATUS[@]}")
    rc="${pipeline_status[0]}"
    tee_rc="${pipeline_status[1]}"
    set -e

    end_ts="$(date +%s)"
    elapsed=$((end_ts - start_ts))
    TEST_IDS+=("${id}")
    TEST_NAMES+=("${name}")
    TEST_DURATIONS+=("${elapsed}s")
    TEST_LOG_FILES+=("${logfile}")
    if [[ "${rc}" -eq 0 && "${tee_rc}" -eq 0 ]]; then
        TEST_RESULTS+=("PASS")
        pass "[${id}] ${name}  (${elapsed}s)"
    else
        TEST_RESULTS+=("FAIL")
        fail "[${id}] ${name}  (${elapsed}s)  exit=${rc} tee_exit=${tee_rc}"
    fi

    return 0
}

print_summary() {
    local requested_id
    if [[ -n "${NPU_SMOKE_TEST_IDS}" ]]; then
        for requested_id in "${NPU_SMOKE_REQUESTED_IDS[@]}"; do
            if [[ "${NPU_SMOKE_REGISTERED_IDS}" != *",${requested_id},"* ]]; then
                fail "Unknown ${NPU_SMOKE_GROUP} smoke test ID: ${requested_id}"
                return 2
            fi
        done
    fi
    [[ -n "${TEST_IDS[*]:-}" ]] || { fail "No smoke tests were selected."; return 2; }
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

dispatch_parallel() {
    if [[ "${NPU_SMOKE_PARALLEL}" == 1 && "${NPU_SMOKE_PARALLEL_CHILD}" == 0 \
        && "${NPU_SMOKE_FULL_TOPOLOGY}" == 0 ]]; then
        exec python3 tests/npu_smoke/run_npu_smoke_parallel.py "${NPU_SMOKE_GROUP}" \
            --num-npus "${NUM_NPUS}" --log-dir "${LOG_DIR}"
    fi
}

print_header() {
    sep
    echo "  verl-omni NPU Smoke Test: ${NPU_SMOKE_GROUP}"
    echo "  Date      : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Repo root : ${REPO_ROOT}"
    echo "  Log dir   : ${LOG_DIR}"
    echo "  NUM_NPUS  : ${NUM_NPUS}"
    echo "  Full topology : ${NPU_SMOKE_FULL_TOPOLOGY}"
    echo "  ASCEND_RT_VISIBLE_DEVICES : ${ASCEND_RT_VISIBLE_DEVICES}"
    sep
}
