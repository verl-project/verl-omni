#!/usr/bin/env bash
set -euo pipefail

export NPU_SMOKE_GROUP=parallel
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib_npu_smoke.sh" "$@"
trap - EXIT

if [[ "${NUM_NPUS}" != 8 || "${ASCEND_RT_VISIBLE_DEVICES}" != 0,1,2,3,4,5,6,7 ]]; then
    fail "Parallel smoke requires the complete physical device list 0,1,2,3,4,5,6,7 and --num-npus 8."
    exit 2
fi
for variable in MODEL_PATH TOKENIZER_PATH DATA_DIR TRAIN_FILES VAL_FILES NPU_SMOKE_TEST_IDS; do
    if [[ -n "${!variable:-}" ]]; then
        fail "Unset ${variable} before running multiple model workloads."
        exit 2
    fi
done
if [[ "${NPU_SMOKE_SKIP_GLOBAL_CLEANUP:-0}" != 0 ]]; then
    fail "Only child groups may skip global cleanup."
    exit 2
fi
for variable in ${!RAY_EXPERIMENTAL_NOSET_@}; do
    export "${variable}="
done
unset RAY_ADDRESS

ray_root=$(mktemp -d /tmp/npu-smoke.XXXXXXXX)
printf '%s\n' "${ray_root}" >"${LOG_DIR}/ray-runtime.path"
pids=()
names=()
finish() {
    local rc=$?
    trap - EXIT INT TERM
    local pid
    for pid in "${pids[@]}"; do
        kill -TERM -- "-${pid}" 2>/dev/null || true
    done
    cleanup_runtime
    return "${rc}"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cleanup_runtime
print_header

start_group() {
    local group=$1 devices=$2 count=$3 ports=$4
    mkdir -p "${LOG_DIR}/${group}" "${ray_root}/${group}"
    log "Starting ${group} on physical NPUs ${devices}; log: ${LOG_DIR}/${group}.log"
    # Only the e2e lane performs colocated weight transfers; its trainers stay serial.
    setsid env NUM_NPUS="${count}" ASCEND_RT_VISIBLE_DEVICES="${devices}" ASCEND_VISIBLE_DEVICES="${devices}" \
        NPU_SMOKE_SKIP_GLOBAL_CLEANUP=1 RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES= \
        RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 RAY_TMPDIR="${ray_root}/${group}" \
        RAY_MASTER_PORT_RANGE="${ports}" LOG_DIR="${LOG_DIR}/${group}" \
        bash "${SCRIPT_DIR}/run_npu_smoke_${group}.sh" --num-npus "${count}" \
        >"${LOG_DIR}/${group}.log" 2>&1 &
    pids+=("$!")
    names+=("${group}")
}

start_group core 0,1 2 10000,20000
start_group e2e 2,3,4,5,6,7 6 26000,27000
failed=0
for i in "${!pids[@]}"; do
    rc=0
    wait "${pids[$i]}" || rc=$?
    printf '%s\t%s\n' "${names[$i]}" "${rc}" >>"${LOG_DIR}/group-exits.tsv"
    if [[ "${rc}" -eq 0 ]]; then
        pass "${names[$i]}"
    else
        fail "${names[$i]} exit=${rc}; see ${LOG_DIR}/${names[$i]}.log"
        failed=1
    fi
done
exit "${failed}"
