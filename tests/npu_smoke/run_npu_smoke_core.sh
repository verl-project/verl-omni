#!/usr/bin/env bash

set -euo pipefail

export NPU_SMOKE_GROUP=core
source "$(dirname "${BASH_SOURCE[0]}")/lib_npu_smoke.sh" "$@"

print_header
run_test 0 "vllm-omni rollout + sleep/wake_up" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    pytest -s tests/workers/rollout/rollout_vllm/test_vllm_omni_generate_npu.py
run_test 3 "diffusers FSDP engine inference + train" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    pytest -s tests/workers/test_diffusers_fsdp_engine_npu.py
cleanup_runtime
print_summary
