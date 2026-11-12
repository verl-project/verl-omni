#!/usr/bin/env bash

set -euo pipefail

export NPU_SMOKE_GROUP=e2e
source "$(dirname "${BASH_SOURCE[0]}")/lib_npu_smoke.sh" "$@"

print_header
trainer_args=()
if [[ -n "${RAY_MASTER_PORT_RANGE:-}" ]]; then
    trainer_args+=("trainer.ray_master_port_range=[${RAY_MASTER_PORT_RANGE}]")
fi
if [[ "${NPU_SMOKE_SKIP_GLOBAL_CLEANUP:-0}" == 1 ]]; then
    trainer_args+=("+ray_kwargs.ray_init.address=local")
fi
run_test 1 "FlowGRPO trainer e2e" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
        DATA_DIR="${DATA_DIR:-${LOG_DIR}/test_1_data}" \
    bash tests/special_e2e/run_flowgrpo_qwen_image_npu.sh "${trainer_args[@]}"
run_test 2 "DanceGRPO Wan2.2 trainer e2e" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
        DATA_DIR="${DATA_DIR:-${LOG_DIR}/test_2_data}" \
    bash tests/special_e2e/run_dancegrpo_wan22_npu.sh "${trainer_args[@]}"
run_test 29 "Qwen3-TTS full-parameter GRPO" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u TRAIN_FILES -u VAL_FILES \
        WORK_DIR="${LOG_DIR}/test29_work" \
    bash tests/special_e2e/run_qwen3_tts_grpo_npu.sh
run_test 35 "SD3 CRD training, EMA and checkpoint restore" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u TRAIN_FILES -u VAL_FILES \
    bash tests/special_e2e/run_crd_sd35_npu.sh
cleanup_runtime
print_summary
