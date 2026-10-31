#!/usr/bin/env bash

set -euo pipefail

export NPU_SMOKE_GROUP=e2e
source "$(dirname "${BASH_SOURCE[0]}")/lib_npu_smoke.sh" "$@"

print_header
run_test 1 "FlowGRPO trainer e2e" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u TRAIN_FILES -u VAL_FILES \
        ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
        DATA_DIR="${LOG_DIR}/test1_data" \
    bash tests/special_e2e/run_flowgrpo_qwen_image_npu.sh
run_test 2 "DanceGRPO Wan2.2 trainer e2e" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u TRAIN_FILES -u VAL_FILES \
        ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
        DATA_DIR="${LOG_DIR}/test2_data" \
    bash tests/special_e2e/run_dancegrpo_wan22_npu.sh
run_test 18 "Qwen-Image FlowGRPO V1 sync trainer e2e" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u TRAIN_FILES -u VAL_FILES \
        ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
        DATA_DIR="${LOG_DIR}/test18_data" \
    bash tests/special_e2e/run_flowgrpo_qwen_image_v1_npu.sh
run_test 19 "Qwen-Image-Edit FlowGRPO V1 sync trainer e2e" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u TRAIN_FILES -u VAL_FILES \
        ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
        DATA_DIR="${LOG_DIR}/test19_data" \
    bash tests/special_e2e/run_flowgrpo_qwen_image_edit_npu.sh
run_test 20 "Qwen-Image online DPO V1 sync trainer e2e" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u TRAIN_FILES -u VAL_FILES \
        ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
        DATA_DIR="${LOG_DIR}/test20_data" \
    bash tests/special_e2e/run_online_dpo_qwen_image_npu.sh
run_test 21 "DiffusionNFT V1 sync trainer e2e" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u TRAIN_FILES -u VAL_FILES \
        ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
        DATA_DIR="${LOG_DIR}/test21_data" \
    bash tests/special_e2e/run_diffusionnft_qwen_image_npu.sh
run_test 22 "FlowGRPO V1 separate-async trainer e2e" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u TRAIN_FILES -u VAL_FILES \
        ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
        DATA_DIR="${LOG_DIR}/test22_data" \
    bash tests/special_e2e/run_flowgrpo_qwen_image_v1_separate_async_npu.sh
run_test 23 "Diffusion OPD V1 sync colocated teachers e2e" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u TRAIN_FILES -u VAL_FILES \
        ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
        DATA_DIR="${LOG_DIR}/test23_data" \
    SMOKE=sync bash tests/special_e2e/run_diffusion_teacher_smoke_npu.sh
run_test 24 "Diffusion OPD V1 separate-async standalone teachers e2e" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u TRAIN_FILES -u VAL_FILES \
        ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
        DATA_DIR="${LOG_DIR}/test24_data" \
    SMOKE=async bash tests/special_e2e/run_diffusion_teacher_smoke_npu.sh
run_test 25 "Bagel PickScore LoRA FlowGRPO e2e (v0)" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u TRAIN_FILES -u VAL_FILES \
        ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
        DATA_DIR="${LOG_DIR}/test25_data" \
    bash tests/special_e2e/run_flowgrpo_bagel_pickscore_npu.sh
run_test 26 "FlowGRPO synchronous separate trainer e2e (v0)" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u TRAIN_FILES -u VAL_FILES \
        ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
        DATA_DIR="${LOG_DIR}/test26_data" \
    bash tests/special_e2e/run_flowgrpo_qwen_image_separate_npu.sh
run_test 34 "MiniMax-H3 T2VA FlowGRPO trainer e2e" \
    env -u MODEL_PATH -u TOKENIZER_PATH -u TRAIN_FILES -u VAL_FILES \
        ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
        DATA_DIR="${LOG_DIR}/test34_data" \
    bash tests/special_e2e/run_flowgrpo_minimax_h3_npu.sh
cleanup_runtime
print_summary
