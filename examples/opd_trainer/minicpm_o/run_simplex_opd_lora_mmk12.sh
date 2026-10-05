#!/usr/bin/env bash
# MiniCPM-o 4.5 simplex thinker OPD with MMK12 task rewards.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
REWARD_FUNCTION_PATH=${REWARD_FUNCTION_PATH:-"${REPO_ROOT}/verl_omni/utils/reward_score/mmk12_reward.py"}

exec bash "${SCRIPT_DIR}/run_simplex_opd_lora.sh" \
    data.filter_overlong_prompts=true \
    distillation.distillation_loss.use_task_rewards=true \
    reward.custom_reward_function.path="${REWARD_FUNCTION_PATH}" \
    reward.custom_reward_function.name=compute_score \
    trainer.experiment_name=minicpm-o45-simplex-mmk12-opd \
    trainer.test_freq=10 \
    "$@"
