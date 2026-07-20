#!/usr/bin/env bash
set -euo pipefail

# 32-GPU controlled probe for the Qwen3-Omni Megatron bshd position-id contract.
#
# Baseline fixed-sequence parity showed vLLM-Omni rollout matches HF, while
# Megatron actor/ref logprobs drift far away. This wrapper keeps the known-good
# async split/fixed-sequence setup and only forces Megatron Qwen3-Omni bshd
# scoring to consume the precomputed position_ids from verl.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RUN_ID="${RUN_ID:-full32_bshd_explicit_pos_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME="${EXP_NAME:-qwen3_omni_megatron_full_32gpu_bshd_explicit_pos_${RUN_ID}}"
export PROFILE_LABEL="${PROFILE_LABEL:-32-GPU fullmodel bshd explicit position-id parity probe}"
export VERL_OMNI_QWEN3_OMNI_BSHD_POSITION_IDS=explicit

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_fixed_sequence_parity_probe.sh" "$@"
