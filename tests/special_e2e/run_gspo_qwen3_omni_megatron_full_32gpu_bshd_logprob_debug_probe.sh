#!/usr/bin/env bash
set -euo pipefail

# 32-GPU controlled probe for Megatron BSHD/vocab-parallel logprob alignment.
#
# Keeps the known-good fixed-sequence async split setup.  This wrapper only
# enables a small Megatron-side dump around logits_processor/postprocess and
# compares Megatron CE logprobs with a manual TP-sharded target-logprob path.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROBE_DIR="/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/probes"
mkdir -p "${PROBE_DIR}"

export RUN_ID="${RUN_ID:-full32_bshd_logprob_debug_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME="${EXP_NAME:-qwen3_omni_megatron_full_32gpu_bshd_logprob_debug_${RUN_ID}}"
export PROFILE_LABEL="${PROFILE_LABEL:-32-GPU fullmodel Megatron bshd logprob debug probe}"
export VERL_OMNI_QWEN3_OMNI_BSHD_POSITION_IDS="${VERL_OMNI_QWEN3_OMNI_BSHD_POSITION_IDS:-explicit}"
export VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL="${VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL:-${PROBE_DIR}/${RUN_ID}.megatron_bshd_debug.jsonl}"
export VERL_OMNI_MEGATRON_BSHD_DEBUG_LIMIT="${VERL_OMNI_MEGATRON_BSHD_DEBUG_LIMIT:-2}"
export VERL_OMNI_MEGATRON_BSHD_DEBUG_ROWS="${VERL_OMNI_MEGATRON_BSHD_DEBUG_ROWS:-2}"
export VERL_OMNI_MEGATRON_BSHD_DEBUG_TOKENS="${VERL_OMNI_MEGATRON_BSHD_DEBUG_TOKENS:-24}"
export VERL_OMNI_MEGATRON_VOCAB_DEBUG_LIMIT="${VERL_OMNI_MEGATRON_VOCAB_DEBUG_LIMIT:-2}"

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_fixed_sequence_parity_probe.sh" "$@"
