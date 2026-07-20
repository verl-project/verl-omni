#!/usr/bin/env bash
set -euo pipefail

# 32-GPU fixed-sequence parity probe.
#
# This keeps the known-good fullmodel async split/logprob-ref probe intact and
# only asks verl debug metrics to persist a few complete samples
# (input_ids/responses/masks + rollout/actor/ref logprobs). Score the dump with
# run_qwen3_omni_hf_score_rollout_corr_dump.sh after the job finishes.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROBE_ROOT="${PROBE_ROOT:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/probes}"
mkdir -p "${PROBE_ROOT}"

export RUN_ID="${RUN_ID:-full32_fixed_sequence_parity_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME="${EXP_NAME:-qwen3_omni_megatron_full_32gpu_fixed_sequence_parity_${RUN_ID}}"
export PROFILE_LABEL="${PROFILE_LABEL:-32-GPU fullmodel fixed-sequence logprob parity probe}"

export VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT="${VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT:-16}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL="${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL:-${PROBE_ROOT}/${RUN_ID}.rollout_corr_samples.jsonl}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS="${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS:-4}"

echo "[info] fixed-sequence dump will be written to ${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL}"

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_logprob_ref_probe.sh"
