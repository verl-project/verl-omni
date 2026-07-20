#!/usr/bin/env bash
# 4-GPU pruned fixed-sequence dump probe.
#
# Runs the known pruned local Megatron + standalone vLLM-Omni parity envelope
# and persists complete rollout-corr rows. Feed the emitted JSONL into
# run_gspo_qwen3_omni_megatron_pruned_4gpu_fixed_sequence_router_replay_r2_probe.sh
# to test Megatron scorer self-consistency without spending 32-GPU e2e cycles.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE_ROOT="${PROBE_ROOT:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/probes}"
mkdir -p "${PROBE_ROOT}"

export RUN_ID="${RUN_ID:-pruned4_fixed_sequence_dump_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME="${EXP_NAME:-qwen3_omni_megatron_pruned_4gpu_fixed_sequence_dump_${RUN_ID}}"
export PROFILE_LABEL="${PROFILE_LABEL:-4-GPU pruned fixed-sequence rollout-corr dump probe}"

export VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT="${VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT:-24}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL="${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL:-${PROBE_ROOT}/${RUN_ID}.rollout_corr_samples.jsonl}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS="${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS:-4}"

echo "[info] pruned 4-GPU fixed-sequence dump will be written to ${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL}"

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_pruned_4gpu_post_sync_logprob_parity_probe.sh" \
    trainer.experiment_name="${EXP_NAME}" \
    "$@"
