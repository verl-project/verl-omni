#!/usr/bin/env bash
# 4-GPU reduced-topology Megatron/vLLM-Omni logprob debug probe.
#
# This intentionally uses the known-good pruned 4-GPU split path and only adds
# the Megatron-side BSHD/vocab debug dumps plus rollout-corr sample dumping.
# It does not reproduce the 32-GPU fullmodel TP2/PP2/EP4 topology.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE_DIR="${PROBE_DIR:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/probes}"
mkdir -p "${PROBE_DIR}"

export RUN_ID="${RUN_ID:-pruned4_bshd_vocab_debug_$(date +%Y%m%d_%H%M%S)}"
export LOG_ROOT="${LOG_ROOT:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs/logs}"

# Keep the local topology small and controlled: 2 train GPUs + 2 rollout GPUs.
export NUM_GPUS="${NUM_GPUS:-4}"
export TRAIN_GPUS="${TRAIN_GPUS:-2}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Use the same debug surface as the 32-GPU fullmodel BSHD/vocab probe.
export VERL_OMNI_QWEN3_OMNI_BSHD_POSITION_IDS="${VERL_OMNI_QWEN3_OMNI_BSHD_POSITION_IDS:-explicit}"
export VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL="${VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL:-${PROBE_DIR}/${RUN_ID}.megatron_bshd_debug.jsonl}"
export VERL_OMNI_MEGATRON_BSHD_DEBUG_LIMIT="${VERL_OMNI_MEGATRON_BSHD_DEBUG_LIMIT:-2}"
export VERL_OMNI_MEGATRON_BSHD_DEBUG_ROWS="${VERL_OMNI_MEGATRON_BSHD_DEBUG_ROWS:-2}"
export VERL_OMNI_MEGATRON_BSHD_DEBUG_TOKENS="${VERL_OMNI_MEGATRON_BSHD_DEBUG_TOKENS:-24}"
export VERL_OMNI_MEGATRON_VOCAB_DEBUG_LIMIT="${VERL_OMNI_MEGATRON_VOCAB_DEBUG_LIMIT:-2}"

# Persist complete samples so HF scoring can be run after the probe.
export VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT="${VERL_OMNI_ROLLOUT_CORR_DEBUG_LIMIT:-24}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL="${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL:-${PROBE_DIR}/${RUN_ID}.rollout_corr_samples.jsonl}"
export VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS="${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL_ROWS:-4}"

echo "[info] reduced 4-GPU rollout-corr dump: ${VERL_OMNI_ROLLOUT_CORR_DEBUG_JSONL}"
echo "[info] reduced 4-GPU Megatron BSHD dump prefix: ${VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL}"

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_pruned_4gpu_post_sync_logprob_parity_probe.sh" "$@"
