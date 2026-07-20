#!/usr/bin/env bash
set -xeuo pipefail

# One controlled multi-node scorer run. It reuses a frozen rollout dump and
# emits response-token target-logit/LSE/logprob components from two actor
# forwards plus the ref scorer. No vLLM startup or weight sync is involved.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT=${OUTPUT_ROOT:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs}
LOG_DIR=${LOG_DIR:-${OUTPUT_ROOT}/logs}
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

export RUN_ID=${RUN_ID:-full32_fixed_sequence_component_audit_${RUN_STAMP}}
export EXP_NAME=${EXP_NAME:-qwen3_omni_megatron_full_32gpu_fixed_sequence_component_audit_${RUN_ID}}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL=${VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL:-${LOG_DIR}/rollout_corr_debug/full32_vllm_omni_multi_probe_20260716_123444.jsonl}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT=${VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT:-${LOG_DIR}/fixed_sequence_score/${RUN_ID}.jsonl}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_ROWS=${VERL_OMNI_FIXED_SEQUENCE_SCORE_ROWS:-4}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_DIRECT_OLD=1
export VERL_OMNI_SKIP_WEIGHT_UPDATE=1
export VERL_OMNI_MEGATRON_LOGPROB_COMPONENT_AUDIT=1
export VERL_OMNI_MEGATRON_DECODER_COMPONENT_AUDIT=${VERL_OMNI_MEGATRON_DECODER_COMPONENT_AUDIT:-1}
export VERL_OMNI_MEGATRON_DECODER_COMPONENT_AUDIT_LIMIT=${VERL_OMNI_MEGATRON_DECODER_COMPONENT_AUDIT_LIMIT:-1}
export VERL_OMNI_MEGATRON_MOE_ROUTER_AUDIT=${VERL_OMNI_MEGATRON_MOE_ROUTER_AUDIT:-1}
export VERL_OMNI_MEGATRON_MOE_ROUTER_AUDIT_LAYERS=${VERL_OMNI_MEGATRON_MOE_ROUTER_AUDIT_LAYERS:-1,4,6,16,32,48}
export VERL_OMNI_MEGATRON_ATTENTION_AUDIT_LAYERS=${VERL_OMNI_MEGATRON_ATTENTION_AUDIT_LAYERS:-1}
export VERL_OMNI_MEGATRON_ATTENTION_LOCAL_REFERENCE=${VERL_OMNI_MEGATRON_ATTENTION_LOCAL_REFERENCE:-1}
export VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL=${VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL:-${VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT%.jsonl}.bshd_debug}
export VERL_OMNI_MEGATRON_BSHD_DEBUG_LIMIT=${VERL_OMNI_MEGATRON_BSHD_DEBUG_LIMIT:-4}
export VERL_OMNI_MEGATRON_VOCAB_DEBUG_LIMIT=${VERL_OMNI_MEGATRON_VOCAB_DEBUG_LIMIT:-4}
export VERL_OMNI_MEGATRON_BSHD_DEBUG_ROWS=${VERL_OMNI_MEGATRON_BSHD_DEBUG_ROWS:-1}
export VERL_OMNI_MEGATRON_BSHD_DEBUG_TOKENS=${VERL_OMNI_MEGATRON_BSHD_DEBUG_TOKENS:-8}

bash "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_fixed_sequence_score_probe.sh" "$@"

if [[ "${CONFIG_ONLY:-0}" == "1" ]]; then
  exit 0
fi

# The nested launcher exits worker pods after rank 0 writes its completion
# marker. Do not let those pods fall through into duplicate local HF scoring.
AUDIT_NODE_RANK=${DISTRIBUTED_NODE_RANK:-${NODE_RANK:-${RAY_NODE_RANK:-0}}}
if [[ "${RAY_WORKER_EXIT_ON_MASTER_DONE:-0}" == "1" && "${AUDIT_NODE_RANK}" != "0" ]]; then
  echo "[info] Worker node rank ${AUDIT_NODE_RANK}; rank 0 owns the HF component audit."
  exit 0
fi

HF_SCORE_OUTPUT=${HF_SCORE_OUTPUT:-${VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT%.jsonl}.hf_components.jsonl}
COMPONENT_COMPARE_OUTPUT=${COMPONENT_COMPARE_OUTPUT:-${VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT%.jsonl}.component_compare.jsonl}
DECODER_COMPARE_OUTPUT=${DECODER_COMPARE_OUTPUT:-${VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT%.jsonl}.decoder_component_compare.json}
ATTENTION_COMPARE_OUTPUT=${ATTENTION_COMPARE_OUTPUT:-${VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT%.jsonl}.attention_stage_compare.json}
MOE_ROUTER_COMPARE_OUTPUT=${MOE_ROUTER_COMPARE_OUTPUT:-${VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT%.jsonl}.moe_router_compare.json}
REPLAY_AUDIT_OUTPUT=${REPLAY_AUDIT_OUTPUT:-${VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT%.jsonl}.router_replay_compare.json}
INDEX_TRACE_OUTPUT=${INDEX_TRACE_OUTPUT:-${VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT%.jsonl}.router_replay_index_compare.json}
HF_CUDA_VISIBLE_DEVICES=${HF_CUDA_VISIBLE_DEVICES:-0,1,2,3}
HF_MOE_ROUTER_AUDIT_LAYERS=${HF_MOE_ROUTER_AUDIT_LAYERS:-${VERL_OMNI_MEGATRON_MOE_ROUTER_AUDIT_LAYERS}}
CUDA_VISIBLE_DEVICES="${HF_CUDA_VISIBLE_DEVICES}" \
  SCORE_TEMPERATURE=${SCORE_TEMPERATURE:-0.8} \
  RECORD_LIMIT=${RECORD_LIMIT:-${VERL_OMNI_FIXED_SEQUENCE_SCORE_ROWS}} \
  OUTPUT_FILE="${HF_SCORE_OUTPUT}" \
  HF_ATTENTION_AUDIT_TP_SIZE=${HF_ATTENTION_AUDIT_TP_SIZE:-2} \
  "${SCRIPT_DIR}/run_qwen3_omni_hf_score_rollout_corr_dump.sh" "${VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL}" \
  --score-temperature "${SCORE_TEMPERATURE:-0.8}" \
  --attention-audit-tp-size "${HF_ATTENTION_AUDIT_TP_SIZE:-2}" \
  --moe-router-audit-layers "${HF_MOE_ROUTER_AUDIT_LAYERS}"

python3 "${SCRIPT_DIR}/qwen3_omni_compare_logprob_components.py" \
  --hf-jsonl "${HF_SCORE_OUTPUT}" \
  --megatron-vocab-glob "${VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL}.vocab.rank*.jsonl" \
  --output-file "${COMPONENT_COMPARE_OUTPUT}"

python3 "${SCRIPT_DIR}/qwen3_omni_compare_decoder_components.py" \
  --hf-jsonl "${HF_SCORE_OUTPUT}" \
  --megatron-decoder-glob "${VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL}.decoder.rank*.jsonl" \
  --output-file "${DECODER_COMPARE_OUTPUT}"

python3 "${SCRIPT_DIR}/qwen3_omni_compare_attention_stages.py" \
  --hf-jsonl "${HF_SCORE_OUTPUT}" \
  --megatron-decoder-glob "${VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL}.decoder.rank*.jsonl" \
  --output-file "${ATTENTION_COMPARE_OUTPUT}"

python3 "${SCRIPT_DIR}/qwen3_omni_compare_moe_routing.py" \
  --hf-jsonl "${HF_SCORE_OUTPUT}" \
  --megatron-decoder-glob "${VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL}.decoder.rank*.jsonl" \
  --output-file "${MOE_ROUTER_COMPARE_OUTPUT}"

if [[ "${VERL_OMNI_MEGATRON_ROUTER_REPLAY_AUDIT:-0}" == "1" ]]; then
  python3 "${SCRIPT_DIR}/qwen3_omni_compare_megatron_router_replay.py" \
    --megatron-decoder-glob "${VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL}.decoder.rank*.jsonl" \
    --output-file "${REPLAY_AUDIT_OUTPUT}"
  echo "REPLAY_AUDIT_OUTPUT=${REPLAY_AUDIT_OUTPUT}"
fi

if [[ "${VERL_OMNI_ROUTER_REPLAY_INDEX_TRACE:-0}" == "1" ]]; then
  python3 "${SCRIPT_DIR}/qwen3_omni_compare_router_replay_index_trace.py" \
    --trace-glob "${VERL_OMNI_MEGATRON_BSHD_DEBUG_JSONL}.router_replay_indices.rank*.jsonl" \
    --output-file "${INDEX_TRACE_OUTPUT}"
  echo "INDEX_TRACE_OUTPUT=${INDEX_TRACE_OUTPUT}"
fi

echo "HF_SCORE_OUTPUT=${HF_SCORE_OUTPUT}"
echo "COMPONENT_COMPARE_OUTPUT=${COMPONENT_COMPARE_OUTPUT}"
echo "DECODER_COMPARE_OUTPUT=${DECODER_COMPARE_OUTPUT}"
echo "ATTENTION_COMPARE_OUTPUT=${ATTENTION_COMPARE_OUTPUT}"
echo "MOE_ROUTER_COMPARE_OUTPUT=${MOE_ROUTER_COMPARE_OUTPUT}"
