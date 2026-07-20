#!/usr/bin/env bash
set -xeuo pipefail

# Offline 4-node Megatron-Bridge parity control. It replays frozen token IDs
# through actor/ref workers only; vLLM-Omni, PPO, reward, and weight sync are
# never initialized. The actor topology remains the current production path
# (TP2/PP2/EP4/SP/alltoall) unless explicitly overridden at submission time.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT=${OUTPUT_ROOT:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs}
LOG_DIR=${LOG_DIR:-${OUTPUT_ROOT}/logs}
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"

export RUN_ID=${RUN_ID:-full32_megatron_bridge_hf_parity_${RUN_STAMP}}
export EXP_NAME=${EXP_NAME:-qwen3_omni_megatron_bridge_hf_parity_${RUN_ID}}
export PROFILE_LABEL=${PROFILE_LABEL:-32-GPU offline Megatron-Bridge actor-HF parity control}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL=${VERL_OMNI_FIXED_SEQUENCE_SCORE_JSONL:-${LOG_DIR}/rollout_corr_debug/full32_vllm_omni_multi_probe_20260716_123444.jsonl}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT=${VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT:-${LOG_DIR}/fixed_sequence_score/${RUN_ID}.jsonl}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_ROWS=${VERL_OMNI_FIXED_SEQUENCE_SCORE_ROWS:-8}
export VERL_OMNI_FIXED_SEQUENCE_SCORE_DIRECT_OLD=${VERL_OMNI_FIXED_SEQUENCE_SCORE_DIRECT_OLD:-1}
export VERL_OMNI_SKIP_WEIGHT_UPDATE=${VERL_OMNI_SKIP_WEIGHT_UPDATE:-1}
# The fixed-sequence probe has no work after rank-0 scoring on Ray workers.
# Let non-head pods exit once the launcher records its result.
export RAY_WORKER_EXIT_ON_MASTER_DONE=${RAY_WORKER_EXIT_ON_MASTER_DONE:-1}
export PARITY_GATE_OUTPUT=${PARITY_GATE_OUTPUT:-${VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT%.jsonl}.parity_gate.json}
export PARITY_MIN_LOGPROB_CORR=${PARITY_MIN_LOGPROB_CORR:-0.99}
export PARITY_GATE_STRICT=${PARITY_GATE_STRICT:-0}

bash "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_fixed_sequence_component_audit.sh" "$@"

if [[ "${CONFIG_ONLY:-0}" == "1" ]]; then
  exit 0
fi

PARITY_NODE_RANK=${DISTRIBUTED_NODE_RANK:-${NODE_RANK:-${RAY_NODE_RANK:-0}}}
if [[ "${RAY_WORKER_EXIT_ON_MASTER_DONE:-0}" == "1" && "${PARITY_NODE_RANK}" != "0" ]]; then
  echo "[info] Worker node rank ${PARITY_NODE_RANK}; rank 0 owns the parity gate."
  exit 0
fi

COMPONENT_COMPARE_OUTPUT=${COMPONENT_COMPARE_OUTPUT:-${VERL_OMNI_FIXED_SEQUENCE_SCORE_OUTPUT%.jsonl}.component_compare.jsonl}
gate_args=(
  --component-jsonl "${COMPONENT_COMPARE_OUTPUT}"
  --output-file "${PARITY_GATE_OUTPUT}"
  --min-logprob-corr "${PARITY_MIN_LOGPROB_CORR}"
)
if [[ "${PARITY_GATE_STRICT}" == "1" ]]; then
  gate_args+=(--strict)
fi
python3 "${SCRIPT_DIR}/qwen3_omni_check_hf_parity_gate.py" "${gate_args[@]}"
echo "PARITY_GATE_OUTPUT=${PARITY_GATE_OUTPUT}"
