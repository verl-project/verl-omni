#!/usr/bin/env bash
set -euo pipefail

# Golden wrapper based on the 2026-07-02 19:42:57 successful B-gate run:
#   outputs/logs/qwen3_omni_megatron_full_32gpu_weight_sync_b_gate_full32_weight_sync_b_gate_20260702_194257.log
#
# Keep the successful probe envelope and only run the same B-gate observability.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RUN_ID="${RUN_ID:-full32_weight_sync_b_gate_success194257_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME="${EXP_NAME:-qwen3_omni_megatron_full_32gpu_weight_sync_b_gate_${RUN_ID}}"
export PROFILE_LABEL="${PROFILE_LABEL:-32-GPU fullmodel weight-sync B-gate success194257 envelope}"

# The successful probe used a single vLLM internal actor base (61512) and did
# not route the stage-core TCPStore through the later experimental MASTER_PORT
# override. Preserve that behavior for this wrapper.
export RAY_PORT_SEED="${RAY_PORT_SEED:-k8s-vj-dp0ief-1782992110515}"
export RAY_WORKER_PORT_SEED="${RAY_WORKER_PORT_SEED:-${RAY_PORT_SEED}}"
export VERL_OMNI_VLLM_PORT_SEED="${VERL_OMNI_VLLM_PORT_SEED:-${RAY_PORT_SEED}}"
export VLLM_PORT="${VLLM_PORT:-61512}"
export VERL_OMNI_RESPECT_EXISTING_VLLM_PORT="${VERL_OMNI_RESPECT_EXISTING_VLLM_PORT:-1}"
export VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE="${VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE:-0}"

# Only intended delta from the completed 194257 probe: force raw rollout
# logprobs so corr/KL reflects sampled-token logprobs rather than processed
# logits.
export ROLLOUT_LOGPROBS_MODE="${ROLLOUT_LOGPROBS_MODE:-processed_logprobs}"
export VERL_OMNI_REQUIRE_PROCESSED_ROLLOUT_LOGPROBS="${VERL_OMNI_REQUIRE_PROCESSED_ROLLOUT_LOGPROBS:-1}"
export VERL_OMNI_LOGPROB_DEBUG_LIMIT="${VERL_OMNI_LOGPROB_DEBUG_LIMIT:-0}"

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_weight_sync_b_gate.sh"
