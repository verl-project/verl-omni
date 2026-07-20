#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper over the known-good logprob parity probe. The code now emits
# rollout/actor/ref three-way metrics after ref_log_prob is computed, so keep
# runtime knobs identical and only stamp a distinct run name.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RUN_ID="${RUN_ID:-full32_logprob_ref_probe_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME="${EXP_NAME:-qwen3_omni_megatron_full_32gpu_logprob_ref_${RUN_ID}}"
export PROFILE_LABEL="${PROFILE_LABEL:-32-GPU fullmodel logprob actor/ref/rollout probe}"

# Thinker-only AR uses one vLLM-Omni stage replica per rollout actor. Reuse
# the reserved worker MASTER_PORT for the first stage-core TCPStore port, which
# avoids the probe/release race seen in the generic stage-core slice allocator.
export VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE="${VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE:-1}"

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_logprob_parity_probe.sh"
