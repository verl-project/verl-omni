#!/usr/bin/env bash
set -euo pipefail

# 32-GPU fullmodel async probe after the Megatron bshd/Qwen3-Omni MRoPE
# position_ids fix. Keep the known-good topology and only stamp a distinct
# run name for log comparison.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RUN_ID="${RUN_ID:-full32_mrope_bshd_fix_probe_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME="${EXP_NAME:-qwen3_omni_megatron_full_32gpu_mrope_bshd_fix_${RUN_ID}}"
export PROFILE_LABEL="${PROFILE_LABEL:-32-GPU fullmodel MRoPE bshd position_ids fix probe}"

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_logprob_ref_probe.sh"
