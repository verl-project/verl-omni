#!/usr/bin/env bash
set -euo pipefail

# Deprecated 32-GPU probe for Qwen3-Omni sequence-parallel mRoPE position ids.
#
# This probe is intentionally disabled by default. The 20260711_183251 run
# proved that scattering position_ids at the thinker boundary makes Qwen3-VL
# attention see full-length query tensors with half-length rotary frequencies
# (for example 178 vs 89), so it fails before old_log_prob metrics are dumped.
# Keep the script for explicit reproduction only.

if [[ "${ALLOW_BROKEN_SP_POSITION_IDS_PROBE:-0}" != "1" ]]; then
  echo "[error] Disabled: SP position_ids scatter at the thinker boundary is known-broken." >&2
  echo "[error] Set ALLOW_BROKEN_SP_POSITION_IDS_PROBE=1 only to reproduce the 178-vs-89 RoPE shape failure." >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RUN_ID="${RUN_ID:-full32_sp_position_ids_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME="${EXP_NAME:-qwen3_omni_megatron_full_32gpu_sp_position_ids_${RUN_ID}}"
export PROFILE_LABEL="${PROFILE_LABEL:-32-GPU fullmodel sequence-parallel position-id parity probe}"
export VERL_OMNI_QWEN3_OMNI_BSHD_POSITION_IDS="${VERL_OMNI_QWEN3_OMNI_BSHD_POSITION_IDS:-model}"
export VERL_OMNI_QWEN3_OMNI_SP_SCATTER_POSITION_IDS=1

exec "${SCRIPT_DIR}/run_gspo_qwen3_omni_megatron_full_32gpu_fixed_sequence_parity_probe.sh" "$@"
