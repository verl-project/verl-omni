#!/usr/bin/env bash
# Qwen3-Omni Thinker GSPO on GSM8K with LoRA and true on-policy log-probs: the true on-policy recipe
# with the LoRA settings of the GSM8K LoRA recipe. The mode requires the adapters to be merged on the
# actor side (lora.merge=True); the actor then forwards with the merged weights the rollout receives.
# See docs/algo/true_on_policy.md. Arguments are appended as Hydra overrides.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

bash "${SCRIPT_DIR}/run_qwen3_omni_thinker_gspo_true_on_policy_v1.sh" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.model.exclude_modules=".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*" \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj']" \
    trainer.experiment_name=qwen3_omni_thinker_lora_true_on_policy \
    "$@"
