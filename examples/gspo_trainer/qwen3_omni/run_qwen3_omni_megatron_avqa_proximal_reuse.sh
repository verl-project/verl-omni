#!/usr/bin/env bash
# EP1 variant of the image+audio AVQA recipe, using one optimizer update per batch.
# Same required MODEL_PATH, TRAIN_FILE and VAL_FILE; see the base recipe for knobs.
# Runtime qualification can still fall back to native old-policy inference.
set -euo pipefail

# Reuse qualification requires deterministic Transformer Engine attention.
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-0}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/run_qwen3_omni_megatron_avqa_separate_async.sh" \
  '+ray_kwargs.ray_init.runtime_env.env_vars.NVTE_ALLOW_NONDETERMINISTIC_ALGO=${oc.env:NVTE_ALLOW_NONDETERMINISTIC_ALGO,0}' \
  trainer.v1.step_extension=proximal_actor_reuse \
  actor_rollout_ref.actor.megatron.expert_model_parallel_size=1 \
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=null \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.shuffle=false \
  "$@"
