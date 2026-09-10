#!/usr/bin/env bash
# One-GPU, one-step Qwen-Image smoke test for all configurable GC diagnostics.
set -euo pipefail

kernel_release=$(uname -r)
if [[ ${kernel_release,,} == *microsoft-standard-wsl2* ]]; then
    export VLLM_WSL2_ENABLE_PIN_MEMORY=1
    export VERL_FORCE_SHM_WEIGHT_TRANSFER=1
fi

log_file=$(mktemp /tmp/verl-omni-gc-diagnostics-XXXXXX.log)

if ! NUM_GPUS=1 \
    bash tests/special_e2e/run_flowgrpo_qwen_image.sh \
        actor_rollout_ref.actor.strategy=fsdp2 \
        actor_rollout_ref.gc_diagnostics=True \
        actor_rollout_ref.actor.fsdp_config.gc_on_train_device_load=0 \
        actor_rollout_ref.actor.fsdp_config.gc_on_eval_device_load=1 \
        actor_rollout_ref.rollout.gc_on_actor_offload=2 \
        actor_rollout_ref.rollout.checkpoint_engine.gc_on_weight_transfer_cleanup=True \
        reward.reward_model.enable=False \
        reward.custom_reward_function.path=null \
        reward.custom_reward_function.name=null \
        reward.reward_manager.name=VisualRewardManager \
        trainer.total_training_steps=1 \
        "$@" \
        2>&1 | tee "${log_file}"; then
    echo "GC diagnostics smoke failed; log preserved at ${log_file}" >&2
    exit 1
fi

for expected in \
    train_device_load:0 \
    eval_device_load:1 \
    actor_offload:2 \
    weight_transfer_cleanup:full; do
    point=${expected%%:*}
    generation=${expected#*:}
    if ! grep -q "\\[gc_diagnostics\\] point=${point} .* generation=${generation} " "${log_file}"; then
        echo "Missing GC diagnostics point=${point} generation=${generation}; log preserved at ${log_file}" >&2
        exit 1
    fi
done

rm -f "${log_file}"
echo "GC diagnostics smoke passed."
