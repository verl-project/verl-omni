#!/usr/bin/env bash
# Single-node recipe for an allocation with all eight Ascend A5 devices.
set -euo pipefail

case "${ASCEND_RT_VISIBLE_DEVICES:-}" in
    ""|0,1,2,3,4,5,6,7) ;;
    *) echo "This recipe requires physical NPUs 0 through 7." >&2; exit 1 ;;
esac

export H3_IMAGE_DEVICE=npu
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6
export N_GPUS=4
export PREFERENCE_SCORER_GPU=7
export ROLLOUT_TP=2
export TEXT_ENCODER_TP=2
export REWARD_TP=1
export REWARD_GPU_MEM_UTIL=0.55
export ACTOR_ATTN_BACKEND=_native_npu
export ROLLOUT_ATTN_BACKEND=TORCH_SDPA
export VERL_OMNI_H3_IMAGE_ROLLOUT_KERNELS=1
export VERL_OMNI_H3_IMAGE_PACK_MICRO_BATCH=0
export REWARD_WORKER_TIMEOUT_S=${REWARD_WORKER_TIMEOUT_S:-1200}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-200}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-minimax_h3_image_ocr_lora_ascend}

script_dir=$(dirname "$(readlink -f "$0")")
exec bash "$script_dir/run_minimax_h3_image_ocr_lora.sh" \
    ++ray_kwargs.ray_init.address=local \
    ++ray_kwargs.ray_init.num_cpus=64 \
    ++ray_kwargs.ray_init.object_store_memory=4294967296 \
    trainer.device=npu \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=4 \
    actor_rollout_ref.actor.fsdp_config.offload_policy=True \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    reward.num_workers=3 \
    reward.reward_model.enable_resource_pool=True \
    reward.reward_model.nnodes=1 \
    reward.reward_model.n_gpus_per_node=3 \
    reward.reward_model.rollout.free_cache_engine=False \
    reward.reward_model.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    +reward.reward_functions.pick.required=true \
    +reward.reward_functions.aes.required=true \
    +reward.reward_functions.ocr.required=true \
    trainer.resume_mode=disable \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    "$@"
