#!/usr/bin/env bash
# Single-node, eight-NPU Ascend 950 recipe using the official image-only Actor.
set -euo pipefail

case "${ASCEND_RT_VISIBLE_DEVICES:-}" in
    "")
        for device in /dev/davinci{0..7}; do
            if ! test -c "$device"; then
                echo "This recipe requires physical NPUs 0 through 7; missing $device." >&2
                exit 1
            fi
        done
        ;;
    0,1,2,3,4,5,6,7) ;;
    *) echo "This recipe requires an allocation of physical NPUs 0 through 7." >&2; exit 1 ;;
esac
: "${MODEL_PATH:?Set MODEL_PATH to the native MiniMax H3 Image release root}"
export ACTOR_PATH=${ACTOR_PATH:-$MODEL_PATH/transformer_diffusers}
python3 - "$ACTOR_PATH/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    name = json.load(handle).get("_class_name")
if name != "MiniMaxH3ImageTransformer2DModel":
    raise SystemExit("ACTOR_PATH must contain official MiniMaxH3ImageTransformer2DModel weights.")
PY

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
export VERL_OMNI_H3_IMAGE_ROLLOUT_KERNELS=0
export VERL_OMNI_H3_IMAGE_PACK_MICRO_BATCH=0
export REWARD_WORKER_TIMEOUT_S=${REWARD_WORKER_TIMEOUT_S:-1200}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-1800}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-minimax_h3_image_ocr_lora_ascend}

script_dir=$(dirname "$(readlink -f "$0")")
output_dir=${OUTPUT_DIR:-$script_dir/outputs/$EXPERIMENT_NAME}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-"$output_dir/tensorboard"}
mkdir -p "$TENSORBOARD_DIR"
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
    +reward.reward_functions.hps.required=true \
    +reward.reward_functions.ocr.required=true \
    trainer.resume_mode=disable \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    "$@"
