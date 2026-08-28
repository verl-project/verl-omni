#!/usr/bin/env bash
# MiniMax H3 T2VA LoRA FlowGRPO with CLAP and ImageBind rewards (Ascend NPU).
set -x

export WANDB_MODE=${WANDB_MODE:-online}
export VERL_DATAPROTO_SERIALIZATION_METHOD=numpy
ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit}

source $ASCEND_HOME_PATH/set_env.sh
source $ASCEND_HOME_PATH/../nnal/atb/set_env.sh

WORKSPACE=${WORKSPACE:-$HOME}
MODEL_PATH=${MODEL_PATH:-$WORKSPACE/models/MiniMax-H3/FL2VA}
DATA_DIR=${DATA_DIR:-$WORKSPACE/data/vid_prompt/verl_omni}
CLAP_MODEL_PATH=${CLAP_MODEL_PATH:-laion/larger_clap_general}
IMAGEBIND_MODEL_PATH=${IMAGEBIND_MODEL_PATH:-.checkpoints/imagebind_huge.pth}
ACTOR_CONFIG_PATH=${ACTOR_CONFIG_PATH:-$(dirname "$MODEL_PATH")/transformer}
NUM_GPUS=${NUM_GPUS:-16}
ROLLOUT_TP=${ROLLOUT_TP:-4}
TEXT_ENCODER_TP=${TEXT_ENCODER_TP:-$ROLLOUT_TP}
REWARD_DEVICE=${REWARD_DEVICE:-npu}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
ASPECT_RATIO=${ASPECT_RATIO:-16:9}
HEIGHT=${HEIGHT:-256}
WIDTH=${WIDTH:-384}
NUM_FRAMES=${NUM_FRAMES:-121}
INFER_STEPS=${INFER_STEPS:-10}
VAL_HEIGHT=${VAL_HEIGHT:-512}
VAL_WIDTH=${VAL_WIDTH:-768}

train_path=$DATA_DIR/train.parquet
test_path=$DATA_DIR/test.parquet

script_path=$(readlink -f "$0")
script_name=$(basename "$script_path" .sh)
repo_root=$(dirname "$script_path")
while [[ "$repo_root" != "/" && ! -f "$repo_root/LICENSE" ]]; do
    repo_root=$(dirname "$repo_root")
done
if [[ ! -f "$repo_root/LICENSE" ]]; then
    echo "Unable to locate repo root from $script_path: no LICENSE found" >&2
    exit 1
fi

output_dir=${OUTPUT_DIR:-$repo_root/outputs/$script_name}
checkpoint_dir=$output_dir/checkpoints
run_timestamp=$(date +"%Y%m%d_%H%M")
log_file=$output_dir/logs/$run_timestamp/${NODE_RANK:-0}.log
mkdir -p "$checkpoint_dir" "$(dirname "$log_file")"
exec > >(tee -a "$log_file") 2>&1

h3_lora_targets="['to_q','to_k','to_v','to_out.0','ff.net.0.proj','ff.net.2']"

python3 -m verl_omni.trainer.main_diffusion \
    trainer.device=npu \
    data.train_files=$train_path \
    data.val_files=$test_path \
    data.train_batch_size=32 \
    data.val_max_samples=128 \
    data.max_prompt_length=1024 \
    data.truncation=error \
    data.seed=42 \
    algorithm.adv_estimator=flow_grpo \
    algorithm.global_std=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.config_path=$ACTOR_CONFIG_PATH \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.transformer_subfolder=transformer \
    actor_rollout_ref.model.attn_backend=_native_npu \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules="$h3_lora_targets" \
    actor_rollout_ref.model.fsdp_layer_prefixes="['transformer_blocks.','token_refiner.refiner_blocks.']" \
    '+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[MiniMaxH3TransformerBlock,MiniMaxH3TokenRefinerBlock]' \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.max_num_seqs=1 \
    actor_rollout_ref.rollout.layered_summon=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.text_encoder_tp_size=$TEXT_ENCODER_TP \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.agent.default_agent_loop=minimax_h3_diffusion_single_turn_agent \
    actor_rollout_ref.rollout.max_prompt_embed_length=1024 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.pipeline.height=$HEIGHT \
    actor_rollout_ref.rollout.pipeline.width=$WIDTH \
    actor_rollout_ref.rollout.pipeline.aspect_ratio=$ASPECT_RATIO \
    actor_rollout_ref.rollout.pipeline.num_frames=$NUM_FRAMES \
    actor_rollout_ref.rollout.pipeline.frame_rate=24 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=$INFER_STEPS \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=1024 \
    +actor_rollout_ref.rollout.pipeline.output_type=np \
    actor_rollout_ref.rollout.val_kwargs.pipeline.height=$VAL_HEIGHT \
    actor_rollout_ref.rollout.val_kwargs.pipeline.width=$VAL_WIDTH \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_frames=$NUM_FRAMES \
    actor_rollout_ref.rollout.val_kwargs.pipeline.frame_rate=24.0 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=40 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.true_cfg_scale=1.0 \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.output_type=pt \
    actor_rollout_ref.rollout.algo.noise_level=0.8 \
    actor_rollout_ref.rollout.algo.sde_type=cps \
    actor_rollout_ref.rollout.algo.sde_window_range='[0,8]' \
    actor_rollout_ref.rollout.algo.sde_window_size=3 \
    actor_rollout_ref.rollout.algo.sde_contiguous=True \
    actor_rollout_ref.rollout.algo.sde_window_seed=42 \
    reward.num_workers=$REWARD_NUM_WORKERS \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path=pkg://verl_omni.reward_loop.reward_manager.multi \
    reward.custom_reward_function.name=_multi_reward_placeholder \
    reward.reward_manager.name=MultiVisualRewardManager \
    reward.reward_manager.module.path=pkg://verl_omni.reward_loop.reward_manager \
    "+reward.reward_functions.clap.path=$repo_root/verl_omni/utils/reward_score/clap.py" \
    '+reward.reward_functions.clap.name=compute_score' \
    '+reward.reward_functions.clap.weight=1.0' \
    '+reward.reward_functions.clap.required=true' \
    "+reward.reward_functions.clap.device=$REWARD_DEVICE:0" \
    "+reward.reward_functions.clap.model_name_or_path=$CLAP_MODEL_PATH" \
    "+reward.reward_functions.imagebind.path=$repo_root/verl_omni/utils/reward_score/imagebind.py" \
    '+reward.reward_functions.imagebind.name=compute_score' \
    '+reward.reward_functions.imagebind.weight=1.0' \
    '+reward.reward_functions.imagebind.required=true' \
    "+reward.reward_functions.imagebind.device=$REWARD_DEVICE:1" \
    "+reward.reward_functions.imagebind.model_name_or_path=$IMAGEBIND_MODEL_PATH" \
    '+reward.reward_functions.imagebind.mode=audio_video' \
    reward.aggregation=weighted_sum \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=flow_grpo_npu \
    trainer.experiment_name=minimax_h3_t2va_lora_npu \
    trainer.default_local_dir=$checkpoint_dir \
    trainer.validation_data_dir=$output_dir/validation_data \
    trainer.rollout_data_dir=$output_dir/rollout_data \
    trainer.rollout_data_save_freq=10 \
    trainer.log_val_generations=8 \
    trainer.video_fps=24 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS "$@"
