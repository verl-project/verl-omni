#!/usr/bin/env bash
# MiniMax-H3 text-to-audio-video (t2va) DiffusionNFT LoRA recipe.
set -x

export WANDB_MODE=${WANDB_MODE:-offline}

WORKSPACE=${WORKSPACE:-$HOME}
MODEL_PATH=${MODEL_PATH:-}
DATA_DIR=${DATA_DIR:-$WORKSPACE/data/vid_prompt/verl_omni}
NUM_GPUS=${NUM_GPUS:-8}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_N=${ROLLOUT_N:-16}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1000}
HEIGHT=${HEIGHT:-256}
WIDTH=${WIDTH:-384}
NUM_FRAMES=${NUM_FRAMES:-121}
INFER_STEPS=${INFER_STEPS:-10}
VAL_HEIGHT=${VAL_HEIGHT:-512}
VAL_WIDTH=${VAL_WIDTH:-768}
ACTOR_ATTN_BACKEND=${ACTOR_ATTN_BACKEND:-_flash_3_varlen_hub}
ROLLOUT_ATTN_BACKEND=${ROLLOUT_ATTN_BACKEND:-FLASH_ATTN_3_HUB}

if [[ -z "$MODEL_PATH" || ! -d "$MODEL_PATH/FL2VA" || ! -d "$MODEL_PATH/transformer" ]]; then
    echo "MODEL_PATH must point to a local MiniMax-H3 repo root containing FL2VA/ and transformer/ (got: '${MODEL_PATH:-<unset>}')" >&2
    exit 1
fi

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

export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

h3_lora_targets="['to_q','to_k','to_v','to_out.0','ff.net.0.proj','ff.net.2']"

lora_warmstart_arg=()
if [[ -n "${LORA_WARMSTART_PATH:-}" ]]; then
    lora_warmstart_arg=(actor_rollout_ref.model.lora_adapter_path=$LORA_WARMSTART_PATH)
fi

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=$train_path \
    data.val_files=$test_path \
    data.train_batch_size=32 \
    data.val_max_samples=128 \
    data.max_prompt_length=1024 \
    data.truncation=error \
    data.seed=42 \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    algorithm.timestep_fraction=1.0 \
    algorithm.old_policy_decay_schedule=delayed_linear_to_0_999 \
    algorithm.old_policy_update_interval=2 \
    algorithm.adv_mode=continuous \
    actor_rollout_ref.model.path=$MODEL_PATH/FL2VA \
    actor_rollout_ref.model.config_path=$MODEL_PATH/transformer \
    +actor_rollout_ref.model.architecture=MiniMaxH3Pipeline \
    actor_rollout_ref.model.algorithm=diffusion_nft \
    actor_rollout_ref.model.model_type=diffusion_nft_model \
    actor_rollout_ref.model.attn_backend=$ACTOR_ATTN_BACKEND \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    "${lora_warmstart_arg[@]}" \
    actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
    actor_rollout_ref.model.target_modules="$h3_lora_targets" \
    actor_rollout_ref.model.fsdp_layer_prefixes="['transformer_blocks.','token_refiner.refiner_blocks.']" \
    '+actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[MiniMaxH3TransformerBlock,MiniMaxH3TokenRefinerBlock]' \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=1e-4 \
    actor_rollout_ref.actor.optim.betas="[0.9,0.999]" \
    actor_rollout_ref.actor.optim.override_optimizer_config="{eps: 1e-8}" \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=diffusion_nft \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.actor.diffusion_loss.mix_beta=0.1 \
    actor_rollout_ref.actor.diffusion_loss.ref_kl_coef=0.0001 \
    actor_rollout_ref.actor.diffusion_loss.adv_clip_max=5.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.max_num_seqs=1 \
    actor_rollout_ref.rollout.rollout_attn_backend=$ROLLOUT_ATTN_BACKEND \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.agent.default_agent_loop=minimax_h3_diffusion_single_turn_agent \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.rollout_adapter=old \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.pipeline.aspect_ratio=16:9 \
    actor_rollout_ref.rollout.pipeline.height=$HEIGHT \
    actor_rollout_ref.rollout.pipeline.width=$WIDTH \
    actor_rollout_ref.rollout.pipeline.num_frames=$NUM_FRAMES \
    actor_rollout_ref.rollout.pipeline.frame_rate=24.0 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=$INFER_STEPS \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=1024 \
    actor_rollout_ref.rollout.pipeline.video_flow_shift=12.0 \
    +actor_rollout_ref.rollout.pipeline.output_type=pt \
    actor_rollout_ref.rollout.val_kwargs.pipeline.height=$VAL_HEIGHT \
    actor_rollout_ref.rollout.val_kwargs.pipeline.width=$VAL_WIDTH \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_frames=$NUM_FRAMES \
    actor_rollout_ref.rollout.val_kwargs.pipeline.frame_rate=24.0 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=40 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.true_cfg_scale=1.0 \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.output_type=pt \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    reward.num_workers=1 \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path=pkg://verl_omni.reward_loop.reward_manager.multi \
    reward.custom_reward_function.name=_multi_reward_placeholder \
    reward.reward_manager.name=MultiVisualRewardManager \
    reward.reward_manager.module.path=pkg://verl_omni.reward_loop.reward_manager \
    "+reward.reward_functions.clap.path=$repo_root/verl_omni/utils/reward_score/clap.py" \
    '+reward.reward_functions.clap.name=compute_score' \
    '+reward.reward_functions.clap.weight=1.0' \
    '+reward.reward_functions.clap.device=cuda:0' \
    '+reward.reward_functions.clap.model_name_or_path=laion/larger_clap_general' \
    "+reward.reward_functions.imagebind.path=$repo_root/verl_omni/utils/reward_score/imagebind.py" \
    '+reward.reward_functions.imagebind.name=compute_score' \
    '+reward.reward_functions.imagebind.weight=1.0' \
    '+reward.reward_functions.imagebind.device=cuda:1' \
    '+reward.reward_functions.imagebind.model_name_or_path=.checkpoints/imagebind_huge.pth' \
    '+reward.reward_functions.imagebind.mode=audio_video' \
    reward.aggregation=weighted_sum \
    trainer.logger='["console","tensorboard","wandb"]' \
    trainer.project_name=diffusion_nft \
    trainer.experiment_name=minimax_h3_t2va_lora \
    trainer.default_local_dir=$checkpoint_dir \
    trainer.validation_data_dir=$output_dir/validation_data \
    trainer.rollout_data_dir=$output_dir/rollout_data \
    trainer.rollout_data_save_freq=10 \
    trainer.log_val_generations=8 \
    trainer.video_fps=24 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS "$@"
