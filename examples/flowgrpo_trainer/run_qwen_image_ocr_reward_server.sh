# Qwen-Image full-weight RL with external OCR reward server, vllm_omni rollout
# use examples/flowgrpo_trainer/data_process/qwenimage_ocr_http_service.py to prepare data
set -x

# Set WORKSPACE to any writable directory; defaults to $HOME
WORKSPACE=${WORKSPACE:-$HOME}
export RAY_DEDUP_LOGS=0

ocr_train_path=$WORKSPACE/data/ocr_http/train.parquet
ocr_test_path=$WORKSPACE/data/ocr_http/test.parquet

model_name=${MODEL_PATH:-$HOME/huggingface/Qwen-Image}

# External OCR reward server URL (must be running before training starts)
OCR_REWARD_SERVER_URL=${OCR_REWARD_SERVER_URL:-http://localhost:19082}

NUM_GPUS_ACTOR_ROLLOUT=4
ACTOR_SP=1
ROLLOUT_TP=1
IMAGE_RESOLUTION=512

ENGINE=vllm_omni

# Optional: disable prompt embed cache to free memory at the cost of
# re-running the text encoder for each rollout sample.
ENABLE_PROMPT_EMBED_CACHE=${ENABLE_PROMPT_EMBED_CACHE:-True}

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

output_dir=$repo_root/outputs/$script_name
checkpoint_dir=$output_dir/checkpoints
run_timestamp=$(date +"%Y%m%d_%H%M")
log_file=$output_dir/logs/$run_timestamp/${NODE_RANK:-0}.log
rollout_data_dir=$output_dir/logs/$run_timestamp/rollout_images
mkdir -p "$checkpoint_dir" "$(dirname "$log_file")"
exec > >(tee -a "$log_file") 2>&1
echo "Logging to $log_file"

python3 -m verl_omni.trainer.main_diffusion \
    algorithm.adv_estimator=flow_grpo \
    data.train_files=$ocr_train_path \
    data.val_files=$ocr_test_path \
    data.train_batch_size=32 \
    data.max_prompt_length=256 \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.actor.optim.lr=3e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=$ACTOR_SP \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.enable_prompt_embed_cache=$ENABLE_PROMPT_EMBED_CACHE \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.height=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.width=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.algo.noise_level=1.2 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,5]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=50 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path=pkg://verl_omni.reward_loop.reward_manager.multi \
    reward.custom_reward_function.name=_multi_reward_placeholder \
    reward.reward_manager.name=MultiVisualRewardManager \
    reward.reward_manager.module.path=pkg://verl_omni.reward_loop.reward_manager \
    "+reward.reward_functions.ocr.path=pkg://verl_omni.utils.reward_score.http_scorer_client" \
    '+reward.reward_functions.ocr.name=compute_score' \
    '+reward.reward_functions.ocr.weight=1.0' \
    "+reward.reward_functions.ocr.server_url=$OCR_REWARD_SERVER_URL" \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=qwen_image_ocr_reward_server \
    trainer.default_local_dir=$checkpoint_dir \
    +trainer.rollout_data_dir=$rollout_data_dir \
    trainer.log_val_generations=8 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=300 "$@"
