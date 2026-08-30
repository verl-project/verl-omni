# LingBot Dense T2V LoRA FSDP2 RL with HPSv3 reward.
set -x

export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

WORKSPACE=${WORKSPACE:-$(cd "$(dirname "$0")/../../.." && pwd)}
model_name=${MODEL_PATH:-$WORKSPACE/models/lingbot-video-dense-1.3b}
tokenizer_path=${TOKENIZER_PATH:-$model_name/processor}
reward_function_path=${REWARD_FUNCTION_PATH:-pkg://verl_omni.utils.reward_score.hpsv3_reward}

export custom_reward_model_path=${custom_reward_model_path:-$WORKSPACE/models/HPSv3/HPSv3.safetensors}

experiment_name=${EXPERIMENT_NAME:-lingbot_dense_t2v_lora_fsdp2}
train_path=${TRAIN_FILES:-$WORKSPACE/data/lingbot_video/train.parquet}
test_path=${VAL_FILES:-$WORKSPACE/data/lingbot_video/val.parquet}

output_dir=${OUTPUT_DIR:-$WORKSPACE/outputs/$experiment_name}
checkpoint_dir=${CHECKPOINT_DIR:-$output_dir/checkpoints}
run_timestamp=$(date +"%Y%m%d_%H%M")
log_file=${LOG_FILE:-$output_dir/logs/$run_timestamp/${NODE_RANK:-0}.log}
rollout_data_dir=${ROLLOUT_DATA_DIR:-$output_dir/logs/$run_timestamp/rollout_videos}
val_data_dir=${VALIDATION_DATA_DIR:-$output_dir/logs/$run_timestamp/val_videos}
mkdir -p "$checkpoint_dir" "$(dirname "$log_file")"
exec > >(tee -a "$log_file") 2>&1
echo "Logging to $log_file"

python3 -m verl_omni.trainer.main_diffusion \
    algorithm.adv_estimator=${ADV_ESTIMATOR:-flow_grpo} \
    data.train_files=$train_path \
    data.val_files=$test_path \
    data.train_batch_size=${TRAIN_BATCH_SIZE:-16} \
    data.val_batch_size=${VAL_BATCH_SIZE:-16} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH:-37698} \
    data.seed=${DATA_SEED:-42} \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.tokenizer_path=$tokenizer_path \
    actor_rollout_ref.model.algorithm=${MODEL_ALGORITHM:-flow_grpo} \
    actor_rollout_ref.model.enable_gradient_checkpointing=${ENABLE_GRADIENT_CHECKPOINTING:-True} \
    actor_rollout_ref.model.fsdp_layer_prefixes="${FSDP_LAYER_PREFIXES:-[\"blocks.\"]}" \
    actor_rollout_ref.model.lora_rank=${LORA_RANK:-64} \
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA:-128} \
    actor_rollout_ref.model.target_modules="${TARGET_MODULES:-['to_q','to_k','to_v','to_out','gate_proj','up_proj','down_proj']}" \
    actor_rollout_ref.actor.optim.lr=${LR:-1e-5} \
    actor_rollout_ref.actor.optim.weight_decay=${WEIGHT_DECAY:-0.0001} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-8} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2} \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=${LOSS_MODE:-flow_grpo} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD:-True} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD:-True} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=${MODEL_DTYPE:-bfloat16} \
    actor_rollout_ref.actor.strategy=${ACTOR_STRATEGY:-fsdp2} \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${ACTOR_SP:-1} \
    actor_rollout_ref.rollout.name=${ENGINE:-vllm_omni} \
    actor_rollout_ref.rollout.load_format=${LOAD_FORMAT:-safetensors} \
    actor_rollout_ref.rollout.layered_summon=${LAYERED_SUMMON:-True} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP:-2} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4} \
    actor_rollout_ref.rollout.n=${ROLLOUT_GROUP_SIZE:-8} \
    actor_rollout_ref.rollout.seed=${ROLLOUT_SEED:-42} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2} \
    actor_rollout_ref.rollout.agent.num_workers=$((${NUM_GPUS_ACTOR_ROLLOUT_REWARD:-8} / ${ROLLOUT_TP:-2})) \
    actor_rollout_ref.rollout.agent.default_agent_loop=${AGENT_LOOP:-lingbot_dense_t2v_agent} \
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH:-37698} \
    actor_rollout_ref.rollout.pipeline.height=${IMAGE_HEIGHT:-480} \
    actor_rollout_ref.rollout.pipeline.width=${IMAGE_WIDTH:-832} \
    actor_rollout_ref.rollout.pipeline.num_frames=${NUM_FRAMES:-81} \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=${NUM_INFERENCE_STEPS:-10} \
    actor_rollout_ref.rollout.pipeline.guidance_scale=${GUIDANCE_SCALE:-3.0} \
    actor_rollout_ref.rollout.pipeline.shift=${FLOW_SHIFT:-3.0} \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=${MAX_PROMPT_LENGTH:-37698} \
    actor_rollout_ref.rollout.algo.noise_level=${ROLLOUT_NOISE_LEVEL:-0.7} \
    actor_rollout_ref.rollout.algo.sde_type=${ROLLOUT_SDE_TYPE:-dance_sde} \
    actor_rollout_ref.rollout.algo.sde_window_size=${SDE_WINDOW_SIZE:-2} \
    actor_rollout_ref.rollout.algo.sde_window_range="${SDE_WINDOW_RANGE:-[0,5]}" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=${VAL_NUM_INFERENCE_STEPS:-40} \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=${VAL_NOISE_LEVEL:-0.0} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}} \
    reward.num_workers=${REWARD_WORKERS:-1} \
    reward.reward_model.enable=${REWARD_MODEL_ENABLE:-False} \
    reward.custom_reward_function.path=$reward_function_path \
    reward.custom_reward_function.name=${REWARD_FUNCTION_NAME:-compute_score_hpsv3} \
    trainer.logger="${TRAINER_LOGGER:-[\"console\", \"tensorboard\", \"wandb\"]}" \
    trainer.project_name=${PROJECT_NAME:-flow_grpo} \
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir=$checkpoint_dir \
    trainer.rollout_data_dir=$rollout_data_dir \
    trainer.validation_data_dir=$val_data_dir \
    trainer.rollout_data_save_freq=${ROLLOUT_DATA_SAVE_FREQ:-5} \
    trainer.validation_data_max_samples=${VALIDATION_DATA_MAX_SAMPLES:-8} \
    trainer.log_val_generations=${LOG_VAL_GENERATIONS:-8} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-False} \
    trainer.n_gpus_per_node=${NUM_GPUS_ACTOR_ROLLOUT_REWARD:-8} \
    trainer.nnodes=${NNODES:-1} \
    ray_kwargs.ray_init.num_cpus=${NUM_CPUS:-64} \
    +ray_kwargs.ray_init.runtime_env.env_vars.custom_reward_model_path=$custom_reward_model_path \
    trainer.save_freq=${SAVE_FREQ:-30} \
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP:-2} \
    trainer.test_freq=${TEST_FREQ:-30} \
    trainer.total_epochs=${TOTAL_EPOCHS:-1} \
    trainer.resume_mode=${RESUME_MODE:-auto} "$@"
