#!/usr/bin/env bash
# Qwen-Image-Edit-2511 LoRA RL with engine and native PickScore rewards.
set -x

# Run the engine and native PickScore models on disjoint NPU subpools.
export VLLM_ASCEND_ENABLE_NZ=0
export VERL_DATAPROTO_SERIALIZATION_METHOD=numpy
model_name=${MODEL_PATH:-Qwen/Qwen-Image-Edit-2511}
pickscore_model_path=${PICKSCORE_MODEL_PATH:-yuvalkirstain/PickScore_v1}

NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS_ACTOR_ROLLOUT_REWARD:-16}
ROLLOUT_TP=${ROLLOUT_TP:-4}
ENGINE_REWARD_NPUS=${ENGINE_REWARD_NPUS:-8}
# Native-pool bundle indices. These are relative to the native subpool, not
# physical NPU IDs or tensor-parallel ranks.
NATIVE_REWARD_DEVICES=${NATIVE_REWARD_DEVICES:-"[0,1,2,3,4,5,6,7]"}
REWARD_OFFLOAD=${REWARD_OFFLOAD:-true}
PICKSCORE_LOGIT_SCALE=${PICKSCORE_LOGIT_SCALE:-98.86447}
IMAGE_RESOLUTION=${IMAGE_RESOLUTION:-512}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}

ENGINE=vllm_omni

WORKSPACE=${WORKSPACE:-$(cd "$(dirname "$0")/../.." && pwd)}
train_path=${TRAIN_FILES:-$WORKSPACE/data/qwen_image_edit/train.parquet}
test_path=${VAL_FILES:-$WORKSPACE/data/qwen_image_edit/test.parquet}

output_dir=$WORKSPACE/outputs/qwen_image_edit_lora
checkpoint_dir=$output_dir/checkpoints
run_timestamp=$(date +"%Y%m%d_%H%M")
log_file=$output_dir/logs/$run_timestamp/${NODE_RANK:-0}.log
rollout_data_dir=$output_dir/logs/$run_timestamp/rollout_images
val_data_dir=$output_dir/logs/$run_timestamp/val_images
mkdir -p "$checkpoint_dir" "$(dirname "$log_file")"
exec > >(tee -a "$log_file") 2>&1
echo "Logging to $log_file"

python3 -m verl_omni.trainer.main_diffusion_v1 \
    data.train_files=$train_path \
    data.val_files=$test_path \
    data.train_batch_size=32 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.seed=42 \
    actor_rollout_ref.model.algorithm=flow_grpo \
    algorithm.global_std=false \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out','img_mlp.net.0.proj','img_mlp.net.2','txt_mlp.net.0.proj','txt_mlp.net.2']" \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=0.0001 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.max_num_seqs=1 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.model.attn_backend='_native_npu' \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    trainer.device=npu \
    trainer.use_v1=True \
    actor_rollout_ref.rollout.cudagraph_capture_sizes="[1,2,4,8,16,32,64,128,256,384,512,640,768,896,1024]" \
    actor_rollout_ref.rollout.enforce_eager=False \
    reward.reward_model.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.mm_processor_cache_gb=0 \
    actor_rollout_ref.rollout.prompt_length=$MAX_PROMPT_LENGTH \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=12 \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=4.0 \
    actor_rollout_ref.rollout.pipeline.height=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.width=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=$MAX_PROMPT_LENGTH \
    actor_rollout_ref.rollout.algo.noise_level=0.7 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=3 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,6]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=40 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    reward.reward_model.enable=False \
    reward.reward_model.enable_resource_pool=False \
    +reward.models.engine_model.backend=engine \
    +reward.models.engine_model.offload=$REWARD_OFFLOAD \
    +reward.models.engine_model.model_path=$pickscore_model_path \
    +reward.models.engine_model.n_gpus_per_node=$ENGINE_REWARD_NPUS \
    +reward.models.engine_model.nnodes=1 \
    +reward.models.engine_model.rollout.name=vllm \
    +reward.models.engine_model.rollout.dtype=bfloat16 \
    +reward.models.engine_model.rollout.tensor_model_parallel_size=1 \
    +reward.models.engine_model.rollout.data_parallel_size=1 \
    +reward.models.engine_model.rollout.pipeline_model_parallel_size=1 \
    +reward.models.engine_model.rollout.max_model_len=77 \
    +reward.models.engine_model.rollout.max_num_seqs=8 \
    +reward.models.engine_model.rollout.limit_images=1 \
    +reward.models.engine_model.rollout.enforce_eager=True \
    +reward.models.engine_model.rollout.engine_kwargs.vllm.runner=pooling \
    +reward.models.native_model.backend=native \
    +reward.models.native_model.offload=$REWARD_OFFLOAD \
    +reward.models.native_model.model_path=$pickscore_model_path \
    +reward.models.native_model.placement.devices="$NATIVE_REWARD_DEVICES" \
    +reward.models.native_model.executor.model=verl_omni.utils.reward_score.pickscore_reward:PickScoreNativeModel \
    +reward.reward_functions.engine_model.path=pkg://verl_omni.utils.reward_score.pickscore_reward \
    +reward.reward_functions.engine_model.name=compute_score_pickscore_engine \
    +reward.reward_functions.engine_model.logit_scale=$PICKSCORE_LOGIT_SCALE \
    +reward.reward_functions.engine_model.weight=0.5 \
    +reward.reward_functions.native_model.path=pkg://verl_omni.utils.reward_score.pickscore_reward \
    +reward.reward_functions.native_model.name=compute_score_pickscore_native \
    +reward.reward_functions.native_model.weight=0.5 \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=qwen_image_edit_lora_pickscore \
    trainer.default_local_dir=$checkpoint_dir \
    +trainer.rollout_data_dir=$rollout_data_dir \
    +trainer.validation_data_dir=$val_data_dir \
    trainer.log_val_generations=8 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=300 \
    trainer.total_epochs=100 \
    trainer.resume_mode=auto "$@"
