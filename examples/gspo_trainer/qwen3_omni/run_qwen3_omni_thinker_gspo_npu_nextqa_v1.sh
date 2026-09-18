#!/usr/bin/env bash
# Full-depth Thinker GSPO training on NExT-QA, with all Thinker parameters trainable.
# Source CANN/ATB first. MODEL_PATH must point to the original full checkpoint.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-16}
export NNODES=${NNODES:-1}
export ROLLOUT_TP=${ROLLOUT_TP:-4}
export MODEL_PATH=${MODEL_PATH:-"/models/Qwen3-Omni-30B-A3B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"/datasets/NextQA/train.parquet"}
VAL_FILE=${VAL_FILE:-"/datasets/NextQA/validation.parquet"}

for name in N_GPUS_PER_NODE NNODES ROLLOUT_TP; do
    if [[ ! ${!name} =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer, got '${!name}'." >&2
        exit 2
    fi
done
WORLD_SIZE_NPU=$((N_GPUS_PER_NODE * NNODES))
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-$((WORLD_SIZE_NPU * 2))}
ROLLOUT_N=${ROLLOUT_N:-8}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-500}
for name in TRAIN_BATCH_SIZE ROLLOUT_N TOTAL_TRAINING_STEPS; do
    if [[ ! ${!name} =~ ^[1-9][0-9]*$ ]]; then
        echo "$name must be a positive integer, got '${!name}'." >&2
        exit 2
    fi
done
if (( TRAIN_BATCH_SIZE % WORLD_SIZE_NPU != 0 || ROLLOUT_N < 2 )); then
    echo "TRAIN_BATCH_SIZE must be divisible by the NPU count; GRPO needs ROLLOUT_N >= 2." >&2
    exit 2
fi
LEARNING_RATE=${LEARNING_RATE:-1e-6}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_omni_nextqa_fullparam}
OUTPUT_DIR=${OUTPUT_DIR:-"${SCRIPT_DIR}/../../../outputs/${EXPERIMENT_NAME}"}

# Defaults target 16 x 64 GB NPUs. Measure full-depth peaks before increasing
# concurrency; total batch controls accumulation, not simultaneous requests.
# Example: MODEL_PATH=/models/Qwen3-Omni-30B-A3B-Instruct OUTPUT_DIR=/checkpoints/nextqa bash <this-script>
# TRAIN_FILE/VAL_FILE use converter outputs. Source CANN/ATB and expose ffmpeg.
ROLLOUT_MEMORY_FRACTION=${ROLLOUT_MEMORY_FRACTION:-0.65}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-4}
if [[ ! $ROLLOUT_MAX_NUM_SEQS =~ ^[1-9][0-9]*$ ]] || (( ROLLOUT_MAX_NUM_SEQS > 8 )); then
    echo "ROLLOUT_MAX_NUM_SEQS must be between 1 and 8." >&2
    exit 2
fi
case "$ROLLOUT_MEMORY_FRACTION" in
    0.[1-9]|0.[1-9][0-9]) ;;
    *) echo "ROLLOUT_MEMORY_FRACTION must be a decimal from 0.1 to 0.99." >&2; exit 2 ;;
esac
GRAPH_CAPTURE_SIZES='[1'
for size in 2 4 8; do
    if (( size <= ROLLOUT_MAX_NUM_SEQS )); then GRAPH_CAPTURE_SIZES+=",$size"; fi
done
GRAPH_CAPTURE_SIZES+=']'
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
cd "$REPO_ROOT"
export VERL_USE_EXTERNAL_MODULES=verl_omni
export VLLM_ASCEND_ENABLE_NZ=0
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false

if (( N_GPUS_PER_NODE % ROLLOUT_TP != 0 )); then
    echo "N_GPUS_PER_NODE must be divisible by ROLLOUT_TP." >&2
    exit 2
fi

exec python3 -m verl_omni.trainer.main_omni \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.nextqa_rl_dataset \
    data.custom_cls.name=NextQARLHFDataset \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.train_max_samples=-1 \
    data.val_max_samples=-1 \
    data.max_prompt_length=8192 \
    data.max_response_length=1024 \
    data.truncation=error \
    data.filter_overlong_prompts=true \
    data.filter_overlong_prompts_workers=128 \
    data.seed=42 \
    data.shuffle=true \
    data.validation_shuffle=false \
    ++data.mm_processor_kwargs.use_audio_in_video=false \
    ++data.mm_processor_kwargs.sampling_rate=16000 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.hf_config_path="${MODEL_PATH}" \
    actor_rollout_ref.model.tokenizer_path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.lora.merge=false \
    actor_rollout_ref.model.target_modules=null \
    actor_rollout_ref.model.exclude_modules=null \
    actor_rollout_ref.actor.freeze_vision_tower=false \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=false \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=true \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=false \
    actor_rollout_ref.actor.fsdp_config.use_no_sync_for_gradient_accumulation=false \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.agent.num_workers=$((N_GPUS_PER_NODE * NNODES / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_omni_moe \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.additional_config.weight_nz_mode=0 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_MEMORY_FRACTION}" \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.cudagraph_capture_sizes="${GRAPH_CAPTURE_SIZES}" \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    actor_rollout_ref.rollout.free_cache_engine=true \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.do_sample=true \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.logprobs_mode=raw_logprobs \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    algorithm.rollout_correction.bypass_mode=false \
    algorithm.rollout_correction.rollout_is=null \
    algorithm.rollout_correction.rollout_rs=null \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    reward.custom_reward_function.path="${REPO_ROOT}/verl_omni/utils/reward_score/choice_reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.device=npu \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.total_epochs=10 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.val_before_train=true \
    trainer.test_freq=10 \
    trainer.save_freq=20 \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.default_local_dir="${OUTPUT_DIR}/checkpoints" \
    trainer.validation_data_dir="${OUTPUT_DIR}/validation" \
    trainer.log_val_generations=8 \
    trainer.project_name=qwen3_omni_nextqa \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger='["console","tensorboard"]' \
    "$@" \
    2>&1 | tee run_qwen3omni_npu_nextqa_full_ms_16.log
