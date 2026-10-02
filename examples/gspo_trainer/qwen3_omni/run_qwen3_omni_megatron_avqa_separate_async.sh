#!/usr/bin/env bash
# Full-parameter Qwen3-Omni Thinker GSPO on real AVQA image + audio inputs.
# Default layout: four Megatron actor GPUs and four standalone rollout GPUs.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
: "${MODEL_PATH:?Set MODEL_PATH to the Qwen3-Omni checkpoint}"
: "${TRAIN_FILE:?Set TRAIN_FILE to a decontaminated AVQA training parquet}"
: "${VAL_FILE:?Set VAL_FILE to the original AVQA validation parquet}"

ACTOR_GPUS=4
NUM_GPUS=${NUM_GPUS:-8}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
ROLLOUT_TP=${ROLLOUT_TP:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-150}
TEST_FREQ=${TEST_FREQ:-30}
for value in "$NUM_GPUS" "$ROLLOUT_GPUS" "$ROLLOUT_TP" "$MAX_PROMPT_LENGTH" "$MAX_RESPONSE_LENGTH" "$TOTAL_TRAINING_STEPS" "$TEST_FREQ"; do
  [[ $value =~ ^[1-9][0-9]*$ ]] || { echo "Expected positive integer: $value" >&2; exit 2; }
done
(( NUM_GPUS == ACTOR_GPUS + ROLLOUT_GPUS && ROLLOUT_GPUS % ROLLOUT_TP == 0 )) || {
  echo "NUM_GPUS must equal 4 actor GPUs plus ROLLOUT_GPUS, divisible by ROLLOUT_TP" >&2
  exit 2
}

if [[ ${AVQA_CONFIG_ONLY:-0} != 1 ]]; then
  [[ -f $TRAIN_FILE && -f $VAL_FILE ]] || { echo "AVQA train/validation parquet missing" >&2; exit 2; }
  [[ $(realpath "$TRAIN_FILE") != $(realpath "$VAL_FILE") ]] || {
    echo "AVQA train and validation must be different files" >&2; exit 2;
  }
fi

OUTPUT_BASE=${OUTPUT_DIR:-${TMPDIR:-/tmp}/avqa-megatron}
mkdir -p "$OUTPUT_BASE"
RUN_DIR=$(mktemp -d "$OUTPUT_BASE/run.XXXXXX")
echo "AVQA artifacts: $RUN_DIR"

export VERL_USE_EXTERNAL_MODULES=verl_omni
export TENSORBOARD_DIR="$RUN_DIR/tensorboard"
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export PYTHONUNBUFFERED=1
cd "$REPO_ROOT"

# Reuse the shared Megatron configuration with AVQA and model/topology overrides.
args=(
  --config-name omni_megatron_trainer
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "data.train_files=${TRAIN_FILE}"
  "data.val_files=${VAL_FILE}"
  data.custom_cls.path=pkg://verl_omni.utils.dataset.omni_rl_datasets
  data.custom_cls.name=QwenOmniRLHFDataset
  data.train_batch_size=16
  data.filter_overlong_prompts=true
  actor_rollout_ref.model.model_type=omni_model
  actor_rollout_ref.model.use_remove_padding=false
  actor_rollout_ref.actor.ppo_mini_batch_size=16
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.clip_ratio_c=10.0
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.optim.weight_decay=0.1
  # Keep FP32 master parameters on CPU to fit the four-GPU actor.
  '+actor_rollout_ref.actor.optim.override_optimizer_config={optimizer_cpu_offload:true,optimizer_offload_fraction:1.0,overlap_cpu_optimizer_d2h_h2d:false}'
  actor_rollout_ref.actor.megatron.use_remove_padding=false
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4
  actor_rollout_ref.actor.megatron.expert_model_parallel_size=4
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
  actor_rollout_ref.actor.megatron.param_offload=true
  actor_rollout_ref.actor.megatron.optimizer_offload=true
  actor_rollout_ref.actor.megatron.grad_offload=true
  # Match the checkpoint: train/eval log-probs must not differ due to dropout.
  +actor_rollout_ref.actor.megatron.override_transformer_config.attention_dropout=0.0
  +actor_rollout_ref.actor.megatron.override_transformer_config.hidden_dropout=0.0
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_load_balancing_type=none
  +actor_rollout_ref.actor.megatron.override_transformer_config.moe_aux_loss_coeff=0.0
  +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=false
  +actor_rollout_ref.actor.megatron.override_transformer_config.freeze_language_model=false
  +actor_rollout_ref.actor.megatron.override_transformer_config.freeze_vision_model=true
  +actor_rollout_ref.actor.megatron.override_transformer_config.freeze_audio_model=true
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  'actor_rollout_ref.ref.megatron.override_transformer_config={gradient_accumulation_fusion:false,attention_dropout:0.0,hidden_dropout:0.0}'
  actor_rollout_ref.rollout.n=8
  actor_rollout_ref.rollout.nnodes=1
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.enable_prefix_caching=false
  actor_rollout_ref.rollout.logprobs_mode=raw_logprobs
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256
  +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_omni_moe
  +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_mode=thinker_only
  +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.limit_mm_per_prompt.image=1
  +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.limit_mm_per_prompt.video=0
  algorithm.adv_estimator=grpo
  algorithm.rollout_correction.bypass_mode=false
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=verl_omni
  '+ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR=${oc.env:TENSORBOARD_DIR}'
  '+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_DEVICE_MAX_CONNECTIONS=${oc.env:CUDA_DEVICE_MAX_CONNECTIONS,1}'
  '+ray_kwargs.ray_init.runtime_env.env_vars.RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="0"'
  '+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONUNBUFFERED="1"'
  trainer.v1.trainer_mode=omni_separate_async
  trainer.v1.separate_async.parameter_sync_step=1
  trainer.n_gpus_per_node=4
  trainer.resume_mode=disable
  'trainer.logger=[console,tensorboard]'
  trainer.project_name=qwen3_omni_avqa
  trainer.experiment_name=avqa_megatron_separate_async
  "trainer.default_local_dir=${RUN_DIR}/checkpoints"
  "+ray_kwargs.ray_init.num_gpus=$NUM_GPUS"
  "actor_rollout_ref.rollout.n_gpus_per_node=$ROLLOUT_GPUS"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP"
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4
  actor_rollout_ref.rollout.max_num_seqs=16
  actor_rollout_ref.actor.policy_loss.loss_mode=gspo
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
  actor_rollout_ref.actor.clip_ratio_low=0.0003
  actor_rollout_ref.actor.clip_ratio_high=0.0004
  "trainer.total_training_steps=$TOTAL_TRAINING_STEPS"
  "trainer.test_freq=$TEST_FREQ"
  trainer.save_freq=-1
  "trainer.rollout_data_dir=$RUN_DIR/samples/rollouts"
  "trainer.validation_data_dir=$RUN_DIR/samples/validation"
  actor_rollout_ref.actor.optim.use_precision_aware_optimizer=true
  "data.max_prompt_length=$MAX_PROMPT_LENGTH"
  "data.max_response_length=$MAX_RESPONSE_LENGTH"
  "actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
  ++data.mm_processor_kwargs.sampling_rate=16000
  ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.limit_mm_per_prompt.audio=1
  reward.custom_reward_function.path=verl_omni/utils/reward_score/choice_reward.py
  "$@"
)
printf '%q ' "${PYTHON:-python3}" -m verl_omni.trainer.main_omni "${args[@]}" > "${RUN_DIR}/command.txt"
printf '\n' >> "${RUN_DIR}/command.txt"
git rev-parse HEAD > "${RUN_DIR}/commit.txt"
VLLM_LOGGING_STREAM=ext://sys.stderr "${PYTHON:-python3}" -m verl_omni.trainer.main_omni "${args[@]}" \
  --cfg job --resolve > "${RUN_DIR}/config.yaml" 2> >(tee "${RUN_DIR}/config.log" >&2)
# Let CPU regression tests exercise the launcher's exact Hydra composition.
if [[ ${AVQA_CONFIG_ONLY:-0} == 1 ]]; then
  exit 0
fi
"${PYTHON:-python3}" -m verl_omni.trainer.main_omni "${args[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
