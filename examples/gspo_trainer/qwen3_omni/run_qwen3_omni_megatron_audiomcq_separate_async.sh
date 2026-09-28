#!/usr/bin/env bash
# Run on the head of an allocated Ray cluster; resource counts are Hydra overrides.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
: "${MODEL_PATH:?Set MODEL_PATH to a Qwen3-Omni checkpoint}"
: "${TRAIN_FILE:?Set TRAIN_FILE to prepared AudioMCQ train.parquet}"
: "${VAL_FILE:?Set VAL_FILE to prepared AudioMCQ validation.parquet}"
OUTPUT_DIR=${OUTPUT_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/audiomcq-run.XXXXXX")}
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR=$(cd "${OUTPUT_DIR}" && pwd)
# Each invocation gets a separate log and resolved configuration.
RUN_DIR=$(mktemp -d "${OUTPUT_DIR}/run.XXXXXX")
echo "AudioMCQ artifacts: ${RUN_DIR}"
export VERL_USE_EXTERNAL_MODULES=verl_omni
export TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export PYTHONUNBUFFERED=1
cd "${REPO_ROOT}"

args=(
  --config-name omni_megatron_trainer
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "data.train_files=${TRAIN_FILE}"
  "data.val_files=${VAL_FILE}"
  data.custom_cls.path=pkg://verl_omni.utils.dataset.omni_rl_datasets
  data.custom_cls.name=QwenOmniRLHFDataset
  +data.mm_processor_kwargs.sampling_rate=16000
  data.train_batch_size=16
  data.max_prompt_length=768
  data.max_response_length=256
  data.filter_overlong_prompts=true
  actor_rollout_ref.model.model_type=omni_model
  actor_rollout_ref.model.use_remove_padding=false
  actor_rollout_ref.actor.ppo_mini_batch_size=16
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_dynamic_bsz=false
  actor_rollout_ref.actor.policy_loss.loss_mode=gspo
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
  actor_rollout_ref.actor.clip_ratio_low=0.0003
  actor_rollout_ref.actor.clip_ratio_high=0.0004
  actor_rollout_ref.actor.clip_ratio_c=10.0
  actor_rollout_ref.actor.optim.weight_decay=0.1
  actor_rollout_ref.actor.megatron.use_remove_padding=false
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
  actor_rollout_ref.actor.megatron.context_parallel_size=1
  actor_rollout_ref.actor.megatron.expert_model_parallel_size=4
  actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
  actor_rollout_ref.actor.megatron.param_offload=true
  actor_rollout_ref.actor.megatron.optimizer_offload=true
  actor_rollout_ref.actor.megatron.grad_offload=true
  +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=false
  +actor_rollout_ref.actor.megatron.override_transformer_config.freeze_language_model=false
  +actor_rollout_ref.actor.megatron.override_transformer_config.freeze_vision_model=true
  +actor_rollout_ref.actor.megatron.override_transformer_config.freeze_audio_model=true
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
  actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  'actor_rollout_ref.ref.megatron.override_transformer_config={gradient_accumulation_fusion:false}'
  actor_rollout_ref.rollout.n=8
  actor_rollout_ref.rollout.nnodes=4
  actor_rollout_ref.rollout.n_gpus_per_node=4
  actor_rollout_ref.rollout.tensor_model_parallel_size=4
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.max_model_len=1024
  actor_rollout_ref.rollout.max_num_seqs=32
  actor_rollout_ref.rollout.max_num_batched_tokens=4096
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6
  actor_rollout_ref.rollout.enable_prefix_caching=false
  actor_rollout_ref.rollout.logprobs_mode=raw_logprobs
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256
  +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_omni_moe
  +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_mode=thinker_only
  +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.limit_mm_per_prompt.audio=1
  +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.limit_mm_per_prompt.image=1
  +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.limit_mm_per_prompt.video=0
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=false
  algorithm.rollout_correction.bypass_mode=false
  reward.custom_reward_function.path=pkg://verl_omni.utils.reward_score.audio_mcq
  reward.custom_reward_function.name=compute_score
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=verl_omni
  '+ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR=${oc.env:TENSORBOARD_DIR}'
  '+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_DEVICE_MAX_CONNECTIONS=${oc.env:CUDA_DEVICE_MAX_CONNECTIONS,1}'
  '+ray_kwargs.ray_init.runtime_env.env_vars.RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="0"'
  '+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONUNBUFFERED="1"'
  trainer.v1.trainer_mode=omni_separate_async
  trainer.v1.separate_async.parameter_sync_step=1
  trainer.nnodes=4
  trainer.n_gpus_per_node=4
  trainer.total_training_steps=150
  trainer.test_freq=10
  trainer.resume_mode=disable
  'trainer.logger=[console,tensorboard]'
  trainer.project_name=qwen3_omni_audiomcq
  trainer.experiment_name=megatron_separate_async
  "trainer.default_local_dir=${RUN_DIR}/checkpoints"
  "$@"
)
printf '%q ' "${PYTHON:-python3}" -m verl_omni.trainer.main_omni "${args[@]}" > "${RUN_DIR}/command.txt"
printf '\n' >> "${RUN_DIR}/command.txt"
git rev-parse HEAD > "${RUN_DIR}/commit.txt"
VLLM_LOGGING_STREAM=ext://sys.stderr "${PYTHON:-python3}" -m verl_omni.trainer.main_omni "${args[@]}" \
  --cfg job --resolve > "${RUN_DIR}/config.yaml" 2> >(tee "${RUN_DIR}/config.log" >&2)
# Let CPU regression tests exercise the launcher's exact Hydra composition.
if [[ ${AUDIO_MCQ_CONFIG_ONLY:-0} == 1 ]]; then
  exit 0
fi
"${PYTHON:-python3}" -m verl_omni.trainer.main_omni "${args[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
