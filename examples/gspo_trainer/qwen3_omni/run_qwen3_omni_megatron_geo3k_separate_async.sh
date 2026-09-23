#!/usr/bin/env bash
# Full-parameter Thinker language-model Geo3K RL; vision/audio towers stay frozen.
# Default: one 8-GPU node, four Megatron actor GPUs and four rollout GPUs.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
: "${MODEL_PATH:?Set MODEL_PATH to a Qwen3-Omni checkpoint}"
: "${TRAIN_FILE:?Set TRAIN_FILE to prepared Geo3K train.parquet}"
: "${VAL_FILE:?Set VAL_FILE to prepared Geo3K validation.parquet}"
OUTPUT_DIR=${OUTPUT_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/geo3k-run.XXXXXX")}
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR=$(cd "${OUTPUT_DIR}" && pwd)
# Each invocation gets a separate log and resolved configuration.
RUN_DIR=$(mktemp -d "${OUTPUT_DIR}/run.XXXXXX")
echo "Geo3K artifacts: ${RUN_DIR}"
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
  data.train_batch_size=16
  data.max_prompt_length=1024
  data.max_response_length=2048
  data.filter_overlong_prompts=true
  actor_rollout_ref.model.model_type=omni_model
  actor_rollout_ref.model.use_remove_padding=false
  actor_rollout_ref.actor.ppo_mini_batch_size=16
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_dynamic_bsz=false
  actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  actor_rollout_ref.actor.clip_ratio_low=0.2
  actor_rollout_ref.actor.clip_ratio_high=0.2
  actor_rollout_ref.actor.clip_ratio_c=10.0
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.optim.lr_warmup_steps=10
  actor_rollout_ref.actor.use_kl_loss=true
  actor_rollout_ref.actor.kl_loss_coef=0.001
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.optim.weight_decay=0.1
  # Keep half of FP32 Adam states/updates on CPU to fit the four-GPU actor.
  '+actor_rollout_ref.actor.optim.override_optimizer_config={optimizer_cpu_offload:true,optimizer_offload_fraction:0.5,overlap_cpu_optimizer_d2h_h2d:false}'
  actor_rollout_ref.actor.megatron.use_remove_padding=false
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
  actor_rollout_ref.actor.megatron.context_parallel_size=1
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
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=false
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  'actor_rollout_ref.ref.megatron.override_transformer_config={gradient_accumulation_fusion:false,attention_dropout:0.0,hidden_dropout:0.0}'
  actor_rollout_ref.rollout.n=8
  actor_rollout_ref.rollout.temperature=0.8
  actor_rollout_ref.rollout.top_p=0.9
  actor_rollout_ref.rollout.val_kwargs.temperature=0
  actor_rollout_ref.rollout.val_kwargs.n=1
  actor_rollout_ref.rollout.nnodes=1
  actor_rollout_ref.rollout.n_gpus_per_node=4
  actor_rollout_ref.rollout.tensor_model_parallel_size=4
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.max_model_len=3072
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
  +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.limit_mm_per_prompt.audio=0
  +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.limit_mm_per_prompt.image=1
  +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.limit_mm_per_prompt.video=0
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=false
  algorithm.rollout_correction.bypass_mode=false
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=verl_omni
  '+ray_kwargs.ray_init.runtime_env.env_vars.TENSORBOARD_DIR=${oc.env:TENSORBOARD_DIR}'
  '+ray_kwargs.ray_init.runtime_env.env_vars.CUDA_DEVICE_MAX_CONNECTIONS=${oc.env:CUDA_DEVICE_MAX_CONNECTIONS,1}'
  '+ray_kwargs.ray_init.runtime_env.env_vars.RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO="0"'
  '+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONUNBUFFERED="1"'
  trainer.v1.trainer_mode=omni_separate_async
  trainer.v1.separate_async.parameter_sync_step=1
  trainer.nnodes=1
  trainer.n_gpus_per_node=4
  trainer.total_training_steps=30
  trainer.val_before_train=true
  trainer.save_freq=-1
  trainer.test_freq=10
  trainer.resume_mode=disable
  'trainer.logger=[console,tensorboard]'
  trainer.project_name=qwen3_omni_geo3k
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
if [[ ${GEO3K_CONFIG_ONLY:-0} == 1 ]]; then
  exit 0
fi
"${PYTHON:-python3}" -m verl_omni.trainer.main_omni "${args[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
