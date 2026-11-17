#!/usr/bin/env bash
# One-update MiniCPM-o 4.5 understanding smoke test on real AVQA data.
# Eight GPUs: TP4 actor and two standalone TP2 rollout replicas.
# Uses the shared Megatron YAML and existing MiniCPM dataset/rollout/reward path.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the MiniCPM-o-4_5 checkpoint}"
: "${TRAIN_FILE:?Set TRAIN_FILE to prepared AVQA training parquet}"
: "${VAL_FILE:?Set VAL_FILE to prepared AVQA validation parquet}"
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/minicpmo-megatron-smoke}
PYTHON=${PYTHON:-python3}
export VERL_USE_EXTERNAL_MODULES=verl_omni
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_DEVICE_MAX_CONNECTIONS=1

args=(
 --config-name omni_megatron_trainer
 "actor_rollout_ref.model.path=${MODEL_PATH}"
 actor_rollout_ref.model.trust_remote_code=true
 actor_rollout_ref.model.model_type=omni_model
 actor_rollout_ref.model.use_remove_padding=false
 "data.train_files=${TRAIN_FILE}"
 "data.val_files=${VAL_FILE}"
 data.train_max_samples=8 data.val_max_samples=4
 data.dataloader_num_workers=0
 data.custom_cls.path=pkg://verl_omni.utils.dataset.omni_rl_datasets
 data.custom_cls.name=MiniCPMORLHFDataset
 ++data.mm_processor_kwargs.sampling_rate=16000
 ++data.apply_chat_template_kwargs.enable_thinking=false
 data.train_batch_size=4 data.max_prompt_length=2048 data.max_response_length=128
 data.filter_overlong_prompts=true data.filter_overlong_prompts_workers=null data.truncation=error
 actor_rollout_ref.actor.ppo_mini_batch_size=4
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
 actor_rollout_ref.actor.use_dynamic_bsz=false
 actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.optim.weight_decay=0.0
 actor_rollout_ref.actor.optim.lr_decay_style=constant
 actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4
 actor_rollout_ref.actor.megatron.expert_model_parallel_size=1
 actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
 actor_rollout_ref.actor.megatron.sequence_parallel=false
 actor_rollout_ref.actor.megatron.use_distributed_optimizer=true
 actor_rollout_ref.actor.megatron.use_remove_padding=false
 actor_rollout_ref.actor.megatron.param_offload=false
 actor_rollout_ref.actor.megatron.optimizer_offload=false
 actor_rollout_ref.actor.megatron.grad_offload=false
 '++actor_rollout_ref.actor.megatron.override_ddp_config={overlap_grad_reduce:false}'
 '++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=unfused'
 '++actor_rollout_ref.actor.megatron.override_transformer_config.attention_dropout=0.0'
 '++actor_rollout_ref.actor.megatron.override_transformer_config.hidden_dropout=0.0'
 '++actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=false'
 '++actor_rollout_ref.actor.megatron.override_transformer_config.freeze_language_model=false'
 '++actor_rollout_ref.actor.megatron.override_transformer_config.freeze_vision_model=true'
 '++actor_rollout_ref.actor.megatron.override_transformer_config.freeze_audio_model=true'
 '++actor_rollout_ref.actor.megatron.override_transformer_config.freeze_vision_projection=true'
 '++actor_rollout_ref.actor.megatron.override_transformer_config.freeze_audio_projection=true'
 actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=null
 actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=null
 actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=null
 actor_rollout_ref.actor.policy_loss.loss_mode=gspo
 actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
 actor_rollout_ref.actor.clip_ratio_low=0.0003 actor_rollout_ref.actor.clip_ratio_high=0.0004
 actor_rollout_ref.actor.clip_ratio_c=10.0 actor_rollout_ref.actor.use_kl_loss=false
 actor_rollout_ref.rollout.n=2 actor_rollout_ref.rollout.nnodes=1
 actor_rollout_ref.rollout.n_gpus_per_node=4 actor_rollout_ref.rollout.tensor_model_parallel_size=2
 actor_rollout_ref.rollout.load_format=safetensors actor_rollout_ref.rollout.enable_prefix_caching=false
 actor_rollout_ref.rollout.logprobs_mode=raw_logprobs
 actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=false
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
 actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 actor_rollout_ref.rollout.max_num_seqs=4
 actor_rollout_ref.rollout.max_model_len=2176 actor_rollout_ref.rollout.enforce_eager=true
 ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=minicpmo_4_5
 ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.mm_processor_cache_gb=0
 ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.limit_mm_per_prompt.image=1
 ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.limit_mm_per_prompt.audio=1
 ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.limit_mm_per_prompt.video=0
 actor_rollout_ref.rollout.val_kwargs.n=1 actor_rollout_ref.rollout.val_kwargs.temperature=0.0
 algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=false
 algorithm.rollout_correction.bypass_mode=false
 reward.reward_manager.source=register reward.reward_manager.name=minicpm_naive
 reward.custom_reward_function.path=verl_omni/utils/reward_score/choice_reward.py
 reward.custom_reward_function.name=compute_score
 trainer.v1.trainer_mode=omni_separate_async
 trainer.v1.separate_async.parameter_sync_step=1
 trainer.n_gpus_per_node=4 trainer.nnodes=1 trainer.resume_mode=disable
 'trainer.logger=[console,tensorboard]'
 trainer.project_name=minicpmo_megatron trainer.experiment_name=minicpmo_megatron_smoke
 trainer.total_training_steps=1 trainer.total_epochs=1 trainer.test_freq=1 trainer.val_before_train=true trainer.save_freq=-1
 "trainer.default_local_dir=${OUTPUT_DIR}/checkpoints"
 "trainer.rollout_data_dir=${OUTPUT_DIR}/rollouts" "trainer.validation_data_dir=${OUTPUT_DIR}/validation"
 +ray_kwargs.ray_init.num_gpus=8
 '++ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=verl_omni'
 '++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_WORKER_MULTIPROC_METHOD=spawn'
)
exec "$PYTHON" -m verl_omni.trainer.main_omni "${args[@]}" "$@"
