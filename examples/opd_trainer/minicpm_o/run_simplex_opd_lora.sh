#!/usr/bin/env bash
# MiniCPM-o 4.5 simplex thinker OPD with a separate frozen teacher pool.
set -euo pipefail

export VERL_USE_EXTERNAL_MODULES=verl_omni

: "${STUDENT_MODEL:?Set STUDENT_MODEL to a MiniCPM-o 4.5 checkpoint}"
: "${TEACHER_MODEL:?Set TEACHER_MODEL to a compatible frozen checkpoint}"
: "${DATA_DIR:?Set DATA_DIR to a directory containing train.parquet and test.parquet}"

STUDENT_GPUS=${STUDENT_GPUS:-4}
TEACHER_GPUS=${TEACHER_GPUS:-4}
ROLLOUT_TP=${ROLLOUT_TP:-2}
TEACHER_TP=${TEACHER_TP:-2}
PROMPT_LENGTH=${PROMPT_LENGTH:-1024}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-512}

python3 -m verl_omni.trainer.main_omni \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size=16 \
    data.max_prompt_length="${PROMPT_LENGTH}" \
    data.max_response_length="${RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=false \
    data.truncation=error \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.model.model_stage=thinker \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.lora.merge=true \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj']" \
    actor_rollout_ref.model.exclude_modules='.*vpm.*|.*apm.*|.*resampler.*|.*audio_projection_layer.*' \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.freeze_vision_tower=false \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.enable_prefix_caching=false \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.agent.default_agent_loop=minicpm_simplex_agent \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl_omni.pipelines.minicpm.agent_loop.MiniCPMAgentLoopManager \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=minicpmo_4_5 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.async_chunk=false \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    distillation.enabled=true \
    distillation.nnodes=1 \
    distillation.n_gpus_per_node="${TEACHER_GPUS}" \
    distillation.distillation_loss.loss_mode=kl \
    distillation.distillation_loss.use_policy_gradient=true \
    distillation.distillation_loss.use_task_rewards=false \
    distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}" \
    distillation.teacher_models.teacher_model.inference.name=vllm_omni \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="${TEACHER_TP}" \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.4 \
    distillation.teacher_models.teacher_model.inference.max_num_seqs=8 \
    distillation.teacher_models.teacher_model.inference.enable_prefix_caching=false \
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm_omni.output_mode=ar \
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm_omni.pipeline_name=minicpmo_4_5 \
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm_omni.async_chunk=false \
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm_omni.trust_remote_code=true \
    reward.reward_model.enable=false \
    reward.reward_manager.source=register \
    reward.reward_manager.name=naive \
    reward.num_workers=1 \
    trainer.n_gpus_per_node="${STUDENT_GPUS}" \
    trainer.nnodes=1 \
    trainer.val_before_train=false \
    trainer.logger='[console]' \
    trainer.project_name=minicpm-opd \
    trainer.experiment_name=minicpm-o45-simplex \
    trainer.total_training_steps=100 \
    trainer.save_freq=20 \
    trainer.test_freq=-1 \
    "$@"
