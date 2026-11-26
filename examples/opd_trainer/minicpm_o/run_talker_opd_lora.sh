#!/usr/bin/env bash
# MiniCPM-o 4.5 Talker on-policy distillation (OPD): train the TTS AR policy
# over speech codec tokens while the thinker is frozen.
#
# This recipe implements the talker milestone of verl-omni#565. The student's
# rollout runs the thinker→talker pipeline (pipeline_mode="thinker_talker"):
# the frozen thinker writes the text response on-policy and the trainable
# talker speaks it as codec frames. The batch's response tokens are the codec
# frames themselves; the real prompt and the thinker's text response travel
# in the per-sample replay payload (see
# verl_omni/pipelines/minicpm/talker_forward.py).
#
# Prerequisites:
#   - The thinker OPD recipe (PR #858) — its adapter registration, FSDP
#     engine, and weight-sync patterns are reused here.
#   - The teacher must be a MiniCPM-o 4.5 checkpoint sharing the student's
#     codec vocabulary (num_vq=1, num_audio_tokens=6562).
#
# Hyperparameters follow the thinker OPD recipe (which follows the GSPO AVQA
# recipe) with talker-specific stage selection and LoRA targeting.

set -x

export VERL_USE_EXTERNAL_MODULES=verl_omni

STUDENT_MODEL=${STUDENT_MODEL:-"/user/yangzhongyu/models/MiniCPM-o-4_5"}
TEACHER_MODEL=${TEACHER_MODEL:-"/user/yangzhongyu/models/MiniCPM-o-4_5"}
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
TRAIN_FILE=${TRAIN_FILE:-"/user/yangzhongyu/data/talker_opd/train_prompts.parquet"}
VAL_FILE=${VAL_FILE:-"/user/yangzhongyu/data/talker_opd/val_prompts.parquet"}

# The talker OPD teacher requires teacher-forced codec scoring in the vLLM-Omni
# engine (not yet available for MiniCPM-o); the default runs the supported
# GSPO + KL-to-reference smoke scored by talker_codec_reward. Enable the
# distillation group once the engine supports it.
python3 -m verl_omni.trainer.main_omni \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.truncation='error' \
    data.filter_overlong_prompts=false \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.omni_rl_datasets \
    data.custom_cls.name=MiniCPMORLHFDataset \
    ++data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.model_stage="talker" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=false \
    actor_rollout_ref.model.exclude_modules=".*llm.*|.*vpm.*|.*apm.*|.*resampler.*|.*code2wav.*|.*code_predictor.*|.*codec_decoder.*|.*audio_decoder.*|.*audio_generator.*" \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj']" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.prompt_length=1024 \
    actor_rollout_ref.rollout.response_length=2048 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode="ar" \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="minicpmo_4_5" \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_mode="thinker_talker" \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=8 \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_init_timeout=3600 \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.async_chunk=false \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.agent.default_agent_loop=omni_single_turn_agent \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    ++distillation.enabled=false \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    reward.reward_manager.source=register \
    reward.reward_manager.name=minicpm_naive \
    reward.custom_reward_function.path="${REPO_ROOT}/verl_omni/utils/reward_score/talker_codec_reward.py" \
    reward.custom_reward_function.name=compute_score \
    trainer.val_before_train=false \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=talker_opd \
    trainer.experiment_name=minicpm_o_4_5_talker_opd_lora \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=2 \
    "$@"
