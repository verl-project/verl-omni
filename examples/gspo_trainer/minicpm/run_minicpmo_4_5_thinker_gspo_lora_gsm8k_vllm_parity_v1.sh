#!/usr/bin/env bash
# MiniCPM-o 4.5 Thinker GSPO + LoRA on GSM8K with opt-in train/rollout kernel parity.
#
# Experimental. When VERL_OMNI_VLLM_PARITY=1 the FSDP2 actor's HF Qwen3 text
# decoder runs vLLM batch-invariant kernels (same residual dataflow as the
# eager rollout), log-probs follow the vLLM V1 sampler math, and merged LoRA
# uses one shared merge for actor forward and weight sync.
#
# Required switches (also exported below):
#   VERL_OMNI_VLLM_PARITY=1
#   VLLM_BATCH_INVARIANT=1
#   actor_rollout_ref.rollout.enforce_eager=true
#   actor_rollout_ref.rollout.tensor_model_parallel_size=1
#
# Do NOT use the production AVQA script for this mode (CUDA graphs / TP=2 were
# not measured with parity). Re-verify mismatch counts on current pins before
# treating a run as zero-mismatch; the RSI acceptance numbers were on older
# pins. See docs/algo/minicpm_vllm_parity.md.
#
# Data (once):
#   python <verl>/examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k

set -x

export VERL_USE_EXTERNAL_MODULES=verl_omni
export VERL_OMNI_VLLM_PARITY=1
export VLLM_BATCH_INVARIANT=1

MODEL_PATH=${MODEL_PATH:-"$HOME/models/openbmb/MiniCPM-o-4_5"}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/gsm8k/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/gsm8k/test.parquet"}

python3 -m verl_omni.trainer.main_omni \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.truncation=error \
    data.filter_overlong_prompts=true \
    data.shuffle=false \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.omni_rl_datasets \
    data.custom_cls.name=MiniCPMORLHFDataset \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.exclude_modules=".*vpm.*|.*apm.*|.*talker.*|.*code2wav.*|.*code_predictor.*|.*codec.*|.*audio_decoder.*|.*audio_generator.*|.*audio_head.*|.*tts.*|.*vocoder.*" \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj']" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=minicpmo_4_5 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    reward.reward_manager.source=register \
    reward.reward_manager.name=minicpm_naive \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/choice_reward.py \
    reward.custom_reward_function.name=compute_score \
    trainer.val_before_train=false \
    trainer.balance_batch=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=gspo \
    trainer.experiment_name=minicpm_o_4_5_thinker_lora_gsm8k_vllm_parity \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.resume_mode=disable \
    "$@"
