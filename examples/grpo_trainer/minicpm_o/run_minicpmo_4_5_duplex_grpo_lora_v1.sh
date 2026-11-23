#!/usr/bin/env bash
# MiniCPM-o 4.5 Duplex GRPO + LoRA training on DailyTalkContiguous event segments
# with the omni V1 trainer (verl-omni#905 §3.8).
#
# This recipe is the duplex sibling of the proven thinker GSPO AVQA recipe
# (examples/gspo_trainer/minicpm/run_minicpmo_4_5_thinker_gspo_lora_avqa_v1.sh).
# The pipeline_mode, dataset, and reward differ; hyperparameters follow the
# RFC's §3.5 defaults. Requires the duplex rollout bridge and engine action
# capture (vllm-project/vllm-omni#10536) to be landed before the first run.
#
# Data preparation (mine DailyTalkContiguous events):
#   python verl_omni/utils/dataset/duplex/event_miner.py \
#       --input_dir <path_to_DailyTalkContiguous> \
#       --output_dir ~/data/dailytalk_events
#
# Hyperparameters from verl-omni#905 §3.5:
#   group size ≈ 16, clipping ε = 0.2, text-token loss, KL on (small β).
#   LoRA + frozen audio/vision towers (validated recipe pattern).
#   Reward-calibration entry gate: run the base checkpoint first
#   (see verl_omni/utils/dataset/duplex/calibration.py).

set -x

# Make verl_omni available to Ray workers
export VERL_USE_EXTERNAL_MODULES=verl_omni

MODEL_PATH=${MODEL_PATH:-"$HOME/models/openbmb/MiniCPM-o-4_5"}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/dailytalk_events/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/dailytalk_events/validation.parquet"}

python3 -m verl_omni.trainer.main_omni \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.duplex \
    data.custom_cls.name=DuplexEventDataset \
    +data.mm_processor_kwargs.sampling_rate=16000 \
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
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.policy_loss.loss_mode=grpo \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.prompt_length=8192 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode="ar" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="minicpmo_4_5" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_mode="duplex" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=256 \
    actor_rollout_ref.rollout.cudagraph_capture_sizes='[1,2,4,8,16,32,64,128,256]' \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    reward.reward_manager.source=register \
    reward.reward_manager.name=minicpm_naive \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/duplex/rule_rewards.py \
    reward.custom_reward_function.name=compute_all_rewards \
    trainer.val_before_train=false \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=duplex_grpo \
    trainer.experiment_name=minicpm_o_4_5_duplex_grpo_lora_dailytalk \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    "$@"
