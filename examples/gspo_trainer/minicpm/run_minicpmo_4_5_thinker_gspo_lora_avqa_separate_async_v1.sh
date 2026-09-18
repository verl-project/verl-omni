#!/usr/bin/env bash
# MiniCPM-o 4.5 Thinker GSPO + LoRA training on AVQA (audio + image -> text) with
# the omni separate-async V1 trainer: 4 GPUs = 2 for the FSDP trainer, 2 for two
# standalone TP=1 rollout replicas. Generation runs one batch ahead of training;
# LoRA adapter deltas sync to the standalone replicas via add_lora every
# trainer.v1.separate_async.parameter_sync_step inner steps.
#
# Hyperparameters are copied verbatim from the two parent recipes: the colocated
# MiniCPM-o AVQA recipe
# (examples/gspo_trainer/minicpm/run_minicpmo_4_5_thinker_gspo_lora_avqa_v1.sh)
# supplies the model/data/reward lines; the Qwen3-Omni separate-async recipe
# (examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_mmk12_separate_async_v1.sh)
# supplies the disaggregated-topology lines.
#
# Requirements (asserted at startup by the trainer):
#   - actor_rollout_ref.rollout.nnodes > 0  (standalone rollout on dedicated GPUs)
#   - actor_rollout_ref.rollout.checkpoint_engine.backend != naive
#   - data.train_batch_size == parameter_sync_step * actor.ppo_mini_batch_size
#     (128 == 8 * 16, matching both parent recipes)
#
# If the 2-replica DP shape misbehaves (balancer fan-out under abort), fall back
# to one TP=2 replica: rollout.tensor_model_parallel_size=2 — config-only.
#
# Data preparation (run once, same as the colocated recipe):
#   python examples/gspo_trainer/data_process/avqa.py \
#       --input_dir <path_to_raw_AVQA_R1> \
#       --output_dir ~/data/avqa_r1_6k
#
# Runtime dependencies (all Ray worker nodes): soundfile + torchaudio (audio
# decode in MiniCPMORLHFDataset) and Pillow; both ship with the usual
# verl-omni GPU environment. flash_attention_2 only — do not switch the model
# to sdpa: it breaks train/rollout consistency (verified on Qwen3-Omni).

set -x

# Standalone server actors import verl_omni through this export; a silent
# module-miss fails the first warmup batch (trust_remote_code reaches the
# replica-side load through the same mechanism).
export VERL_USE_EXTERNAL_MODULES=verl_omni

MODEL_PATH=${MODEL_PATH:-"$HOME/models/openbmb/MiniCPM-o-4_5"}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/avqa_r1_6k/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/avqa_r1_6k/validation.parquet"}

python3 -m verl_omni.trainer.main_omni \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=12288 \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.omni_rl_datasets \
    data.custom_cls.name=MiniCPMORLHFDataset \
    +data.mm_processor_kwargs.sampling_rate=16000 \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.lora.merge=False \
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
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.nnodes=1 \
    actor_rollout_ref.rollout.n_gpus_per_node=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.prompt_length=4160 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode="ar" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="minicpmo_4_5" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=256 \
    actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16,32,64,128,256] \
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
    reward.custom_reward_function.path=verl_omni/utils/reward_score/choice_reward.py \
    reward.custom_reward_function.name=compute_score \
    trainer.v1.trainer_mode=omni_separate_async \
    trainer.v1.separate_async.num_warmup_batches=1 \
    trainer.v1.separate_async.parameter_sync_step=8 \
    trainer.v1.sampler.max_off_policy_threshold=8 \
    trainer.val_before_train=false \
    trainer.balance_batch=True \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=gspo \
    trainer.experiment_name=minicpm_o_4_5_thinker_lora_avqa_separate_async \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    "$@"
