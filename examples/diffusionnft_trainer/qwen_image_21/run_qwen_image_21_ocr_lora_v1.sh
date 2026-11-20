#!/usr/bin/env bash
# Qwen-Image 2.1 LoRA NFT；encoder、VAE 和 reward 模型冻结。
set -euo pipefail

: "${QWEN21_MODEL_PATH:?Set the local path of the pinned Qwen-Image 2.1 checkpoint}"
: "${QWEN21_REWARD_MODEL_PATH:?Set the local Qwen3-VL-8B-Instruct checkpoint path}"
: "${QWEN21_TRAIN_DATA:?Set the OCR training parquet path}"
: "${QWEN21_VAL_DATA:?Set the held-out OCR parquet path}"
: "${QWEN21_OUTPUT_DIR:?Set a writable local checkpoint directory}"

resolution=${QWEN21_RESOLUTION:-512}
updates=${QWEN21_UPDATES:-20}
seed=${QWEN21_SEED:-42}

python3 -m verl_omni.trainer.main_diffusion_v1 \
    data.train_files="$QWEN21_TRAIN_DATA" \
    data.val_files="$QWEN21_VAL_DATA" \
    data.train_batch_size=8 \
    data.max_prompt_length=32768 \
    actor_rollout_ref.model.path="$QWEN21_MODEL_PATH" \
    actor_rollout_ref.model.tokenizer_path="$QWEN21_MODEL_PATH/processor" \
    actor_rollout_ref.model.algorithm=diffusion_nft \
    actor_rollout_ref.model.model_type=diffusion_nft_model \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
    actor_rollout_ref.model.target_modules='["to_q","to_k","to_v","to_out.0"]' \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.offload_policy=true \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=4 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.seed="$seed" \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=diffusion_nft \
    actor_rollout_ref.actor.diffusion_loss.mix_beta=0.1 \
    actor_rollout_ref.actor.diffusion_loss.ref_kl_coef=0.0001 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.agent.default_agent_loop=qwen_image_21_diffusion_single_turn_agent \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.agent.num_workers=4 \
    actor_rollout_ref.rollout.max_num_seqs=1 \
    actor_rollout_ref.rollout.max_prompt_embed_length=2048 \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.calculate_log_probs=false \
    actor_rollout_ref.rollout.rollout_adapter=old \
    actor_rollout_ref.rollout.checkpoint_engine.backend=naive \
    actor_rollout_ref.rollout.seed="$seed" \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=40 \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.height="$resolution" \
    actor_rollout_ref.rollout.pipeline.width="$resolution" \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=2048 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=40 \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    algorithm.timestep_fraction=1.0 \
    algorithm.old_policy_decay_schedule=delayed_linear_to_0_999 \
    algorithm.old_policy_update_interval=2 \
    algorithm.adv_mode=continuous \
    reward.num_workers=1 \
    reward.reward_model.enable=true \
    reward.reward_model.model_path="$QWEN21_REWARD_MODEL_PATH" \
    reward.reward_model.rollout.name=vllm \
    reward.reward_model.rollout.tensor_model_parallel_size=4 \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/genrm_ocr.py \
    reward.custom_reward_function.name=compute_score_ocr \
    trainer.default_local_dir="$QWEN21_OUTPUT_DIR" \
    trainer.logger='[console]' \
    trainer.project_name=qwen_image_21_diffusion_nft \
    trainer.experiment_name=ocr_lora_v1 \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=100 \
    trainer.total_training_steps="$updates" \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=sync "$@"
