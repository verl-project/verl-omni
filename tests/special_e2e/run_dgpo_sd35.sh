#!/usr/bin/env bash
# DGPO e2e smoke on a tiny random SD3 checkpoint, V1 sync trainer, vllm_omni rollout.
#
# Covers trainer routing (direct_preference, online), the SD3 DGPO rollout adapter
# (deterministic rollout returning latents_clean), DGPOLoss.prepare_actor_batch, the
# NFT engine forwarding DGPO group fields, and the old-adapter EMA. Each actor GPU
# receives exactly one whole rollout group, which DGPO requires per micro batch.
#
# Reward is the pure-CPU jpeg compressibility score, so no reward-model server is needed.
# Builds the tiny checkpoint offline unless MODEL_PATH points at an existing one.
#
# Override via env: NUM_GPUS, MODEL_PATH, DATA_DIR, TOTAL_TRAIN_STEPS
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-1}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/stable-diffusion-3-tiny-random}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_dgpo}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-2}

if [[ ! -f "${MODEL_PATH}/model_index.json" ]]; then
    python3 tests/special_e2e/build_sd3_tiny_random.py --output-dir "${MODEL_PATH}"
fi

ENGINE=vllm_omni
max_prompt_length=128

# The tiny checkpoint has head_dim 4, below flash-attention's minimum of 8.
ATTN_BACKEND=native
ROLLOUT_ATTN_BACKEND=TORCH_SDPA

n_resp_per_prompt=2
# One whole group per GPU: the micro batch holds all n samples of a prompt.
micro_bsz_per_gpu=${n_resp_per_prompt}
mini_bsz=${NUM_GPUS}
train_batch_size=${mini_bsz}

python3 tests/special_e2e/create_dummy_diffusion_data.py \
    --local_save_dir "${DATA_DIR}" \
    --train_size $((train_batch_size * TOTAL_TRAIN_STEPS)) \
    --val_size 2

# SD3's CLIP tokenizer ships no chat template; pass through the raw user content.
custom_chat_template='{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}{{ message['\''content'\''] }}{% endif %}{% endfor %}'

python3 -m verl_omni.trainer.main_diffusion_v1 \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    algorithm.train_timestep_range="[0,3]" \
    algorithm.train_timestep_count=2 \
    algorithm.old_policy_decay_schedule=linear_to_0_3 \
    algorithm.old_policy_update_interval=1 \
    actor_rollout_ref.model.algorithm=dgpo \
    actor_rollout_ref.model.model_type=diffusion_nft_model \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.attn_backend=${ATTN_BACKEND} \
    actor_rollout_ref.model.custom_chat_template="\"${custom_chat_template}\"" \
    "actor_rollout_ref.model.extra_tokenizers={clip: {path: tokenizer, max_length: 77}, t5: {path: tokenizer_3, max_length: ${max_prompt_length}}}" \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out']" \
    actor_rollout_ref.actor.diffusion_loss.dgpo_beta=100.0 \
    actor_rollout_ref.actor.diffusion_loss.dgpo_clip_range=0.01 \
    actor_rollout_ref.actor.diffusion_loss.ref_kl_coef=0.02 \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.rollout_attn_backend=${ROLLOUT_ATTN_BACKEND} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.rollout_adapter=old \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.pipeline.height=256 \
    actor_rollout_ref.rollout.pipeline.width=256 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=${max_prompt_length} \
    actor_rollout_ref.rollout.max_prompt_embed_length=$((77 + max_prompt_length)) \
    actor_rollout_ref.rollout.algo.noise_level=0.0 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=4 \
    reward.num_workers=1 \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path=pkg://verl_omni.reward_loop.reward_manager.multi \
    reward.custom_reward_function.name=_multi_reward_placeholder \
    reward.reward_manager.name=MultiVisualRewardManager \
    reward.reward_manager.module.path=pkg://verl_omni.reward_loop.reward_manager \
    "+reward.reward_functions.jpeg.path=pkg://verl_omni.utils.reward_score.jpeg_compressibility" \
    '+reward.reward_functions.jpeg.name=compute_score' \
    '+reward.reward_functions.jpeg.weight=1.0' \
    reward.aggregation=weighted_sum \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=dgpo-sd35-e2e \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS} \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=sync \
    trainer.v1.sampler.drop_incomplete_groups=True \
    "$@"

echo "DGPO SD3.5 e2e smoke passed."
