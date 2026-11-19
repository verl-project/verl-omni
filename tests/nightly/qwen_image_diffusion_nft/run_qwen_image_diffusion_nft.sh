#!/usr/bin/env bash
# Nightly Qwen-Image DiffusionNFT regression on local tiny random weights.
# V1 sync trainer only (verl_omni.trainer.main_diffusion_v1).
# OCR reward scoring matches the FlowGRPO and online DPO nightlies.
set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

export L3_TEST_CASE=${L3_TEST_CASE:-qwen_image_diffusion_nft}
export L3_ALGORITHM=${L3_ALGORITHM:-diffusion-nft}
export L3_CASE_LABEL=${L3_CASE_LABEL:-"Qwen-Image DiffusionNFT"}
export L3_DATA_SOURCE=${L3_DATA_SOURCE:-qwen_image_diffusion_nft}
export NUM_GPUS=${NUM_GPUS:-1}
export REWARD_TP=${REWARD_TP:-1}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-256}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-4}
MICRO_BSZ_PER_GPU=${MICRO_BSZ_PER_GPU:-4}
MICRO_BSZ=$((MICRO_BSZ_PER_GPU * NUM_GPUS))
MINI_BSZ=${MICRO_BSZ}
export TRAIN_BATCH_SIZE=$((MINI_BSZ * N_RESP_PER_PROMPT))

ENGINE=vllm_omni
REWARD_ENGINE=vllm
ATTN_BACKEND=${ATTN_BACKEND:-native}
ROLLOUT_ATTN_BACKEND=${ROLLOUT_ATTN_BACKEND:-TORCH_SDPA}

bash "${REPO_ROOT}/tests/nightly/run_l3_case.sh" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.shuffle=false \
    data.seed=42 \
    actor_rollout_ref.model.algorithm=diffusion_nft \
    actor_rollout_ref.model.model_type=diffusion_nft_model \
    actor_rollout_ref.model.path="${MODEL_PATH:-${HOME}/models/tiny-random/Qwen-Image}" \
    actor_rollout_ref.model.tokenizer_path="${TOKENIZER_PATH:-${MODEL_PATH:-${HOME}/models/tiny-random/Qwen-Image}/tokenizer}" \
    actor_rollout_ref.model.attn_backend=${ATTN_BACKEND} \
    actor_rollout_ref.rollout.rollout_attn_backend=${ROLLOUT_ATTN_BACKEND} \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.pipeline.height=256 \
    actor_rollout_ref.model.pipeline.width=256 \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BSZ} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.shuffle=false \
    actor_rollout_ref.actor.data_loader_seed=42 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=diffusion_nft \
    actor_rollout_ref.actor.diffusion_loss.mix_beta=0.5 \
    actor_rollout_ref.actor.diffusion_loss.ref_kl_coef=0.001 \
    actor_rollout_ref.actor.diffusion_loss.adv_clip_max=5.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.seed=42 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT} \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.rollout_adapter=old \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.pipeline.height=256 \
    actor_rollout_ref.rollout.pipeline.width=256 \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=4.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=${MAX_PROMPT_LENGTH} \
    actor_rollout_ref.rollout.algo.noise_level=0.0 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=null \
    actor_rollout_ref.rollout.algo.sde_window_range=null \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ_PER_GPU} \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    algorithm.timestep_fraction=1.0 \
    algorithm.old_policy_decay_schedule=delayed_linear_to_0_999 \
    algorithm.old_policy_update_interval=1 \
    algorithm.adv_mode=continuous \
    reward.num_workers=1 \
    reward.reward_model.enable=True \
    reward.reward_model.model_path="${REWARD_MODEL_PATH:-${HOME}/models/tiny-random/qwen3-vl}" \
    reward.reward_model.rollout.name=${REWARD_ENGINE} \
    reward.reward_model.rollout.tensor_model_parallel_size=${REWARD_TP} \
    reward.reward_model.rollout.gpu_memory_utilization=0.4 \
    reward.reward_model.rollout.enforce_eager=True \
    reward.reward_model.rollout.do_sample=False \
    reward.reward_model.rollout.temperature=0.0 \
    reward.reward_model.rollout.prompt_length=${MAX_PROMPT_LENGTH} \
    reward.reward_model.rollout.response_length=32 \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/genrm_ocr.py \
    reward.custom_reward_function.name=compute_score_ocr \
    trainer.logger=console \
    trainer.project_name=verl-nightly \
    trainer.experiment_name=qwen-image-diffusion-nft-nightly \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS:-20} \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=sync \
    "$@"
