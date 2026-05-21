#!/usr/bin/env bash
# DiffusionNFT diffusion e2e smoke test, vllm_omni rollout.
#
# Requires: vllm-omni, diffusers>=0.37, and a tiny Qwen-Image checkpoint at
#   ~/models/tiny-random/Qwen-Image
set -xeuo pipefail
export FLASHINFER_DISABLE_VERSION_CHECK=${FLASHINFER_DISABLE_VERSION_CHECK:-1}

NUM_GPUS=${NUM_GPUS:-4}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/Qwen-Image}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}/tokenizer}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_diffusion}
dummy_train_path=${TRAIN_FILES:-${DATA_DIR}/train.parquet}
dummy_test_path=${VAL_FILES:-${DATA_DIR}/test.parquet}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-4}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}

ENGINE=vllm_omni
max_prompt_length=256

n_resp_per_prompt=2
micro_bsz_per_gpu=1
micro_bsz=$((micro_bsz_per_gpu * NUM_GPUS))
mini_bsz=${micro_bsz}
train_batch_size=$((mini_bsz * n_resp_per_prompt))
steps_per_epoch=$(((TOTAL_TRAIN_STEPS + TOTAL_EPOCHS - 1) / TOTAL_EPOCHS))
synthetic_train_size=$((train_batch_size * steps_per_epoch))

python3 tests/special_e2e/create_dummy_diffusion_data.py \
    --local_save_dir "${DATA_DIR}" \
    --train_size "${synthetic_train_size}" \
    --val_size 4

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=${dummy_train_path} \
    data.val_files=${dummy_test_path} \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    actor_rollout_ref.model.algorithm=diffusion_nft \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.tokenizer_path=${TOKENIZER_PATH} \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.collect_mode=final_latent \
    actor_rollout_ref.rollout.rollout_adapter=old \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.pipeline.height=256 \
    actor_rollout_ref.rollout.pipeline.width=256 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=4.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=${max_prompt_length} \
    actor_rollout_ref.rollout.algo.noise_level=0.0 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=null \
    actor_rollout_ref.rollout.algo.sde_window_range=null \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    algorithm.diffusion_nft.mix_beta=0.5 \
    algorithm.diffusion_nft.ref_kl_coef=0.001 \
    algorithm.diffusion_nft.timestep_fraction=1.0 \
    algorithm.diffusion_nft.old_policy_decay_type=1 \
    algorithm.diffusion_nft.old_policy_update_interval=1 \
    algorithm.diffusion_nft.adv_clip_max=5.0 \
    algorithm.diffusion_nft.adv_mode=continuous \
    algorithm.diffusion_nft.rollout_adapter=old \
    algorithm.diffusion_nft.collect_mode=final_latent \
    reward.num_workers=1 \
    reward.reward_model.enable=False \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=diffusionnft-diffusion-e2e \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS} \
    "$@"

echo "DiffusionNFT diffusion e2e test passed (training completed successfully)."
