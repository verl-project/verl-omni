#!/usr/bin/env bash
# VeOmni actor engine + LoRA e2e smoke on tiny-random Qwen-Image and dummy data.
#
# Single pass covering:
#   parquet load -> vllm_omni rollout -> jpeg_compressibility rule reward ->
#   flow_grpo -> VeOmni LoRA actor update -> adapter weight sync.
#
# Requires: veomni>=0.1.12 (veomni.lora), tiny Qwen-Image at
#   ~/models/tiny-random/Qwen-Image (hf: tiny-random/Qwen-Image)
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
NUM_GPUS=${NUM_GPUS:-2}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/Qwen-Image}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}/tokenizer}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_diffusion}
dummy_train_path=${TRAIN_FILES:-${DATA_DIR}/train.parquet}
dummy_test_path=${VAL_FILES:-${DATA_DIR}/test.parquet}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-2}

ENGINE=vllm_omni
max_prompt_length=256

# The tiny checkpoint has no FA3-compatible head dim; keep the portable backends.
ATTN_BACKEND=native
ROLLOUT_ATTN_BACKEND=TORCH_SDPA

# BACKEND=veomni|fsdp2 -- both arms share every other setting so the A/B is matched,
# except lora_init_weights: veomni.lora only implements Kaiming-uniform A / zero B and
# rejects PEFT's "gaussian" rather than silently substituting it, while PEFT rejects the
# string "true". The two arms therefore differ in LoRA init scale by construction.
BACKEND=${BACKEND:-veomni}
if [[ "${BACKEND}" == "veomni" ]]; then
    engine_args=(
        diffusion/model_engine=veomni_diffusion
        actor_rollout_ref.actor.strategy=veomni
        actor_rollout_ref.actor.veomni_config.strategy=veomni
        actor_rollout_ref.actor.veomni_config.param_offload=True
        actor_rollout_ref.actor.veomni_config.optimizer_offload=True
        actor_rollout_ref.ref.veomni_config.strategy=veomni
        actor_rollout_ref.model.lora_init_weights=true
    )
elif [[ "${BACKEND}" == "fsdp2" ]]; then
    engine_args=(
        actor_rollout_ref.actor.strategy=fsdp2
        actor_rollout_ref.actor.fsdp_config.param_offload=True
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
        actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
    )
else
    echo "FAIL: BACKEND must be veomni or fsdp2, got '${BACKEND}'" >&2
    exit 1
fi

n_resp_per_prompt=2
micro_bsz_per_gpu=1
mini_bsz=$((micro_bsz_per_gpu * NUM_GPUS))
train_batch_size=$((mini_bsz * n_resp_per_prompt))

python3 tests/special_e2e/create_dummy_diffusion_data.py \
    --local_save_dir "${DATA_DIR}" \
    --train_size "${train_batch_size}" \
    --val_size 4 \
    --data_sources jpeg_compressibility

python3 -m verl_omni.trainer.main_diffusion \
    trainer.use_v1=false \
    "${engine_args[@]}" \
    algorithm.adv_estimator=flow_grpo \
    data.train_files=${dummy_train_path} \
    data.val_files=${dummy_test_path} \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.tokenizer_path=${TOKENIZER_PATH} \
    actor_rollout_ref.model.attn_backend=${ATTN_BACKEND} \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    'actor_rollout_ref.model.target_modules=[to_q,to_k,to_v,to_out.0]' \
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-1e-4} \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.rollout_attn_backend=${ROLLOUT_ATTN_BACKEND} \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.pipeline.height=256 \
    actor_rollout_ref.rollout.pipeline.width=256 \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=4.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=${max_prompt_length} \
    actor_rollout_ref.rollout.algo.noise_level=1.0 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,4]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    reward.num_workers=1 \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path=pkg://verl_omni.reward_loop.reward_manager.multi \
    reward.custom_reward_function.name=_multi_reward_placeholder \
    reward.reward_manager.name=MultiVisualRewardManager \
    reward.reward_manager.module.path=pkg://verl_omni.reward_loop.reward_manager \
    "+reward.reward_functions.jpeg.path=pkg://verl_omni.utils.reward_score.jpeg_compressibility" \
    '+reward.reward_functions.jpeg.name=compute_score' \
    '+reward.reward_functions.jpeg.weight=1.0' \
    trainer.logger=console \
    trainer.project_name=veomni_lora_smoke \
    trainer.experiment_name=qwen_image_${BACKEND}_lora \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_epochs=${TOTAL_EPOCHS:-1} \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS} \
    ray_kwargs.ray_init.num_cpus=32 \
    "+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=${REPO_ROOT}" \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_LOGGING_LEVEL=DEBUG' \
    "$@"
