#!/usr/bin/env bash
# LTX-2.3 FlowGRPO smoke test using a prebuilt tiny-random checkpoint.
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-1}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/LTX-2.3-Diffusers}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_ltx2_diffusion}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-1}
N_RESP_PER_PROMPT=2
TRAIN_BATCH_SIZE=$((NUM_GPUS * N_RESP_PER_PROMPT))

if [[ ! -f "${MODEL_PATH}/model_index.json" ]]; then
    echo "Missing tiny-random LTX-2.3 checkpoint: ${MODEL_PATH}" >&2
    exit 1
fi

python3 tests/special_e2e/create_dummy_diffusion_data.py \
    --local_save_dir "${DATA_DIR}" \
    --train_size "${TRAIN_BATCH_SIZE}" \
    --val_size 1

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length=32 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.model.lora_rank=4 \
    actor_rollout_ref.model.lora_alpha=8 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${NUM_GPUS}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n="${N_RESP_PER_PROMPT}" \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.agent.default_agent_loop=ltx2_diffusion_single_turn_agent \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.pipeline.height=64 \
    actor_rollout_ref.rollout.pipeline.width=96 \
    actor_rollout_ref.rollout.pipeline.num_frames=9 \
    actor_rollout_ref.rollout.pipeline.frame_rate=24.0 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=32 \
    actor_rollout_ref.rollout.algo.noise_level=0.8 \
    actor_rollout_ref.rollout.algo.sde_type=cps \
    actor_rollout_ref.rollout.algo.sde_steps="[0,1,2]" \
    actor_rollout_ref.rollout.algo.num_sde_steps=2 \
    actor_rollout_ref.rollout.algo.sde_window_seed=42 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.guidance_scale=1.0 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    reward.num_workers=1 \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path=pkg://verl_omni.utils.reward_score.jpeg_compressibility \
    reward.custom_reward_function.name=compute_score \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=flowgrpo-ltx2-3-e2e \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps="${TOTAL_TRAIN_STEPS}" \
    "$@"

echo "LTX-2.3 FlowGRPO e2e test passed."
