#!/usr/bin/env bash
# LTX-2.3 T2AV on-policy distillation e2e smoke on the v1 sync trainer.
#
# One training step with a single frozen teacher and a reduced video geometry.
# It exercises the teacher runtime end to end: the student's joint audio-video
# trajectory is replayed by a second full LTX-2.3 checkpoint and scored into
# teacher_prev_sample_mean, and distill_kl is the only objective. Reward is the
# pure-CPU jpeg compressibility score, so no reward-model server is needed --
# this smoke is about the teacher runtime, not about reward quality.
#
# Unlike run_diffusion_teacher_smoke.sh this needs real checkpoints: there is no
# tiny-random LTX-2.3 builder, so MODEL_PATH and TEACHER_PATH must both point at
# full diffusers pipelines resolving to the same scheduler config. That is why it
# is not registered in the unattended smoke suites.
#
# Override via env: NUM_GPUS, ROLLOUT_TP, MODEL_PATH, TEACHER_PATH, DATA_DIR,
# PROMPT_DIR, TOTAL_TRAIN_STEPS, TEACHER_NNODES, TEACHER_NPUS
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-8}
ROLLOUT_TP=${ROLLOUT_TP:-2}
MODEL_PATH=${MODEL_PATH:-dg845/LTX-2.3-Diffusers}
TEACHER_PATH=${TEACHER_PATH:?set TEACHER_PATH to a full LTX-2.3 diffusers pipeline}
PROMPT_DIR=${PROMPT_DIR:?set PROMPT_DIR to a directory holding train.txt and test.txt}
DATA_DIR=${DATA_DIR:-${HOME}/data/ltx2_opd_smoke}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-1}
TEACHER_NNODES=${TEACHER_NNODES:-0}
TEACHER_NPUS=${TEACHER_NPUS:-0}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${repo_root}"

n_resp_per_prompt=2
micro_bsz_per_gpu=1
mini_bsz=$((micro_bsz_per_gpu * NUM_GPUS))
train_batch_size=$((mini_bsz * n_resp_per_prompt))

if [[ ! -f "${DATA_DIR}/train.parquet" ]]; then
    python3 examples/flowgrpo_trainer/ltx2/prepare_data.py \
        --input_dir "${PROMPT_DIR}" \
        --output_dir "${DATA_DIR}" \
        --train_size "${train_batch_size}" \
        --val_size 2
fi

python3 -m verl_omni.trainer.main_diffusion_v1 \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size="${train_batch_size}" \
    data.val_max_samples=2 \
    data.max_prompt_length=128 \
    data.truncation=error \
    algorithm.adv_estimator=flow_grpo \
    distillation.enabled=True \
    distillation.nnodes="${TEACHER_NNODES}" \
    distillation.n_gpus_per_node="${TEACHER_NPUS}" \
    distillation.teacher_models.teacher_model.model_path="${TEACHER_PATH}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.fsdp_layer_prefixes="['transformer_blocks.']" \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size="${mini_bsz}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${micro_bsz_per_gpu}" \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=distill_kl \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.n="${n_resp_per_prompt}" \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.agent.default_agent_loop=ltx2_diffusion_single_turn_agent \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.pipeline.height=128 \
    actor_rollout_ref.rollout.pipeline.width=128 \
    actor_rollout_ref.rollout.pipeline.num_frames=25 \
    actor_rollout_ref.rollout.pipeline.frame_rate=24.0 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=128 \
    +actor_rollout_ref.rollout.pipeline.output_type=pt \
    actor_rollout_ref.rollout.algo.noise_level=0.8 \
    actor_rollout_ref.rollout.algo.sde_type=cps \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,2]" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.val_kwargs.pipeline.height=128 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.width=128 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_frames=25 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=4 \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.output_type=pt \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
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
    trainer.log_val_generations=0 \
    trainer.project_name=diffusion_opd_smoke \
    trainer.experiment_name=ltx2_3_t2av_opd_smoke \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps="${TOTAL_TRAIN_STEPS}" \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=sync "$@"
