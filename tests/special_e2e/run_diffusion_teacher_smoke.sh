#!/usr/bin/env bash
# Diffusion OPD teacher-runtime e2e smoke.
#
# Two runs over one SD3 checkpoint pair:
#   SMOKE=distill      pure distillation -- teacher is a *distinct* checkpoint and
#                      `diffusion_loss.loss_mode=distill_kl` is the only objective.
#   SMOKE=coexistence  actor + reference + teacher live at once: flow_grpo plus the
#                      auxiliary distill term (use_distill_loss, distill_kl) and use_kl_loss.
#                      lora_rank/lora_adapter_path are pinned empty on purpose: either
#                      one set makes main_diffusion fold the ref into the actor, and the
#                      run would pass while holding only two model states.
#
# Reward is the pure-CPU jpeg compressibility score, so no reward-model server is
# needed -- this smoke is about the teacher runtime, not about reward quality.
#
# By default the script builds two tiny random SD3 checkpoints (different seeds,
# so the step-1 KL is positive) and needs no downloads; set MODEL_PATH and
# TEACHER_PATH to run against real checkpoints instead.
#
# Override via env: NUM_GPUS, MODEL_PATH, TEACHER_PATH, DATA_DIR, TOTAL_TRAIN_STEPS, SMOKE
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-1}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/sd3-teacher-smoke-student}
TEACHER_PATH=${TEACHER_PATH:-${HOME}/models/tiny-random/sd3-teacher-smoke-teacher}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_diffusion_teacher}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-1}
SMOKE=${SMOKE:-distill}

if [[ ! -f "${MODEL_PATH}/model_index.json" ]]; then
    python3 tests/special_e2e/build_sd3_tiny_random.py --output-dir "${MODEL_PATH}" --seed 0
fi
if [[ ! -f "${TEACHER_PATH}/model_index.json" ]]; then
    python3 tests/special_e2e/build_sd3_tiny_random.py --output-dir "${TEACHER_PATH}" --seed 1
fi

ENGINE=vllm_omni
max_prompt_length=128

# The tiny checkpoints have head_dim 4, below flash-attention's minimum of 8.
ATTN_BACKEND=native
ROLLOUT_ATTN_BACKEND=TORCH_SDPA

n_resp_per_prompt=2
micro_bsz_per_gpu=1
mini_bsz=$((micro_bsz_per_gpu * NUM_GPUS))
train_batch_size=$((mini_bsz * n_resp_per_prompt))

python3 tests/special_e2e/create_dummy_diffusion_data.py \
    --local_save_dir "${DATA_DIR}" \
    --train_size "${train_batch_size}" \
    --val_size 2

# SD3's CLIP tokenizer ships no chat template, and the diffusion agent loop applies
# one to every prompt; pass through the raw user content.
custom_chat_template='{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}{{ message['\''content'\''] }}{% endif %}{% endfor %}'

# Objective differs per smoke; the teacher runtime config does not.
case "${SMOKE}" in
    distill)
        objective=(
            actor_rollout_ref.actor.diffusion_loss.loss_mode=distill_kl
            actor_rollout_ref.actor.use_kl_loss=False
        )
        ;;
    coexistence)
        objective=(
            actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo
            actor_rollout_ref.actor.use_distill_loss=True
            actor_rollout_ref.actor.distill_loss_mode=distill_kl
            actor_rollout_ref.actor.use_kl_loss=True
            actor_rollout_ref.actor.kl_loss_coef=0.04
        )
        ;;
    *)
        echo "Unknown SMOKE=${SMOKE}; expected 'distill' or 'coexistence'." >&2
        exit 1
        ;;
esac

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.attn_backend=${ATTN_BACKEND} \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.lora_adapter_path=null \
    actor_rollout_ref.model.custom_chat_template="\"${custom_chat_template}\"" \
    "actor_rollout_ref.model.extra_tokenizers={clip: {path: tokenizer, max_length: 77}, t5: {path: tokenizer_3, max_length: ${max_prompt_length}}}" \
    distillation.enabled=True \
    distillation.teacher_models.teacher_model.model_path="${TEACHER_PATH}" \
    "${objective[@]}" \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.rollout_attn_backend=${ROLLOUT_ATTN_BACKEND} \
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
    actor_rollout_ref.rollout.pipeline.max_sequence_length=${max_prompt_length} \
    actor_rollout_ref.rollout.max_prompt_embed_length=$((77 + max_prompt_length)) \
    actor_rollout_ref.rollout.algo.noise_level=1.0 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
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
    trainer.experiment_name=diffusion-teacher-${SMOKE} \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS} \
    "$@"

echo "Diffusion teacher e2e smoke (${SMOKE}) passed."
