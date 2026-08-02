#!/usr/bin/env bash
# Diffusion OPD teacher-runtime e2e smoke.
#
# Two runs over one SD3 checkpoint pair:
#   SMOKE=distill      pure distillation -- teacher is a *distinct* checkpoint and
#                      `diffusion_loss.loss_mode=distill_kl` is the only objective.
#   SMOKE=coexistence  actor + reference + teacher live at once, with use_kl_loss on.
#                      lora_rank/lora_adapter_path are pinned empty on purpose: either
#                      one set makes main_diffusion fold the ref into the actor, and the
#                      run would pass while holding only two model states.
#
# Reward is the pure-CPU jpeg compressibility score, so no reward-model server is
# needed -- this smoke is about the teacher runtime, not about reward quality.
#
# MODEL_PATH/TEACHER_PATH must be real SD3 checkpoints: the online path serves
# rollout through vllm-omni, whose SD3 pipeline builds the *slow* T5Tokenizer, and
# build_sd3_tiny_random.py writes a fast tokenizer with no sentencepiece model
# (`TypeError: argument 'vocab' ...`). The tiny checkpoint is therefore usable by
# the offline-DPO recipe but not by any rollout-backed recipe, this one included.
#
# Override via env: NUM_GPUS, MODEL_PATH, TEACHER_PATH, DATA_DIR, TOTAL_TRAIN_STEPS, SMOKE
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-1}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to an SD3 checkpoint (see note above)}
TEACHER_PATH=${TEACHER_PATH:?set TEACHER_PATH to a second SD3 checkpoint}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_diffusion_teacher}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-1}
SMOKE=${SMOKE:-distill}

ENGINE=vllm_omni
max_prompt_length=128

ATTN_BACKEND=_flash_3_varlen_hub
ROLLOUT_ATTN_BACKEND=FLASH_ATTN
if ! python3 -c 'from verl_omni.utils.diffusion_attention import fa3_available; raise SystemExit(0 if fa3_available() else 1)' >/dev/null 2>&1; then
    ATTN_BACKEND=native
    ROLLOUT_ATTN_BACKEND=TORCH_SDPA
fi

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
    actor_rollout_ref.teacher.enabled=True \
    "+actor_rollout_ref.teacher.models.default.model.path=${TEACHER_PATH}" \
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
