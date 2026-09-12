#!/usr/bin/env bash
# Diffusion OPD teacher-runtime e2e smoke on the v1 trainer.
#
# Two runs over tiny SD3 checkpoints, both routing rows between two teachers by
# `data_source`:
#   SMOKE=sync   v1 sync trainer, both teachers colocated with the actor, and the
#                flow_grpo objective plus the auxiliary distill term
#                (use_distill_loss, distill_kl) and use_kl_loss, so actor,
#                reference and teachers all live at once. lora_rank and
#                lora_adapter_path are pinned empty on purpose: either one set
#                folds the ref into the actor, and the run would pass while
#                holding one model state fewer.
#   SMOKE=async  v1 separate_async trainer with the one_step_off teacher
#                schedule, each teacher on its own GPU in the distillation
#                resource pool, and distillation as the only objective. Needs
#                NUM_GPUS >= 4: two teacher GPUs, one standalone rollout GPU and
#                at least one actor GPU.
#
# Reward is the pure-CPU jpeg compressibility score, so no reward-model server is
# needed -- this smoke is about the teacher runtime, not about reward quality.
#
# By default the script builds three tiny random SD3 checkpoints (different seeds,
# so the step-1 KL is positive) and needs no downloads; set MODEL_PATH,
# TEACHER_PATH and TEACHER2_PATH to run against real checkpoints instead.
#
# Override via env: NUM_GPUS, MODEL_PATH, TEACHER_PATH, TEACHER2_PATH, DATA_DIR, TOTAL_TRAIN_STEPS, SMOKE
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-1}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/sd3-teacher-smoke-student}
TEACHER_PATH=${TEACHER_PATH:-${HOME}/models/tiny-random/sd3-teacher-smoke-teacher}
TEACHER2_PATH=${TEACHER2_PATH:-${HOME}/models/tiny-random/sd3-teacher-smoke-teacher2}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_diffusion_teacher}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-1}
SMOKE=${SMOKE:-sync}

if [[ ! -f "${MODEL_PATH}/model_index.json" ]]; then
    python3 tests/special_e2e/build_sd3_tiny_random.py --output-dir "${MODEL_PATH}" --seed 0
fi
if [[ ! -f "${TEACHER_PATH}/model_index.json" ]]; then
    python3 tests/special_e2e/build_sd3_tiny_random.py --output-dir "${TEACHER_PATH}" --seed 1
fi
if [[ ! -f "${TEACHER2_PATH}/model_index.json" ]]; then
    python3 tests/special_e2e/build_sd3_tiny_random.py --output-dir "${TEACHER2_PATH}" --seed 2
fi

ENGINE=vllm_omni
max_prompt_length=128

# The tiny checkpoints have head_dim 4, below flash-attention's minimum of 8.
ATTN_BACKEND=native
ROLLOUT_ATTN_BACKEND=TORCH_SDPA

# The async smoke gives two GPUs to the teacher pool and one to the standalone
# rollout; the actor keeps the rest.
if [[ "${SMOKE}" == "async" ]]; then
    actor_gpus=$((NUM_GPUS - 3))
else
    actor_gpus=${NUM_GPUS}
fi

n_resp_per_prompt=2
micro_bsz_per_gpu=1
mini_bsz=$((micro_bsz_per_gpu * actor_gpus))
train_batch_size=$((mini_bsz * n_resp_per_prompt))

python3 tests/special_e2e/create_dummy_diffusion_data.py \
    --local_save_dir "${DATA_DIR}" \
    --train_size "${train_batch_size}" \
    --val_size 2 \
    --data_sources teacher_a,teacher_b

# SD3's CLIP tokenizer ships no chat template, and the diffusion agent loop applies
# one to every prompt; pass through the raw user content.
custom_chat_template='{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}{{ message['\''content'\''] }}{% endif %}{% endfor %}'

# Objective, teacher placement and trainer mode differ per smoke.
case "${SMOKE}" in
    sync)
        placement=()
        objective=(
            actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo
            actor_rollout_ref.actor.use_distill_loss=True
            actor_rollout_ref.actor.distill_loss_mode=distill_kl
            actor_rollout_ref.actor.use_kl_loss=True
            actor_rollout_ref.actor.kl_loss_coef=0.04
        )
        mode=(trainer.v1.trainer_mode=sync)
        ;;
    async)
        placement=(
            distillation.n_gpus_per_node=2
            distillation.nnodes=1
        )
        objective=(
            actor_rollout_ref.actor.diffusion_loss.loss_mode=distill_kl
            actor_rollout_ref.actor.use_kl_loss=False
        )
        mode=(
            trainer.v1.trainer_mode=separate_async
            actor_rollout_ref.rollout.mode=async
            actor_rollout_ref.rollout.calculate_log_probs=True
            actor_rollout_ref.rollout.nnodes=1
            actor_rollout_ref.rollout.n_gpus_per_node=1
            actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
            "actor_rollout_ref.actor.ppo_mini_batch_size=${train_batch_size}"
            # one_step_off fills a one-batch teacher pipeline and needs that much generation lead.
            trainer.v1.separate_async.num_warmup_batches=2
            trainer.v1.separate_async.parameter_sync_step=1
            distillation.scheduler=one_step_off
        )
        ;;
    *)
        echo "Unknown SMOKE=${SMOKE}; expected 'sync' or 'async'." >&2
        exit 1
        ;;
esac

python3 -m verl_omni.trainer.main_diffusion_v1 \
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
    +distillation.teacher_models.a.key=teacher_a \
    +distillation.teacher_models.a.model_path="${TEACHER_PATH}" \
    +distillation.teacher_models.a.world_size=1 \
    +distillation.teacher_models.b.key=teacher_b \
    +distillation.teacher_models.b.model_path="${TEACHER2_PATH}" \
    +distillation.teacher_models.b.world_size=1 \
    "${placement[@]}" \
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
    trainer.n_gpus_per_node=${actor_gpus} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS} \
    trainer.use_v1=true \
    "${mode[@]}" \
    "$@"

echo "Diffusion teacher e2e smoke (${SMOKE}) passed."
