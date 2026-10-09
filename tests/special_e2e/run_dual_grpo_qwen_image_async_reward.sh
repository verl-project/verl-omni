#!/usr/bin/env bash
# DualGRPO async-reward e2e smoke (minimal runtime), composite agent loop.
#
# Exercises:
#   dummy parquet -> composite_single_turn_agent (AR rewrite + DiT images) ->
#   streaming MultiVisualRewardManager on a dedicated reward pool ->
#   AR reward/ar aggregation inside the composite loop -> FlowGRPO LoRA update.
#
# GPU layout (async reward needs a reward pool separate from actor/rollout):
#   - NUM_GPUS_ACTOR:  colocated actor + rollout
#   - NUM_GPUS_REWARD: UnifiedReward stand-in (tiny Qwen3-VL) via vLLM
#   Defaults split NUM_GPUS as (NUM_GPUS - 1) + 1. At least 2 visible GPUs.
#
# Requires: vllm-omni, diffusers>=0.37,
#   tiny Qwen-Image at ~/models/tiny-random/Qwen-Image
#   tiny qwen3-vl  at ~/models/tiny-random/qwen3-vl
#
# Override via env: NUM_GPUS, NUM_GPUS_ACTOR, NUM_GPUS_REWARD, MODEL_PATH,
#                   REWARD_MODEL_PATH, DATA_DIR, TOTAL_TRAIN_STEPS, ...
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-4}
NUM_GPUS_REWARD=${NUM_GPUS_REWARD:-1}
NUM_GPUS_ACTOR=${NUM_GPUS_ACTOR:-$((NUM_GPUS - NUM_GPUS_REWARD))}
if [[ "${NUM_GPUS_ACTOR}" -lt 1 || "${NUM_GPUS_REWARD}" -lt 1 ]]; then
    echo "Need at least 1 actor GPU and 1 reward GPU" \
         "(NUM_GPUS=${NUM_GPUS}, NUM_GPUS_ACTOR=${NUM_GPUS_ACTOR}," \
         "NUM_GPUS_REWARD=${NUM_GPUS_REWARD})" >&2
    exit 2
fi

MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/Qwen-Image}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}/tokenizer}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-${HOME}/models/tiny-random/qwen3-vl}
REWARD_TP=${REWARD_TP:-1}
ROLLOUT_TP=${ROLLOUT_TP:-1}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_dualgrpo_async_reward}
dummy_train_path=${TRAIN_FILES:-${DATA_DIR}/train.parquet}
dummy_test_path=${VAL_FILES:-${DATA_DIR}/test.parquet}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-1}

ENGINE=vllm_omni
REWARD_ENGINE=vllm
max_prompt_length=256
IMAGE_RESOLUTION=256
VLLM_USE_FLASHINFER_SAMPLER=0

# Smoke: pin local FLASH_ATTN (product default remains FLASH_ATTN_3_HUB).
# Use exit-code checks only — importing verl/vllm may print INFO lines on stdout.
ATTN_BACKEND=_flash_3_varlen_hub
ROLLOUT_ATTN_BACKEND=FLASH_ATTN
if ! python3 -c 'from verl_omni.utils.diffusion_attention import fa_available; raise SystemExit(0 if fa_available() else 1)' >/dev/null 2>&1; then
    ATTN_BACKEND=native
    ROLLOUT_ATTN_BACKEND=TORCH_SDPA
fi

# rollout.m AR traces per prompt, rollout.n images per trace.
# Diffusion samples = train_batch_size * m * n; AR traces = train_batch_size * m.
# ppo_mini_batch_size is the prompt mini-batch; the trainer multiplies it by n
# (DiT) and by m (AR). Keep it equal to the prompt batch so each step is one
# group, and size both from the actor GPU count so micro-batches divide evenly.
rollout_m=2
rollout_n=2
micro_bsz_per_gpu=1
train_batch_size=4 #${NUM_GPUS_ACTOR}
mini_bsz=${train_batch_size}
synthetic_train_size=$((train_batch_size * TOTAL_TRAIN_STEPS))
if [[ "${synthetic_train_size}" -lt 2 ]]; then
    synthetic_train_size=2
fi

python3 tests/special_e2e/create_dummy_diffusion_data.py \
    --local_save_dir "${DATA_DIR}" \
    --train_size "${synthetic_train_size}" \
    --val_size 4

python3 -m verl_omni.trainer.main_diffusion \
    algorithm.adv_estimator=flow_grpo \
    data.train_files="${dummy_train_path}" \
    data.val_files="${dummy_test_path}" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    actor_rollout_ref.model.model_type=diffusion_composite_model \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.tokenizer_path="${TOKENIZER_PATH}" \
    actor_rollout_ref.model.algorithm=dual_grpo \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.attn_backend=${ATTN_BACKEND} \
    actor_rollout_ref.rollout.rollout_attn_backend=${ROLLOUT_ATTN_BACKEND} \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.exclude_modules=".*visual.*" \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.use_dynamic_bsz=False \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.rollout.ar_calculate_log_probs=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.m=${rollout_m} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.agent.default_agent_loop=composite_single_turn_agent \
    actor_rollout_ref.rollout.ar.response_length=32 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.height=${IMAGE_RESOLUTION} \
    actor_rollout_ref.rollout.pipeline.width=${IMAGE_RESOLUTION} \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=${max_prompt_length} \
    actor_rollout_ref.rollout.algo.noise_level=1.0 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,4]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.rollout.step_execution=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    reward.num_workers=$((NUM_GPUS_REWARD / REWARD_TP)) \
    reward.reward_model.enable=True \
    reward.reward_model.enable_resource_pool=True \
    reward.reward_model.model_path="${REWARD_MODEL_PATH}" \
    reward.reward_model.nnodes=1 \
    reward.reward_model.n_gpus_per_node=${NUM_GPUS_REWARD} \
    reward.reward_model.rollout.name=${REWARD_ENGINE} \
    reward.reward_model.rollout.tensor_model_parallel_size=${REWARD_TP} \
    reward.reward_model.rollout.gpu_memory_utilization=0.4 \
    reward.reward_model.rollout.free_cache_engine=False \
    reward.reward_model.rollout.enforce_eager=True \
    reward.reward_model.rollout.prompt_length=${max_prompt_length} \
    reward.reward_model.rollout.response_length=32 \
    reward.custom_reward_function.path=pkg://verl_omni.reward_loop.reward_manager.multi \
    reward.custom_reward_function.name=_multi_reward_placeholder \
    reward.reward_manager.name=MultiVisualRewardManager \
    reward.reward_manager.module.path=pkg://verl_omni.reward_loop.reward_manager \
    "+reward.reward_functions.ar.path=pkg://verl_omni.utils.reward_score.jpeg_compressibility" \
    '+reward.reward_functions.ar.name=compute_score' \
    '+reward.reward_functions.ar.weight=0.0' \
    "+reward.reward_functions.dit.path=verl_omni/utils/reward_score/genrm_ocr.py" \
    '+reward.reward_functions.dit.name=compute_score_ocr' \
    '+reward.reward_functions.dit.weight=1.0' \
    "+trainer.train_ar=True" \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=dualgrpo-qwen-image-async-reward-e2e \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node=${NUM_GPUS_ACTOR} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS} \
    "$@"

echo "DualGRPO async-reward e2e test passed (training completed successfully)."
