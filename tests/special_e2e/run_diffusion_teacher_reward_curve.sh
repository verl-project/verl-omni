#!/usr/bin/env bash
# Diffusion OPD teacher-runtime *reward-curve* experiment.
#
# NOT a CI smoke. Reproduces the base -> teacher -> student OCR-reward result
# through the teacher runtime, so the runtime carries its own algorithm evidence
# instead of borrowing an earlier figure.
#
# This is the exact, proven command (LoRA rank 32, cps SDE, and a dedicated
# 1-GPU reward pool serving Qwen2.5-VL-3B at gpu_memory_utilization=0.9,
# enforce_eager=False -- the config that actually produced that curve). The
# ONLY deliberate change is the teacher mechanism: the earlier run overloaded
# the ref slot (`ref.model_path=<teacher>`); this uses the separate teacher
# runtime (`actor_rollout_ref.teacher.*`). Everything else is held fixed so a
# difference in the curve is attributable to the runtime, not to a re-tuned
# recipe.
#
# Reward is monitored, not optimised: the loss is pure distill_kl. val reward is
# the plotted series (base at step 0 via val_before_train, then every TEST_FREQ).
#
# Required env: MODEL_PATH, TEACHER_PATH, REWARD_MODEL_PATH, TRAIN_FILES, VAL_FILES
# Optional:     TOTAL_TRAIN_STEPS (default 100), TEST_FREQ (20), NUM_TRAIN_GPUS (2)
set -euo pipefail

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to a real SD3.5 student checkpoint}
TEACHER_PATH=${TEACHER_PATH:?set TEACHER_PATH to the merged OCR teacher checkpoint}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:?set REWARD_MODEL_PATH to a real vision-LLM (e.g. Qwen2.5-VL-3B-Instruct)}
TRAIN_FILES=${TRAIN_FILES:?set TRAIN_FILES to the OCR train parquet}
VAL_FILES=${VAL_FILES:?set VAL_FILES to the OCR val parquet}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-100}
TEST_FREQ=${TEST_FREQ:-20}
NUM_TRAIN_GPUS=${NUM_TRAIN_GPUS:-2}   # reward pool takes 1 more GPU (dedicated), so 3 total

# SD3's CLIP tokenizer ships no chat template; pass the raw user content through.
custom_chat_template='{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}{{ message['\''content'\''] }}{% endif %}{% endfor %}'

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=8 \
    data.val_max_samples=32 \
    data.max_prompt_length=512 \
    data.truncation=error \
    data.seed=42 \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.custom_chat_template="\"${custom_chat_template}\"" \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    "actor_rollout_ref.model.target_modules=[to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out]" \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=distill_kl \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.teacher.enabled=True \
    "+actor_rollout_ref.teacher.models.default.model.path=${TEACHER_PATH}" \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.pipeline.height=384 \
    actor_rollout_ref.rollout.pipeline.width=384 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.algo.noise_level=0.8 \
    actor_rollout_ref.rollout.algo.sde_type=cps \
    actor_rollout_ref.rollout.algo.sde_window_size=3 \
    "actor_rollout_ref.rollout.algo.sde_window_range=[0,5]" \
    "+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=256" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=28 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    reward.num_workers=1 \
    reward.reward_model.enable=True \
    reward.reward_model.enable_resource_pool=True \
    reward.reward_model.nnodes=1 \
    reward.reward_model.n_gpus_per_node=1 \
    reward.reward_model.model_path="${REWARD_MODEL_PATH}" \
    reward.reward_model.rollout.name=vllm \
    reward.reward_model.rollout.tensor_model_parallel_size=1 \
    reward.reward_model.rollout.gpu_memory_utilization=0.9 \
    reward.reward_model.rollout.free_cache_engine=False \
    reward.reward_model.rollout.enforce_eager=False \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/genrm_ocr.py \
    reward.custom_reward_function.name=compute_score_ocr \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=diffusion-teacher-reward-curve \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node=${NUM_TRAIN_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=${TEST_FREQ} \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS} \
    "$@"

echo "Diffusion teacher reward-curve experiment finished."
