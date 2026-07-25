#!/usr/bin/env bash
# Qwen-Image online DPO e2e smoke test (minimal runtime), vllm_omni rollout + OCR reward.
#
# Flow: dummy OCR parquet -> vllm_omni rollout -> genrm_ocr reward (vllm Qwen3-VL) ->
#       online DPO pairing -> ref noise pred -> FSDP LoRA actor update.
#
# Requires: vllm-omni, diffusers>=0.37, Levenshtein,
#   tiny Qwen-Image at ~/models/tiny-random/Qwen-Image
#   tiny qwen3-vl  at ~/models/tiny-random/qwen3-vl
#
# Override via env: NUM_GPUS, MODEL_PATH, REWARD_MODEL_PATH, DATA_DIR, TOTAL_TRAIN_STEPS,
#                   TRAIN_FILES, VAL_FILES
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-2}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/Qwen-Image}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}/tokenizer}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-${HOME}/models/tiny-random/qwen3-vl}
REWARD_TP=${REWARD_TP:-1}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_ocr_diffusion}
TRAIN_FILES=${TRAIN_FILES:-${DATA_DIR}/train.parquet}
VAL_FILES=${VAL_FILES:-${DATA_DIR}/test.parquet}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-1}

ENGINE=vllm_omni
REWARD_ENGINE=vllm
max_prompt_length=256

# Online DPO needs at least two candidates per prompt for win/reject pairing.
n_resp_per_prompt=2
micro_bsz_per_gpu=2
micro_bsz=$((micro_bsz_per_gpu * NUM_GPUS))
mini_bsz=${micro_bsz}
train_batch_size=${mini_bsz}

python3 tests/special_e2e/create_dummy_diffusion_data.py \
    --local_save_dir "${DATA_DIR}" \
    --train_size "${train_batch_size}" \
    --val_size 4
    
python3 -m verl_omni.trainer.main_diffusion \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    algorithm.paired_preference=true \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.tokenizer_path=${TOKENIZER_PATH} \
    actor_rollout_ref.model.algorithm=dpo \
    actor_rollout_ref.model.model_type=diffusion_dpo_model \
    actor_rollout_ref.model.external_lib=verl_omni.pipelines.qwen_image_dpo \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=dpo \
    actor_rollout_ref.actor.diffusion_loss.dpo_beta=100.0 \
    actor_rollout_ref.actor.optim.lr=2e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.calculate_log_probs=false \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.pipeline.height=256 \
    actor_rollout_ref.rollout.pipeline.width=256 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=${max_prompt_length} \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    reward.num_workers=1 \
    reward.reward_model.enable=True \
    reward.reward_model.model_path=${REWARD_MODEL_PATH} \
    reward.reward_model.rollout.name=${REWARD_ENGINE} \
    reward.reward_model.rollout.tensor_model_parallel_size=${REWARD_TP} \
    reward.reward_model.rollout.gpu_memory_utilization=0.4 \
    reward.reward_model.rollout.prompt_length=${max_prompt_length} \
    reward.reward_model.rollout.response_length=32 \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/genrm_ocr.py \
    reward.custom_reward_function.name=compute_score_ocr \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=qwen-image-online-dpo-ocr-e2e \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS} \
    "$@"

echo "Qwen-Image online DPO OCR e2e test passed (training completed successfully)."
