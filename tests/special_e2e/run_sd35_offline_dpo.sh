#!/usr/bin/env bash
# SD3.5 offline DPO e2e smoke test (minimal runtime).
#
# Flow: construct parquet (precomputed latents/embeddings) -> offline DPO dataset ->
#       DirectPreferenceRayTrainer -> FSDP LoRA on actor-only workers (no rollout/reward).
#
# Requires: diffusers
# Builds a tiny local SD3 checkpoint fully offline (random weights, no Hub access)
# at ~/models/tiny-random/stable-diffusion-3-tiny-random unless MODEL_PATH is overridden.
#
# Override via env: NUM_GPUS, MODEL_PATH, DATA_DIR, DATA_FILE, TOTAL_TRAIN_STEPS, NUM_PAIRS
set -xeuo pipefail

NUM_GPUS=${NUM_GPUS:-1}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/stable-diffusion-3-tiny-random}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_offline_dpo}
DATA_FILE=${DATA_FILE:-${DATA_DIR}/smoke.parquet}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-1}
NUM_PAIRS=${NUM_PAIRS:-2}

python3 tests/special_e2e/build_sd3_tiny_random.py --output-dir "${MODEL_PATH}"

train_batch_size=${NUM_PAIRS}
# Each parquet row expands to [win, lose]; collated batch size is 2 * NUM_PAIRS.
ppo_mini_batch_size=${NUM_PAIRS}
ppo_micro_batch_size_per_gpu=$((NUM_PAIRS * 2))

custom_chat_template='{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}{{ message['\''content'\''] }}{% endif %}{% endfor %}'

python3 tests/special_e2e/create_dummy_offline_dpo_data.py \
    --local_save_dir "${DATA_DIR}" \
    --output_file "${DATA_FILE}" \
    --num_pairs "${NUM_PAIRS}" \
    --model_path "${MODEL_PATH}" \
    --height 256 \
    --width 256 \
    --guidance_scale 4.0 \
    --max_sequence_length 256

python3 -m verl_omni.trainer.main_diffusion \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=offline \
    algorithm.paired_preference=true \
    data.train_files="${DATA_FILE}" \
    data.val_files="${DATA_FILE}" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=256 \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.offline_dpo_dataset \
    data.custom_cls.name=OfflineDPODataset \
    data.custom_cls.collate_fn=offline_dpo_collate_fn \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.algorithm=dpo \
    actor_rollout_ref.model.model_type=diffusion_dp_model \
    actor_rollout_ref.model.custom_chat_template="\"${custom_chat_template}\"" \
    actor_rollout_ref.model.external_lib=verl_omni.pipelines.sd3_dpo \
    actor_rollout_ref.model.pipeline.guidance_scale=4.0 \
    actor_rollout_ref.model.pipeline.height=256 \
    actor_rollout_ref.model.pipeline.width=256 \
    actor_rollout_ref.model.pipeline.num_inference_steps=4 \
    actor_rollout_ref.model.pipeline.max_sequence_length=256 \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out']" \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=dpo \
    actor_rollout_ref.actor.diffusion_loss.dpo_beta=100.0 \
    actor_rollout_ref.actor.optim.lr=2e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    trainer.resume_mode=disable \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=sd35-offline-dpo-e2e \
    trainer.val_before_train=false \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS} \
    "$@"

echo "SD3.5 offline DPO e2e test passed (training completed successfully)."
