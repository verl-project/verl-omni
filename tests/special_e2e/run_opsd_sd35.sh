#!/usr/bin/env bash
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# DiffusionOPSD e2e smoke test (minimal runtime), v1 sync trainer.
#
# Covers: tiny SD3 load -> deterministic vllm_omni rollout with the OPSD rollout
#         adapter (keeps the trailing low-noise states) -> endpoint reward scored
#         by the tiny PickScore checkpoint -> opsd target construction in the
#         `diffusion_opsd_model` engine (old-adapter anchor, differentiable
#         PickScore through the VAE decoder, relative trust region) -> OPSDLoss
#         fit -> old-adapter EMA refresh.
#
# Requires: vllm-omni, diffusers
#   Builds offline if missing:
#     ~/models/tiny-random/stable-diffusion-3-tiny-random
#     ~/models/tiny-random/PickScore
#
# Override via env: NUM_GPUS, MODEL_PATH, PICKSCORE_PATH, DATA_DIR, TOTAL_TRAIN_STEPS
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

NUM_GPUS=${NUM_GPUS:-2}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/stable-diffusion-3-tiny-random}
PICKSCORE_PATH=${PICKSCORE_PATH:-${HOME}/models/tiny-random/PickScore}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_opsd}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-2}
max_prompt_length=256
ENGINE=vllm_omni

ATTN_BACKEND=native
ROLLOUT_ATTN_BACKEND=TORCH_SDPA

n_resp_per_prompt=${N_RESP_PER_PROMPT:-2}
micro_bsz_per_gpu=${MICRO_BSZ_PER_GPU:-1}
train_batch_size=${TRAIN_BATCH_SIZE:-2}
mini_bsz=${train_batch_size}

python3 tests/special_e2e/build_sd3_tiny_random.py --output-dir "${MODEL_PATH}"
# The differentiable target runtime preprocesses latents with the fixed CLIP geometry
# (224x224), so the tiny reward checkpoint must use the same image size.
python3 tests/special_e2e/build_pickscore_tiny_random.py \
    --output-dir "${PICKSCORE_PATH}" \
    --image-size 224 \
    --force
python3 tests/special_e2e/create_dummy_diffusion_data.py \
    --local_save_dir "${DATA_DIR}" \
    --train_size "${train_batch_size}" \
    --val_size 2 \
    --data_sources prompt

export PICKSCORE_PATH

custom_chat_template='{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}{{ message['\''content'\''] }}{% endif %}{% endfor %}'

python3 -m verl_omni.trainer.main_diffusion_v1 \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/test.parquet" \
    data.train_batch_size=${train_batch_size} \
    data.val_max_samples=2 \
    data.max_prompt_length=${max_prompt_length} \
    data.truncation=error \
    data.seed=42 \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    algorithm.global_std=True \
    algorithm.old_policy_decay_schedule=linear_to_0_5 \
    algorithm.old_policy_update_interval=1 \
    actor_rollout_ref.model.algorithm=opsd \
    actor_rollout_ref.model.model_type=diffusion_opsd_model \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.custom_chat_template="\"${custom_chat_template}\"" \
    'actor_rollout_ref.model.extra_tokenizers={clip: {path: tokenizer, max_length: 77}, t5: {path: tokenizer_3, max_length: 256}}' \
    actor_rollout_ref.model.attn_backend=${ATTN_BACKEND} \
    actor_rollout_ref.rollout.rollout_attn_backend=${ROLLOUT_ATTN_BACKEND} \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out']" \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=opsd \
    actor_rollout_ref.actor.diffusion_loss.opsd_rho=0.10 \
    actor_rollout_ref.actor.diffusion_loss.opsd_n_ascent=2 \
    actor_rollout_ref.actor.diffusion_loss.opsd_eta=1.0 \
    actor_rollout_ref.actor.diffusion_loss.opsd_branch_beta=1.0 \
    actor_rollout_ref.actor.diffusion_loss.opsd_query_sigma=0.278 \
    actor_rollout_ref.actor.diffusion_loss.opsd_query_sigma_tolerance=0.15 \
    actor_rollout_ref.actor.diffusion_loss.opsd_eps_gamma=1e-5 \
    actor_rollout_ref.actor.diffusion_loss.opsd_reward_kind=pickscore \
    actor_rollout_ref.actor.diffusion_loss.opsd_reward_model_path="${PICKSCORE_PATH}" \
    actor_rollout_ref.actor.diffusion_loss.opsd_target_micro_batch=2 \
    actor_rollout_ref.actor.optim.lr=2e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_no_sync_for_gradient_accumulation=true \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.rollout_adapter=old \
    actor_rollout_ref.rollout.pipeline.height=256 \
    actor_rollout_ref.rollout.pipeline.width=256 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=${max_prompt_length} \
    actor_rollout_ref.rollout.max_prompt_embed_length=333 \
    actor_rollout_ref.rollout.algo.noise_level=0.0 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    reward.num_workers=1 \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path=tests/special_e2e/opsd_pickscore_reward.py \
    reward.custom_reward_function.name=compute_score_pickscore \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=sd35-opsd-e2e \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS} \
    trainer.total_epochs=100 \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=sync \
    trainer.v1.sampler.drop_incomplete_groups=True \
    "$@"

echo "DiffusionOPSD e2e test passed (training completed successfully)."
