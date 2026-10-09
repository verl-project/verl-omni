#!/usr/bin/env bash
# DanceGRPO Wan2.2 text-to-video e2e smoke test for Ascend NPU.
set -xeuo pipefail

NUM_NPUS=${NUM_NPUS:-8}
MODEL_PATH=${MODEL_PATH:-${HOME}/.cache/models/tiny-random/Wan2.2-TI2V-5B-Diffusers}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_diffusion}
TRAIN_FILES=${TRAIN_FILES:-${DATA_DIR}/train.parquet}
VAL_FILES=${VAL_FILES:-${DATA_DIR}/test.parquet}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-1}
max_prompt_length=${MAX_PROMPT_LENGTH:-256}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-2}
micro_bsz_per_npu=${MICRO_BATCH_SIZE_PER_NPU:-1}
mini_bsz=$((micro_bsz_per_npu * NUM_NPUS))
train_batch_size=$((mini_bsz * n_resp_per_prompt))

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export DEVICE_NAME=npu
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export VERL_DATAPROTO_SERIALIZATION_METHOD=numpy
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

set +u
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u

if [[ ! -d "${MODEL_PATH}" ]]; then
    python3 tests/special_e2e/build_wan22_tiny_random.py \
        --output-dir "${MODEL_PATH}" \
        --verify-load
fi

if [[ ! -f "${TRAIN_FILES}" || ! -f "${VAL_FILES}" ]]; then
    python3 tests/special_e2e/create_dummy_diffusion_data.py \
        --local_save_dir "${DATA_DIR}" --train_size "${train_batch_size}" --val_size 4
fi

python3 -m verl_omni.trainer.main_diffusion \
    trainer.use_v1=false trainer.device=npu algorithm.adv_estimator=dance_grpo \
    actor_rollout_ref.model.algorithm=dance_grpo \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=dance_grpo \
    data.train_files="${TRAIN_FILES}" data.val_files="${VAL_FILES}" \
    data.train_batch_size="${train_batch_size}" data.max_prompt_length="${max_prompt_length}" data.seed=42 \
    actor_rollout_ref.model.path="${MODEL_PATH}" actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.model.custom_chat_template='"{% if messages %}{% for message in messages %}{% if message[\"role\"] == \"user\" %}{{ message[\"content\"] }}{% endif %}{% endfor %}{% endif %}</s>"' \
    actor_rollout_ref.actor.optim.lr=1e-5 actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${mini_bsz}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${micro_bsz_per_npu}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=10000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${micro_bsz_per_npu}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_omni actor_rollout_ref.rollout.n="${n_resp_per_prompt}" \
    actor_rollout_ref.rollout.seed=42 actor_rollout_ref.rollout.agent.num_workers="${NUM_NPUS}" \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.load_format=safetensors actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=5.0 \
    actor_rollout_ref.rollout.pipeline.height=256 actor_rollout_ref.rollout.pipeline.width=256 \
    actor_rollout_ref.rollout.pipeline.num_frames=4 \
    +actor_rollout_ref.rollout.pipeline.output_type=np \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=5.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length="${max_prompt_length}" \
    actor_rollout_ref.rollout.algo.noise_level=1.2 actor_rollout_ref.rollout.algo.sde_type=dance_sde \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range='[0,4]' \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${micro_bsz_per_npu}" \
    reward.num_workers=1 reward.reward_model.enable=False \
    reward.custom_reward_function.path=tests/special_e2e/wan22_dummy_reward.py \
    reward.custom_reward_function.name=compute_score \
    trainer.logger=console trainer.project_name=verl-test \
    trainer.experiment_name=dancegrpo-wan22-npu-e2e trainer.log_val_generations=0 \
    trainer.val_before_train=False trainer.n_gpus_per_node="${NUM_NPUS}" trainer.nnodes=1 \
    trainer.save_freq=-1 trainer.test_freq=-1 trainer.resume_mode=disable \
    trainer.total_training_steps="${TOTAL_TRAIN_STEPS}" "$@"

echo "DanceGRPO Wan2.2 NPU e2e test passed (training completed successfully)."
