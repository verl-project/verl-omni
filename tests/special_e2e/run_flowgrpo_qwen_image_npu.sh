#!/usr/bin/env bash
# FlowGRPO diffusion e2e smoke test for Ascend NPU, vllm_omni rollout.
set -xeuo pipefail

NUM_NPUS=${NUM_NPUS:-4}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/Qwen-Image}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}/tokenizer}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_diffusion}
dummy_train_path=${TRAIN_FILES:-${DATA_DIR}/train.parquet}
dummy_test_path=${VAL_FILES:-${DATA_DIR}/test.parquet}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-1}

ENGINE=vllm_omni
max_prompt_length=256
n_resp_per_prompt=2
micro_bsz_per_npu=1
micro_bsz=$((micro_bsz_per_npu * NUM_NPUS))
mini_bsz=${micro_bsz}
train_batch_size=$((mini_bsz * n_resp_per_prompt))

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export DEVICE_NAME=npu
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

if [ ! -f "${dummy_train_path}" ] || [ ! -f "${dummy_test_path}" ]; then
    python3 tests/special_e2e/create_dummy_diffusion_data.py \
        --local_save_dir "${DATA_DIR}" \
        --train_size 8 \
        --val_size 4
fi

python3 -m verl_omni.trainer.diffusion.main_flowgrpo \
    trainer.device=npu \
    algorithm.adv_estimator=flow_grpo \
    data.train_files=${dummy_train_path} \
    data.val_files=${dummy_test_path} \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.tokenizer_path=${TOKENIZER_PATH} \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_npu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_npu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.pipeline.height=256 \
    actor_rollout_ref.rollout.pipeline.width=256 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=4.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=${max_prompt_length} \
    actor_rollout_ref.rollout.algo.noise_level=1.0 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,4]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_npu} \
    reward.num_workers=1 \
    reward.reward_model.enable=False \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=flowgrpo-diffusion-npu-e2e \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node=${NUM_NPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS} \
    "$@"

echo "FlowGRPO diffusion NPU e2e test passed (training completed successfully)."