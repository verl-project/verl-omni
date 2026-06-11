#!/usr/bin/env bash
# Flow-DPPO diffusion e2e smoke test (minimal runtime), vllm_omni rollout.
#
# Covers: parquet load -> vllm_omni rollout -> visual rule reward
#         -> flow_dppo loss -> FSDP LoRA -> sync.
#
# Requires: vllm-omni, diffusers>=0.37, and a tiny Qwen-Image checkpoint.
set -euo pipefail

# Override via env: NUM_GPUS, MODEL_PATH, DATA_DIR, TOTAL_TRAIN_STEPS,
#                   TRAIN_FILES, VAL_FILES, KL_MASK_THRESHOLD
NUM_GPUS=${NUM_GPUS:-4}
DEFAULT_MODEL_PATH=/mnt/andy/models/tiny-random/Qwen-Image
if [[ ! -e "${DEFAULT_MODEL_PATH}" ]]; then
    DEFAULT_MODEL_PATH=${HOME}/models/tiny-random/Qwen-Image
fi
MODEL_PATH=${MODEL_PATH:-${DEFAULT_MODEL_PATH}}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}/tokenizer}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_diffusion}
dummy_train_path=${TRAIN_FILES:-${DATA_DIR}/train.parquet}
dummy_test_path=${VAL_FILES:-${DATA_DIR}/test.parquet}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-2}
KL_MASK_THRESHOLD=${KL_MASK_THRESHOLD:-1e-5}

ENGINE=vllm_omni
max_prompt_length=256

# This helper runs nvidia-smi in a background loop during training and
# fails if any vLLMOmniHttpServer process appears.
_LEAK_FILE=$(mktemp)
_LEAK_PID=""
cleanup_leak_monitor() {
    [[ -n "${_LEAK_PID}" ]] && kill "${_LEAK_PID}" 2>/dev/null || true
    rm -f "${_LEAK_FILE}"
}
trap cleanup_leak_monitor EXIT

start_leak_monitor() {
    : > "${_LEAK_FILE}"
    while true; do
        if nvidia-smi -i 0 2>&1 | grep -q "vLLMOmniHttpServer"; then
            echo "LEAK" >> "${_LEAK_FILE}"
        fi
        sleep 1
    done &
    _LEAK_PID=$!
}

check_leak_monitor() {
    kill "${_LEAK_PID}" 2>/dev/null || true
    _LEAK_PID=""
    if grep -q "LEAK" "${_LEAK_FILE}" 2>/dev/null; then
        echo ""
        echo "FAIL: unexpected vLLMOmniHttpServer process(es) detected on GPU-0"
        ray stop --force 2>/dev/null || true
        exit 1
    fi
}

n_resp_per_prompt=2
micro_bsz_per_gpu=1
micro_bsz=$((micro_bsz_per_gpu * NUM_GPUS))
mini_bsz=${micro_bsz}
train_batch_size=$((mini_bsz * n_resp_per_prompt))

python3 tests/special_e2e/create_dummy_diffusion_data.py \
    --local_save_dir "${DATA_DIR}" \
    --train_size "${train_batch_size}" \
    --val_size 4

start_leak_monitor
python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=${dummy_train_path} \
    data.val_files=${dummy_test_path} \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    algorithm.adv_estimator=flow_dppo \
    actor_rollout_ref.model.algorithm=flow_dppo \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.tokenizer_path=${TOKENIZER_PATH} \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_dppo \
    actor_rollout_ref.actor.diffusion_loss.kl_mask_threshold=${KL_MASK_THRESHOLD} \
    actor_rollout_ref.actor.diffusion_loss.add_kl_coefficient=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    reward.num_workers=1 \
    reward.reward_model.enable=False \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=flowdppo-diffusion-e2e \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS} \
    "$@"
check_leak_monitor

echo "Flow-DPPO diffusion e2e test passed (training completed successfully)."
