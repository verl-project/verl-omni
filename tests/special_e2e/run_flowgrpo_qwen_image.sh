#!/usr/bin/env bash
# FlowGRPO diffusion e2e smoke test (minimal runtime), vllm_omni rollout.
#
# Single pass covering:
#   parquet load -> vllm_omni rollout -> multi-reward (jpeg_compressibility rule
#   reward + vLLM OCR reward model, weighted sum) -> flow_grpo -> FSDP LoRA -> sync.
#
# Requires: vllm-omni, diffusers>=0.37,
#   tiny Qwen-Image at ~/models/tiny-random/Qwen-Image
#   tiny qwen3-vl  at ~/models/tiny-random/qwen3-vl
set -euo pipefail

# Override via env: NUM_GPUS, MODEL_PATH, REWARD_MODEL_PATH, DATA_DIR, TOTAL_TRAIN_STEPS,
#                   TRAIN_FILES, VAL_FILES
NUM_GPUS=${NUM_GPUS:-4}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/Qwen-Image}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}/tokenizer}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-${HOME}/models/tiny-random/qwen3-vl}
REWARD_TP=${REWARD_TP:-1}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_diffusion}
dummy_train_path=${TRAIN_FILES:-${DATA_DIR}/train.parquet}
dummy_test_path=${VAL_FILES:-${DATA_DIR}/test.parquet}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-2}

ENGINE=vllm_omni
max_prompt_length=256

# Smoke: pin local FLASH_ATTN (product default remains FLASH_ATTN_3_HUB).
# Use exit-code checks only — importing verl/vllm may print INFO lines on stdout.
ATTN_BACKEND=_flash_3_varlen_hub
ROLLOUT_ATTN_BACKEND=FLASH_ATTN
if ! python3 -c 'from verl_omni.utils.diffusion_attention import fa3_available; raise SystemExit(0 if fa3_available() else 1)' >/dev/null 2>&1; then
    ATTN_BACKEND=native
    ROLLOUT_ATTN_BACKEND=TORCH_SDPA
fi

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
        echo "FAIL: unexpected vLLMOmniHttpServer process(es) detected on GPU-0 —"
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
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.tokenizer_path=${TOKENIZER_PATH} \
    actor_rollout_ref.model.attn_backend=${ATTN_BACKEND} \
    actor_rollout_ref.rollout.rollout_attn_backend=${ROLLOUT_ATTN_BACKEND} \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
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
    reward.reward_model.enable=True \
    reward.reward_model.model_path=${REWARD_MODEL_PATH} \
    reward.reward_model.rollout.name=vllm \
    reward.reward_model.rollout.tensor_model_parallel_size=${REWARD_TP} \
    reward.reward_model.rollout.gpu_memory_utilization=0.4 \
    reward.reward_model.rollout.prompt_length=${max_prompt_length} \
    reward.reward_model.rollout.response_length=32 \
    reward.custom_reward_function.path=pkg://verl_omni.reward_loop.reward_manager.multi \
    reward.custom_reward_function.name=_multi_reward_placeholder \
    reward.reward_manager.name=MultiVisualRewardManager \
    reward.reward_manager.module.path=pkg://verl_omni.reward_loop.reward_manager \
    "+reward.reward_functions.jpeg.path=pkg://verl_omni.utils.reward_score.jpeg_compressibility" \
    '+reward.reward_functions.jpeg.name=compute_score' \
    '+reward.reward_functions.jpeg.weight=0.5' \
    "+reward.reward_functions.ocr.path=verl_omni/utils/reward_score/genrm_ocr.py" \
    '+reward.reward_functions.ocr.name=compute_score_ocr' \
    '+reward.reward_functions.ocr.weight=0.5' \
    reward.aggregation=weighted_sum \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=flowgrpo-diffusion-e2e \
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

echo "FlowGRPO diffusion e2e test passed (training completed successfully)."
