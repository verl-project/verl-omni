#!/usr/bin/env bash
# FlowGRPO BAGEL PickScore LoRA e2e smoke test (minimal runtime), vllm_omni rollout.
#
# Covers: Bagel NonDiffusers load -> vllm_omni BagelPipeline rollout ->
# PickScore reward -> flow_grpo LoRA on *_moe_gen -> FSDP sync.
#
# Requires: vllm-omni
#   Builds offline if missing:
#     ~/models/tiny-random/BAGEL-MoT
#     ~/models/tiny-random/PickScore
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

NUM_GPUS=${NUM_GPUS:-4}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/BAGEL-MoT}
PICKSCORE_PATH=${PICKSCORE_PATH:-${HOME}/models/tiny-random/PickScore}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_bagel_pickscore}
dummy_train_path=${TRAIN_FILES:-${DATA_DIR}/train.parquet}
dummy_test_path=${VAL_FILES:-${DATA_DIR}/test.parquet}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-2}
BAGEL_DEPLOY_CONFIG=${BAGEL_DEPLOY_CONFIG:-"${REPO_ROOT}/examples/flowgrpo_trainer/bagel/bagel_deploy_config.yaml"}

ENGINE=vllm_omni
max_prompt_length=64

# Smoke: prefer FA3 when available; fall back like the Qwen-Image e2e.
ATTN_BACKEND=_flash_3_varlen_hub
ROLLOUT_ATTN_BACKEND=FLASH_ATTN
if ! python3 -c 'from verl_omni.utils.diffusion_attention import fa3_available; raise SystemExit(0 if fa3_available() else 1)' >/dev/null 2>&1; then
    ATTN_BACKEND=native
    ROLLOUT_ATTN_BACKEND=TORCH_SDPA
fi

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

python3 tests/special_e2e/build_bagel_tiny_random.py --output-dir "${MODEL_PATH}"
python3 tests/special_e2e/build_pickscore_tiny_random.py --output-dir "${PICKSCORE_PATH}"
python3 tests/special_e2e/create_dummy_bagel_pickscore_data.py \
    --local_save_dir "${DATA_DIR}" \
    --model_path "${MODEL_PATH}" \
    --train_size "${train_batch_size}" \
    --val_size 4 \
    --max_prompt_length "${max_prompt_length}"

export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export PICKSCORE_PATH

start_leak_monitor
python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=${dummy_train_path} \
    data.val_files=${dummy_test_path} \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.trust_remote_code=True \
    algorithm.global_std=False \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.tokenizer_path=${MODEL_PATH} \
    +actor_rollout_ref.model.architecture=OmniBagelForConditionalGeneration \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.attn_backend=${ATTN_BACKEND} \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.target_modules="['q_proj_moe_gen','k_proj_moe_gen','v_proj_moe_gen','o_proj_moe_gen','mlp_moe_gen.gate_proj','mlp_moe_gen.up_proj','mlp_moe_gen.down_proj']" \
    actor_rollout_ref.model.fsdp_layer_prefixes="['layers.']" \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.rollout_attn_backend=${ROLLOUT_ATTN_BACKEND} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.pipeline.height=256 \
    actor_rollout_ref.rollout.pipeline.width=256 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=${max_prompt_length} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.algo.noise_level=1.0 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,4]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.deploy_config=${BAGEL_DEPLOY_CONFIG} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    reward.num_workers=1 \
    reward.custom_reward_function.path=tests/special_e2e/bagel_pickscore_reward.py \
    reward.custom_reward_function.name=compute_score_pickscore \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=flowgrpo-bagel-pickscore-e2e \
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

echo "FlowGRPO BAGEL PickScore LoRA e2e test passed (training completed successfully)."
