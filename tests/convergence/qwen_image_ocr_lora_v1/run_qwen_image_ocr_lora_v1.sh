#!/usr/bin/env bash
# L4 precision job: real Qwen-Image + Qwen3-VL weights, real OCR parquet,
# 8 GPUs, 100 FlowGRPO steps on the v1 sync trainer.
#
# Fail closed on:
#   - visible GPU count != 8 (or NUM_GPUS != 8)
#   - rollout_prob_diff_mean > 0.01 (diffusion: rollout_corr/logprob_abs_diff_mean)
#   - non-finite actor/grad_norm
#   - step-100 val-core reward/mean below VAL_REWARD_MIN
# Perf (time_per_step, etc.) is recorded in report.json only — not a gate.
# Val reward is logged every TEST_FREQ steps; all trainer metrics are in metrics.jsonl.
#
# Attention: pin native / TORCH_SDPA (not product-default FA3).
# Reference recipe:
#   examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_v1.sh
set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

WORKSPACE=${WORKSPACE:-$HOME}
NUM_GPUS=${NUM_GPUS:-8}
REQUIRED_GPUS=${REQUIRED_GPUS:-${NUM_GPUS}}
ROLLOUT_TP=${ROLLOUT_TP:-1}
REWARD_TP=${REWARD_TP:-4}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-100}
TEST_FREQ=${TEST_FREQ:-20}

MODEL_PATH=${MODEL_PATH:-${WORKSPACE}/models/Qwen-Image}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-${WORKSPACE}/models/Qwen3-VL-8B-Instruct}
OCR_DATA_DIR=${OCR_DATA_DIR:-${WORKSPACE}/data/ocr/qwen_image}
TRAIN_FILES=${TRAIN_FILES:-${OCR_DATA_DIR}/train.parquet}
VAL_FILES=${VAL_FILES:-${OCR_DATA_DIR}/test.parquet}

ENGINE=vllm_omni
REWARD_ENGINE=vllm
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
REQUEST_BATCH_MAX_WAIT_MS=${REQUEST_BATCH_MAX_WAIT_MS:-10}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-16}
LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-32}

L4_TEST_CASE=${L4_TEST_CASE:-qwen_image_ocr_lora_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-${SCRIPT_DIR}/../outputs/l4_convergence}
CASE_DIR=${CASE_DIR:-${OUTPUT_ROOT}/${L4_TEST_CASE}}
LOG_DIR=${LOG_DIR:-${OUTPUT_ROOT}/logs/${L4_TEST_CASE}}
CONSOLE_LOG=${CONSOLE_LOG:-${LOG_DIR}/qwen_image_ocr_lora_v1.log}
L4_METRICS_JSONL=${L4_METRICS_JSONL:-${CASE_DIR}/metrics.jsonl}
L4_REPORT_JSON=${L4_REPORT_JSON:-${CASE_DIR}/report.json}

SKIP_STEPS=${SKIP_STEPS:-2}
MIN_TRAIN_STEPS=${MIN_TRAIN_STEPS:-${TOTAL_TRAIN_STEPS}}
ROLLOUT_PROB_DIFF_MEAN_MAX=${ROLLOUT_PROB_DIFF_MEAN_MAX:-0.01}
# Floor for val-core/*/reward/mean@* at step TOTAL_TRAIN_STEPS.
VAL_REWARD_MIN=${VAL_REWARD_MIN:-0.9}

count_visible_gpus() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        local csv="${CUDA_VISIBLE_DEVICES// /}"
        if [[ -z "${csv}" ]]; then
            echo 0
            return
        fi
        local IFS=,
        local -a ids=(${csv})
        echo "${#ids[@]}"
        return
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo 0
        return
    fi
    local n
    n=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | grep -c . || true)
    echo "${n:-0}"
}

VISIBLE_GPUS=$(count_visible_gpus)
if [[ "${NUM_GPUS}" -ne "${REQUIRED_GPUS}" ]] || [[ "${VISIBLE_GPUS}" -ne "${REQUIRED_GPUS}" ]]; then
    echo "L4 Qwen-Image OCR LoRA v1 requires exactly ${REQUIRED_GPUS} GPUs (NUM_GPUS=${NUM_GPUS}, visible=${VISIBLE_GPUS})." >&2
    exit 1
fi

if [ ! -f "${TRAIN_FILES}" ] || [ ! -f "${VAL_FILES}" ]; then
    echo "Missing OCR parquet under ${OCR_DATA_DIR} (train=${TRAIN_FILES}, val=${VAL_FILES})." >&2
    exit 1
fi

if [ ! -e "${MODEL_PATH}" ] && [[ "${MODEL_PATH}" == /* ]]; then
    echo "Missing policy weights at ${MODEL_PATH}." >&2
    exit 1
fi
if [ ! -e "${REWARD_MODEL_PATH}" ] && [[ "${REWARD_MODEL_PATH}" == /* ]]; then
    echo "Missing reward weights at ${REWARD_MODEL_PATH}." >&2
    exit 1
fi

if [ $((NUM_GPUS % ROLLOUT_TP)) -ne 0 ]; then
    echo "NUM_GPUS=${NUM_GPUS} must be divisible by ROLLOUT_TP=${ROLLOUT_TP}." >&2
    exit 1
fi
if [ $((NUM_GPUS % REWARD_TP)) -ne 0 ]; then
    echo "NUM_GPUS=${NUM_GPUS} must be divisible by REWARD_TP=${REWARD_TP}." >&2
    exit 1
fi

export PYTHONHASHSEED=${PYTHONHASHSEED:-42}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export GENRM_OCR_TEMPERATURE=${GENRM_OCR_TEMPERATURE:-0.0}
export GENRM_OCR_TOP_P=${GENRM_OCR_TOP_P:-1.0}
export GENRM_OCR_MAX_TOKENS=${GENRM_OCR_MAX_TOKENS:-32}
export GENRM_OCR_SEED=${GENRM_OCR_SEED:-42}
export L4_METRICS_JSONL

rm -rf "${CASE_DIR}"
mkdir -p "${CASE_DIR}" "${LOG_DIR}"

python3 "${SCRIPT_DIR}/run.py" \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=32 \
    data.max_prompt_length=256 \
    data.shuffle=false \
    data.seed=42 \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out','img_mlp.net.0.proj','img_mlp.net.2','txt_mlp.net.0.proj','txt_mlp.net.2']" \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.shuffle=false \
    actor_rollout_ref.actor.data_loader_seed=42 \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.seed=42 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=4.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.algo.noise_level=1.2 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,5]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=50 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs="${MAX_NUM_SEQS}" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.request_batch_max_wait_ms="${REQUEST_BATCH_MAX_WAIT_MS}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE}" \
    reward.num_workers=$((NUM_GPUS / REWARD_TP)) \
    reward.reward_model.enable=True \
    reward.reward_model.model_path="${REWARD_MODEL_PATH}" \
    reward.reward_model.rollout.name="${REWARD_ENGINE}" \
    reward.reward_model.rollout.tensor_model_parallel_size="${REWARD_TP}" \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/genrm_ocr.py \
    reward.custom_reward_function.name=compute_score_ocr \
    trainer.logger=console \
    trainer.project_name=verl-l4 \
    trainer.experiment_name=qwen_image_ocr_lora_v1 \
    trainer.log_val_generations=0 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.resume_mode=disable \
    trainer.total_training_steps="${TOTAL_TRAIN_STEPS}" \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=sync \
    "$@" 2>&1 | tee "${CONSOLE_LOG}"

python3 "${SCRIPT_DIR}/collect_report.py" \
    --metrics-jsonl "${L4_METRICS_JSONL}" \
    --log-file "${CONSOLE_LOG}" \
    --output "${L4_REPORT_JSON}" \
    --skip-steps "${SKIP_STEPS}" \
    --min-train-steps "${MIN_TRAIN_STEPS}" \
    --rollout-prob-diff-mean-max "${ROLLOUT_PROB_DIFF_MEAN_MAX}" \
    --val-reward-min "${VAL_REWARD_MIN}"

echo "L4 Qwen-Image OCR LoRA v1 precision check passed."
