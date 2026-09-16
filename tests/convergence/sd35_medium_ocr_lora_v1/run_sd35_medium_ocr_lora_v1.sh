#!/usr/bin/env bash
# L4 precision job: real SD3.5-Medium + Qwen2.5-VL GenRM OCR weights, real OCR parquet,
# 8 GPUs (6 actor+rollout, 2 reward), 100 FlowGRPO steps on the v1 sync trainer.
#
# Route A (memory-first sync): TP-sharded rollout/reward, FSDP offload, smaller
# micro-batches, and a larger train batch + rollout.n for faster convergence.
#
# Fail closed on:
#   - visible GPU count != 8 (NUM_GPUS_ACTOR_ROLLOUT + NUM_GPUS_REWARD)
#   - rollout_prob_diff_mean > ROLLOUT_PROB_DIFF_MEAN_MAX (default 0.01)
#   - non-finite actor/grad_norm
#   - step-100 val-core reward/mean below VAL_REWARD_MIN
# Perf (time_per_step, etc.) is recorded in report.json only — not a gate.
#
# Attention: pin native / TORCH_SDPA (not product-default FA3).
# Reference recipe:
#   examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora_v1.sh
set -xeuo pipefail
export FLASHINFER_DISABLE_VERSION_CHECK=1
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

WORKSPACE=${OCR_WORKSPACE:-${WORKSPACE:-$HOME}}
NUM_GPUS_ACTOR_ROLLOUT=${NUM_GPUS_ACTOR_ROLLOUT:-6}
NUM_GPUS_REWARD=${NUM_GPUS_REWARD:-2}
REQUIRED_GPUS=${REQUIRED_GPUS:-$((NUM_GPUS_ACTOR_ROLLOUT + NUM_GPUS_REWARD))}
ROLLOUT_TP=${ROLLOUT_TP:-2}
REWARD_TP=${REWARD_TP:-2}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-100}
TEST_FREQ=${TEST_FREQ:-20}
IMAGE_RESOLUTION=${IMAGE_RESOLUTION:-384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
ROLLOUT_N=${ROLLOUT_N:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-4}
LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-4}
REWARD_GPU_MEMORY_UTILIZATION=${REWARD_GPU_MEMORY_UTILIZATION:-0.85}

MODEL_PATH=${MODEL_PATH:-stabilityai/stable-diffusion-3.5-medium}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}
OCR_DATA_DIR=${OCR_DATA_DIR:-${WORKSPACE}/data/ocr/sd3}
TRAIN_FILES=${TRAIN_FILES:-${OCR_DATA_DIR}/train.parquet}
VAL_FILES=${VAL_FILES:-${OCR_DATA_DIR}/test.parquet}

ENGINE=vllm_omni
REWARD_ENGINE=vllm
ATTN_BACKEND=native
ROLLOUT_ATTN_BACKEND=TORCH_SDPA
CUSTOM_CHAT_TEMPLATE='{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}{{ message['\''content'\''] }}{% endif %}{% endfor %}'

L4_TEST_CASE=${L4_TEST_CASE:-sd35_medium_ocr_lora_v1}
OUTPUT_ROOT=${OUTPUT_ROOT:-${SCRIPT_DIR}/../outputs/l4_convergence}
CASE_DIR=${CASE_DIR:-${OUTPUT_ROOT}/${L4_TEST_CASE}}
LOG_DIR=${LOG_DIR:-${OUTPUT_ROOT}/logs/${L4_TEST_CASE}}
CONSOLE_LOG=${CONSOLE_LOG:-${LOG_DIR}/sd35_medium_ocr_lora_v1.log}
L4_METRICS_JSONL=${L4_METRICS_JSONL:-${CASE_DIR}/metrics.jsonl}
L4_REPORT_JSON=${L4_REPORT_JSON:-${CASE_DIR}/report.json}

SKIP_STEPS=${SKIP_STEPS:-2}
MIN_TRAIN_STEPS=${MIN_TRAIN_STEPS:-${TOTAL_TRAIN_STEPS}}
ROLLOUT_PROB_DIFF_MEAN_MAX=${ROLLOUT_PROB_DIFF_MEAN_MAX:-0.01}
# Floor for val-core/*/reward/mean@* at step TOTAL_TRAIN_STEPS.
VAL_REWARD_MIN=${VAL_REWARD_MIN:-0.6}

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

print_l4_thresholds() {
    cat <<EOF
================================================================================
[L4] sd35_medium_ocr_lora_v1 precision gates (Route A: 8-GPU sync)
================================================================================
  GPUs required           : ${REQUIRED_GPUS} (${NUM_GPUS_ACTOR_ROLLOUT} actor+rollout + ${NUM_GPUS_REWARD} reward)
  Rollout TP / Reward TP  : ${ROLLOUT_TP} / ${REWARD_TP}
  Train batch / rollout.n : ${TRAIN_BATCH_SIZE} / ${ROLLOUT_N}
  PPO mini / micro (GPU)  : ${PPO_MINI_BATCH_SIZE} / ${PPO_MICRO_BATCH_SIZE}
  max_num_seqs            : ${MAX_NUM_SEQS}
  FSDP offload            : param + optimizer enabled
  Training steps (gate)   : ${MIN_TRAIN_STEPS}
  Warmup skip steps       : ${SKIP_STEPS} (gates apply to steps > ${SKIP_STEPS})
  Train-infer gap metric  : training/rollout_probs_diff_mean
                            or rollout_corr/logprob_abs_diff_mean
  Train-infer gap cap     : ${ROLLOUT_PROB_DIFF_MEAN_MAX} (every post-warmup step must be <= cap)
  Grad norm gate          : actor/grad_norm must be finite (no NaN/Inf)
  Val reward gate         : val-core/*/reward/mean@* at step ${TOTAL_TRAIN_STEPS}
  Val reward floor        : ${VAL_REWARD_MIN}
  Recorded only (no gate) : perf/time_per_step, timing_s/*, perf/throughput
================================================================================
EOF
}

if [ "${TRAIN_BATCH_SIZE}" -lt "${PPO_MINI_BATCH_SIZE}" ] || [ $((TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE)) -ne 0 ]; then
    echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} must be >= and divisible by PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE}." >&2
    exit 1
fi

VISIBLE_GPUS=$(count_visible_gpus)
if [[ "${REQUIRED_GPUS}" -ne "${VISIBLE_GPUS}" ]]; then
    echo "L4 SD3.5 OCR LoRA v1 requires exactly ${REQUIRED_GPUS} GPUs (visible=${VISIBLE_GPUS})." >&2
    exit 1
fi

if [ ! -f "${TRAIN_FILES}" ] || [ ! -f "${VAL_FILES}" ]; then
    echo "Missing OCR parquet under ${OCR_DATA_DIR} (train=${TRAIN_FILES}, val=${VAL_FILES})." >&2
    echo "Prepare data with: python3 examples/flowgrpo_trainer/data_process/sd3_ocr.py --output_dir ${OCR_DATA_DIR}" >&2
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

if [ $((NUM_GPUS_ACTOR_ROLLOUT % ROLLOUT_TP)) -ne 0 ]; then
    echo "NUM_GPUS_ACTOR_ROLLOUT=${NUM_GPUS_ACTOR_ROLLOUT} must be divisible by ROLLOUT_TP=${ROLLOUT_TP}." >&2
    exit 1
fi
if [ $((NUM_GPUS_REWARD % REWARD_TP)) -ne 0 ]; then
    echo "NUM_GPUS_REWARD=${NUM_GPUS_REWARD} must be divisible by REWARD_TP=${REWARD_TP}." >&2
    exit 1
fi

export PYTHONHASHSEED=${PYTHONHASHSEED:-42}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export GENRM_OCR_TEMPERATURE=${GENRM_OCR_TEMPERATURE:-0.0}
export GENRM_OCR_TOP_P=${GENRM_OCR_TOP_P:-1.0}
export GENRM_OCR_MAX_TOKENS=${GENRM_OCR_MAX_TOKENS:-32}
export GENRM_OCR_SEED=${GENRM_OCR_SEED:-42}
export L4_METRICS_JSONL

print_l4_thresholds

rm -rf "${CASE_DIR}"
mkdir -p "${CASE_DIR}" "${LOG_DIR}"

python3 "${SCRIPT_DIR}/run.py" \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_max_samples=32 \
    data.max_prompt_length=512 \
    data.truncation=error \
    data.shuffle=false \
    data.seed=42 \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.custom_chat_template="\"${CUSTOM_CHAT_TEMPLATE}\"" \
    'actor_rollout_ref.model.extra_tokenizers={clip: {path: tokenizer, max_length: 77}, t5: {path: tokenizer_3, max_length: 256}}' \
    actor_rollout_ref.model.attn_backend="${ATTN_BACKEND}" \
    actor_rollout_ref.rollout.rollout_attn_backend="${ROLLOUT_ATTN_BACKEND}" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out']" \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.seed=42 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.rollout.pipeline.height="${IMAGE_RESOLUTION}" \
    actor_rollout_ref.rollout.pipeline.width="${IMAGE_RESOLUTION}" \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.max_prompt_embed_length=333 \
    actor_rollout_ref.rollout.algo.noise_level=0.8 \
    actor_rollout_ref.rollout.algo.sde_type="cps" \
    actor_rollout_ref.rollout.algo.sde_window_size=3 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,5]" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs="${MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=28 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE}" \
    reward.num_workers=$((NUM_GPUS_REWARD / REWARD_TP)) \
    reward.reward_model.enable=True \
    reward.reward_model.model_path="${REWARD_MODEL_PATH}" \
    reward.reward_model.rollout.name="${REWARD_ENGINE}" \
    reward.reward_model.enable_resource_pool=True \
    reward.reward_model.nnodes=1 \
    reward.reward_model.n_gpus_per_node="${NUM_GPUS_REWARD}" \
    reward.reward_model.rollout.gpu_memory_utilization="${REWARD_GPU_MEMORY_UTILIZATION}" \
    reward.reward_model.rollout.free_cache_engine=False \
    reward.reward_model.rollout.tensor_model_parallel_size="${REWARD_TP}" \
    reward.reward_model.rollout.enforce_eager=False \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/genrm_ocr.py \
    reward.custom_reward_function.name=compute_score_ocr \
    trainer.logger=console \
    trainer.project_name=verl-l4 \
    trainer.experiment_name=sd35_medium_ocr_lora_v1 \
    trainer.log_val_generations=0 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node="${NUM_GPUS_ACTOR_ROLLOUT}" \
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

echo "L4 SD3.5 OCR LoRA v1 precision check passed."
