#!/usr/bin/env bash
# Nightly Qwen-Image FlowGRPO regression on local tiny random weights.
set -xeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)

NUM_GPUS=${NUM_GPUS:-4}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/Qwen-Image}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}/tokenizer}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-${HOME}/models/tiny-random/qwen3-vl}
REWARD_TP=${REWARD_TP:-1}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-20}
DEBUG_DUMP_STEPS=${DEBUG_DUMP_STEPS:-1,2}
PERF_SKIP_STEPS=${PERF_SKIP_STEPS:-2}
PERF_THRESHOLD=${PERF_THRESHOLD:-0.10}
PRECISION_ATOL=${PRECISION_ATOL:-1e-4}
PRECISION_RTOL=${PRECISION_RTOL:-1e-3}
PRECISION_MIN_COS_SIM=${PRECISION_MIN_COS_SIM:-0.999}
BOOTSTRAP_MISSING_BASELINE=${BOOTSTRAP_MISSING_BASELINE:-1}
NIGHTLY_DETERMINISTIC_SEED=${NIGHTLY_DETERMINISTIC_SEED:-42}

DATA_DIR=${DATA_DIR:-${HOME}/data/qwen_image_flowgrpo_single_sample}
TRAIN_FILES=${TRAIN_FILES:-${DATA_DIR}/train.parquet}
VAL_FILES=${VAL_FILES:-${DATA_DIR}/test.parquet}
OUTPUT_ROOT=${OUTPUT_ROOT:-${REPO_ROOT}/outputs/debug_dumps}
CURRENT_DUMP_DIR=${CURRENT_DUMP_DIR:-${OUTPUT_ROOT}/current}
BASELINE_DUMP_DIR=${BASELINE_DUMP_DIR:-${OUTPUT_ROOT}/baseline}
LOG_DIR=${LOG_DIR:-${OUTPUT_ROOT}/logs}
CONSOLE_LOG=${CONSOLE_LOG:-${LOG_DIR}/qwen_image_flowgrpo_single_sample.log}
DEBUG_METRICS_JSONL=${DEBUG_METRICS_JSONL:-${CURRENT_DUMP_DIR}/metrics.jsonl}
CURRENT_METRICS_JSON=${CURRENT_METRICS_JSON:-${CURRENT_DUMP_DIR}/metrics.json}
BASELINE_METRICS_JSON=${BASELINE_METRICS_JSON:-${BASELINE_DUMP_DIR}/metrics.json}
DUMP_COMPARE_JSON=${DUMP_COMPARE_JSON:-${CURRENT_DUMP_DIR}/dump_compare.json}

ENGINE=vllm_omni
REWARD_ENGINE=vllm
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-256}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-4}
MICRO_BSZ_PER_GPU=${MICRO_BSZ_PER_GPU:-1}
MICRO_BSZ=$((MICRO_BSZ_PER_GPU * NUM_GPUS))
MINI_BSZ=${MICRO_BSZ}
TRAIN_BATCH_SIZE=$((MINI_BSZ * N_RESP_PER_PROMPT))

ATTN_BACKEND=_flash_3_varlen_hub
ROLLOUT_ATTN_BACKEND=FLASH_ATTN
if ! python3 -c 'from verl_omni.utils.diffusion_attention import fa3_available; raise SystemExit(0 if fa3_available() else 1)' >/dev/null 2>&1; then
    ATTN_BACKEND=native
    ROLLOUT_ATTN_BACKEND=TORCH_SDPA
fi

export NIGHTLY_DETERMINISTIC_SEED
export PYTHONHASHSEED=${PYTHONHASHSEED:-${NIGHTLY_DETERMINISTIC_SEED}}
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export GENRM_OCR_TEMPERATURE=${GENRM_OCR_TEMPERATURE:-0.0}
export GENRM_OCR_TOP_P=${GENRM_OCR_TOP_P:-1.0}
export GENRM_OCR_MAX_TOKENS=${GENRM_OCR_MAX_TOKENS:-32}
export GENRM_OCR_SEED=${GENRM_OCR_SEED:-${NIGHTLY_DETERMINISTIC_SEED}}

rm -rf "${CURRENT_DUMP_DIR}"
mkdir -p "${CURRENT_DUMP_DIR}" "${BASELINE_DUMP_DIR}" "${LOG_DIR}"

python3 "${SCRIPT_DIR}/create_single_sample_data.py" \
    --local_save_dir "${DATA_DIR}" \
    --train_size "${TRAIN_BATCH_SIZE}" \
    --val_size 4

export DEBUG_DUMP_ENABLED=1
export DEBUG_DUMP_DIR="${CURRENT_DUMP_DIR}"
export DEBUG_DUMP_STEPS
export DEBUG_METRICS_JSONL

BOOTSTRAP_ARGS=()
if [ "${BOOTSTRAP_MISSING_BASELINE}" = "1" ]; then
    BOOTSTRAP_ARGS+=(--bootstrap-missing)
fi

python3 "${SCRIPT_DIR}/run.py" \
    algorithm.adv_estimator=flow_grpo \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.shuffle=false \
    data.seed=42 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.tokenizer_path="${TOKENIZER_PATH}" \
    actor_rollout_ref.model.attn_backend=${ATTN_BACKEND} \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.pipeline.height=256 \
    actor_rollout_ref.model.pipeline.width=256 \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BSZ} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.shuffle=false \
    actor_rollout_ref.actor.data_loader_seed=42 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.seed=42 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ_PER_GPU} \
    actor_rollout_ref.rollout.rollout_attn_backend=${ROLLOUT_ATTN_BACKEND} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT} \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.pipeline.height=256 \
    actor_rollout_ref.rollout.pipeline.width=256 \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=${MAX_PROMPT_LENGTH} \
    actor_rollout_ref.rollout.algo.noise_level=1.0 \
    actor_rollout_ref.rollout.algo.sde_type=sde \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,2]" \
    actor_rollout_ref.rollout.algo.sde_window_seed=42 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MICRO_BSZ_PER_GPU} \
    reward.num_workers=$((NUM_GPUS / REWARD_TP)) \
    reward.reward_model.enable=True \
    reward.reward_model.model_path="${REWARD_MODEL_PATH}" \
    reward.reward_model.rollout.name=${REWARD_ENGINE} \
    reward.reward_model.rollout.tensor_model_parallel_size=${REWARD_TP} \
    reward.reward_model.rollout.gpu_memory_utilization=0.4 \
    reward.reward_model.rollout.enforce_eager=True \
    reward.reward_model.rollout.do_sample=False \
    reward.reward_model.rollout.temperature=0.0 \
    reward.reward_model.rollout.prompt_length=${MAX_PROMPT_LENGTH} \
    reward.reward_model.rollout.response_length=32 \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/genrm_ocr.py \
    reward.custom_reward_function.name=compute_score_ocr \
    trainer.logger=console \
    trainer.project_name=verl-nightly \
    trainer.experiment_name=qwen-image-flowgrpo-single-sample \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS} \
    "$@" 2>&1 | tee "${CONSOLE_LOG}"

METRICS_STATUS=0
python3 "${SCRIPT_DIR}/collect_metrics.py" \
    --metrics-jsonl "${DEBUG_METRICS_JSONL}" \
    --log-file "${CONSOLE_LOG}" \
    --baseline "${BASELINE_METRICS_JSON}" \
    --output "${CURRENT_METRICS_JSON}" \
    --perf-skip-steps "${PERF_SKIP_STEPS}" \
    --threshold "${PERF_THRESHOLD}" \
    "${BOOTSTRAP_ARGS[@]}" || METRICS_STATUS=$?

DUMP_STATUS=0
python3 "${SCRIPT_DIR}/compare_dumps.py" \
    --baseline "${BASELINE_DUMP_DIR}" \
    --current "${CURRENT_DUMP_DIR}" \
    --output "${DUMP_COMPARE_JSON}" \
    --atol "${PRECISION_ATOL}" \
    --rtol "${PRECISION_RTOL}" \
    --min-cos-sim "${PRECISION_MIN_COS_SIM}" \
    "${BOOTSTRAP_ARGS[@]}" || DUMP_STATUS=$?

if [ "${METRICS_STATUS}" -ne 0 ] || [ "${DUMP_STATUS}" -ne 0 ]; then
    echo "Qwen-Image FlowGRPO single-sample nightly regression failed: metrics_status=${METRICS_STATUS}, dump_status=${DUMP_STATUS}."
    exit 1
fi

echo "Qwen-Image FlowGRPO single-sample nightly regression passed."
