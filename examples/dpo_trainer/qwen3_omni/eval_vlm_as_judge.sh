CKPT_ROOT=${CKPT_ROOT:-checkpoints/omni-preference-dpo/qwen3-omni-offline-dpo-lora}
DATA_DIR=${DATA_DIR:-data/omni-preference/parquet_dpo}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-Omni-30B-A3B-Instruct}
OUT_DIR=${OUT_DIR:-outputs/qwen3_omni_judge_eval}
MAX_SAMPLES=60

CUDA_DEVICES=${CUDA_DEVICES:-0}
STEPS=(50 100 150 200)
MODALITIES=(image video audio)

mkdir -p "${OUT_DIR}"

HF_ENABLE_PARALLEL_LOADING=true \
HF_PARALLEL_LOADING_WORKERS=8 \
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" .venv/bin/python examples/dpo_trainer/qwen3_omni/vlm_as_judge.py \
  --data-dir "${DATA_DIR}" \
  --modalities "${MODALITIES[@]}" \
  --output-jsonl "${OUT_DIR}/reference.jsonl" \
  --stage reference \
  --max-samples "${MAX_SAMPLES}" \
  --model-path "${MODEL_PATH}" \
  --device-map cuda

for step in "${STEPS[@]}"; do
  HF_ENABLE_PARALLEL_LOADING=true \
  HF_PARALLEL_LOADING_WORKERS=8 \
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" .venv/bin/python examples/dpo_trainer/qwen3_omni/vlm_as_judge.py \
    --data-dir "${DATA_DIR}" \
    --modalities "${MODALITIES[@]}" \
    --output-jsonl "${OUT_DIR}/global_step_${step}.trained.jsonl" \
    --stage trained \
    --max-samples "${MAX_SAMPLES}" \
    --model-path "${MODEL_PATH}" \
    --device-map cuda \
    --adapter-path "${CKPT_ROOT}/global_step_${step}"
done

for step in "${STEPS[@]}"; do
  HF_ENABLE_PARALLEL_LOADING=true \
  HF_PARALLEL_LOADING_WORKERS=8 \
  .venv/bin/python examples/dpo_trainer/qwen3_omni/vlm_as_judge.py \
    --data-dir "${DATA_DIR}" \
    --modalities "${MODALITIES[@]}" \
    --output-jsonl "${OUT_DIR}/global_step_${step}.jsonl" \
    --summary-json "${OUT_DIR}/global_step_${step}.summary.json" \
    --reference-jsonl "${OUT_DIR}/reference.jsonl" \
    --trained-jsonl "${OUT_DIR}/global_step_${step}.trained.jsonl" \
    --stage judge \
    --max-samples "${MAX_SAMPLES}" \
    --judge-max-tokens 4096 \
    --judge-router-address 127.0.0.1:8001
done
