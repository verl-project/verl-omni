#!/usr/bin/env bash
# Qwen3-Omni Thinker GSPO + LoRA training (FSDP + vLLM-Omni AR rollout).
# Hardware: 4× H100 80GB.
#
# Recipe config lives in config/qwen3_omni_thinker_gspo.yaml (inherits verl's
# ppo_trainer). Only volatile values (paths, GPU/node counts) are set here.
set -x

export NCCL_IB_DISABLE=1
export CPATH=/usr/include${CPATH:+:$CPATH}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-Omni-30B-A3B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/math/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/math/test.parquet"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_CONFIG="${SCRIPT_DIR}/qwen3_omni_thinker_only.yaml"

python3 -m verl.trainer.main_ppo \
    --config-path="${SCRIPT_DIR}/config" \
    --config-name=qwen3_omni_thinker_gspo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.external_lib=verl_omni.models.transformers.qwen3_omni_thinker \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_configs_path="${STAGE_CONFIG}" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    "$@"
