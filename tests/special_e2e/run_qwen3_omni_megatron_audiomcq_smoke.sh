#!/usr/bin/env bash
# 4 GPUs: 2 Megatron actor GPUs + one standalone TP2 vLLM-Omni replica.
# Structural coverage only: random-model accuracy/nonzero reward is not a gate.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPO_ROOT}"
SMOKE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/audiomcq-smoke.XXXXXX")
export MODEL_PATH="${SMOKE_DIR}/model"
export TRAIN_FILE="${SMOKE_DIR}/data/train.parquet"
export VAL_FILE="${SMOKE_DIR}/data/validation.parquet"
export OUTPUT_DIR=${OUTPUT_DIR:-${SMOKE_DIR}/outputs}
"${PYTHON:-python3}" tests/special_e2e/build_qwen3_omni_multimodal_tiny_random.py --output-dir "${MODEL_PATH}"
"${PYTHON:-python3}" -m tests.special_e2e.build_audiomcq_smoke_data --output-dir "${SMOKE_DIR}/data"
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_megatron_audiomcq_separate_async.sh \
  trainer.nnodes=1 trainer.n_gpus_per_node=2 \
  actor_rollout_ref.rollout.nnodes=1 actor_rollout_ref.rollout.n_gpus_per_node=2 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1 \
  actor_rollout_ref.actor.megatron.expert_model_parallel_size=1 \
  actor_rollout_ref.actor.megatron.sequence_parallel=false \
  actor_rollout_ref.actor.megatron.param_offload=false \
  actor_rollout_ref.actor.megatron.optimizer_offload=false \
  actor_rollout_ref.actor.megatron.grad_offload=false \
  data.train_batch_size=4 actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  trainer.v1.separate_async.parameter_sync_step=2 \
  actor_rollout_ref.rollout.top_k=1 \
  actor_rollout_ref.rollout.n=2 actor_rollout_ref.rollout.max_num_seqs=4 \
  actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
  actor_rollout_ref.rollout.agent.num_workers=2 reward.num_workers=1 \
  data.dataloader_num_workers=0 actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
  actor_rollout_ref.rollout.enforce_eager=true \
  data.max_prompt_length=256 data.max_response_length=32 \
  actor_rollout_ref.rollout.max_model_len=288 \
  data.val_max_samples=2 trainer.total_training_steps=4 trainer.test_freq=2 \
  "$@"
