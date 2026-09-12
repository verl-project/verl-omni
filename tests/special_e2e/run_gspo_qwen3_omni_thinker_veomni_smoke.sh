#!/usr/bin/env bash
# Two-step V1 GSPO test, including VeOmni actor-to-vLLM-Omni full-weight sync.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/Qwen3-Omni-VeOmni}
DATA_DIR=${DATA_DIR:-${HOME}/data/math}
export NUM_GPUS=${NUM_GPUS:-2}
export NNODES=1 ACTOR_EP=1 ROLLOUT_TP=2 ATTN_IMPL=sdpa MOE_IMPL=eager

python3 "${REPO_ROOT}/tests/special_e2e/build_qwen3_omni_tiny_random.py" \
    --output-dir "${MODEL_PATH}" --force
if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    python3 "${REPO_ROOT}/tests/special_e2e/create_dummy_math_data.py" --local_save_dir "${DATA_DIR}"
fi

TRAIN_FILE="${DATA_DIR}/train.parquet" VAL_FILE="${DATA_DIR}/test.parquet" \
bash "${REPO_ROOT}/examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_veomni.sh" \
    data.train_batch_size=4 \
    data.max_prompt_length=256 \
    data.max_response_length=128 \
    data.val_max_samples=4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.veomni.param_offload=false \
    actor_rollout_ref.actor.veomni.optimizer_offload=false \
    actor_rollout_ref.ref.veomni.param_offload=false \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.enforce_eager=true \
    reward.custom_reward_function.path=null \
    trainer.val_before_train=false \
    trainer.test_freq=1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps="${TOTAL_TRAIN_STEPS:-2}" \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=qwen3-omni-thinker-veomni-smoke \
    "$@"
