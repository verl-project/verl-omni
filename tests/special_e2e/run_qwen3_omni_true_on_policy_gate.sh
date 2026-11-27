#!/usr/bin/env bash
# Qwen3-Omni Thinker true on-policy gate (tiny random model, full-parameter, 2 GPUs).
#
# Phase 1: 2 steps with actor_rollout_ref.model.true_on_policy=True; rollout and recomputed
# actor log-probs must be bitwise equal at both steps (the second after a real update and
# weight sync), the update forward must reproduce the recomputed log-probs token for token
# (actor/old_logprobs_mismatch_count), and the update must be finite and nonzero.
# Phase 2: 1 step with algorithm.rollout_correction.bypass_mode=true, which reuses the rollout
# log-probs as the old policy instead of recomputing them. The update forward must then match
# the rollout log-probs token for token. Sampling is not seeded, so the two runs see different
# batches and grad_norm is not compared across them.
#
# GSPO with GRPO advantages and a bf16 actor, as the true on-policy recipe.
# ppo_micro_batch_size_per_gpu equals the per-GPU batch, so there is no gradient accumulation.
#
# The tiny model is shaped for the mode: 128-dim heads (with smaller ones the actor's hub
# FlashAttention-3 kernel and vLLM's paged one disagree past 128 tokens), a vocabulary padded to a
# multiple of 64 like the real checkpoint's (keeps the lm_head GEMM on the same kernels on both
# sides), and 4 decoder layers (vLLM-Omni's hf_numerics mode needs more of them than deepstack levels).
#
# Requires: verl, verl-omni, vllm-omni installed; Hopper GPUs (the rollout picks FlashAttention-3,
# which the actor matches through the hub kernel). On other GPUs the script exits with code 5,
# which the GPU smoke runner reports as skipped.
#
# Override via env: MODEL_PATH, DATA_DIR, OUTPUT_DIR. Arguments are appended to both runs as Hydra overrides.
set -xeuo pipefail

export NCCL_IB_DISABLE=1
export CPATH=/usr/include${CPATH:+:$CPATH}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export VERL_USE_EXTERNAL_MODULES=${VERL_USE_EXTERNAL_MODULES:+${VERL_USE_EXTERNAL_MODULES},}verl_omni

capability=$(python3 -c "import torch; print(torch.cuda.get_device_capability())")
if [ "${capability}" != "(9, 0)" ]; then
    echo "The true on-policy gate needs Hopper GPUs; skipping."
    exit 5
fi

NUM_GPUS=2
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/Qwen3-Omni-true-on-policy}
DATA_DIR=${DATA_DIR:-${HOME}/data/gsm8k}
OUTPUT_DIR=${OUTPUT_DIR:-$(mktemp -d)}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python3 "${REPO_ROOT}/tests/special_e2e/build_qwen3_omni_tiny_random.py" \
    --output-dir "${MODEL_PATH}" --head-dim 128 --num-hidden-layers 4 --vocab-size-multiple 64

if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    python3 "${REPO_ROOT}/tests/special_e2e/create_dummy_math_data.py" \
        --local_save_dir "${DATA_DIR}"
fi

run_train() {
    local name=$1
    shift
    VERL_FILE_LOGGER_PATH="${OUTPUT_DIR}/${name}.jsonl" python3 -m verl_omni.trainer.main_omni \
        data.train_files="${DATA_DIR}/train.parquet" \
        data.val_files="${DATA_DIR}/test.parquet" \
        data.train_batch_size=4 \
        data.max_prompt_length=256 \
        data.max_response_length=256 \
        data.truncation='error' \
        data.filter_overlong_prompts=true \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.model.true_on_policy=True \
        actor_rollout_ref.model.lora_rank=0 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.freeze_vision_tower=true \
        actor_rollout_ref.actor.strategy=fsdp2 \
        actor_rollout_ref.actor.optim.lr=1e-5 \
        actor_rollout_ref.actor.optim.weight_decay=0.0 \
        actor_rollout_ref.actor.optim.clip_grad=1.0 \
        actor_rollout_ref.actor.ppo_mini_batch_size=4 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.actor.use_dynamic_bsz=False \
        actor_rollout_ref.actor.use_kl_loss=false \
        actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
        actor_rollout_ref.actor.clip_ratio_low=3e-4 \
        actor_rollout_ref.actor.clip_ratio_high=4e-4 \
        actor_rollout_ref.actor.clip_ratio_c=10.0 \
        actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
        actor_rollout_ref.actor.fsdp_config.batch_invariant=True \
        actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
        actor_rollout_ref.rollout.name=vllm_omni \
        actor_rollout_ref.rollout.n=2 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.calculate_log_probs=True \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
        actor_rollout_ref.rollout.max_num_seqs=8 \
        actor_rollout_ref.rollout.load_format=safetensors \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.enable_prefix_caching=False \
        +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode="ar" \
        +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="qwen3_omni_moe" \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        reward.reward_manager.source=register \
        reward.reward_manager.name=naive \
        reward.custom_reward_function.path="${REPO_ROOT}/tests/special_e2e/true_on_policy_hash_reward.py" \
        reward.custom_reward_function.name=compute_score \
        trainer.val_before_train=false \
        trainer.balance_batch=False \
        trainer.critic_warmup=0 \
        trainer.logger='["console","file"]' \
        trainer.project_name=verl-test \
        trainer.experiment_name="qwen3-omni-true-on-policy-${name}" \
        trainer.n_gpus_per_node="${NUM_GPUS}" \
        trainer.nnodes=1 \
        trainer.test_freq=-1 \
        trainer.save_freq=-1 \
        trainer.resume_mode=disable \
        "$@"
}

run_train recompute trainer.total_training_steps=2 "$@"
run_train bypass trainer.total_training_steps=1 algorithm.rollout_correction.bypass_mode=true "$@"

python3 - "${OUTPUT_DIR}/recompute.jsonl" "${OUTPUT_DIR}/bypass.jsonl" <<'PY'
import json
import math
import sys


def load(path):
    with open(path) as f:
        return {row["step"]: row["data"] for row in map(json.loads, f)}


recompute, bypass = load(sys.argv[1]), load(sys.argv[2])
for step in (1, 2):
    metrics = recompute[step]
    assert metrics["training/rollout_logprobs_mismatch_count"] == 0, (step, metrics)
    assert metrics["actor/old_logprobs_mismatch_count"] == 0, (step, metrics)
    assert math.isfinite(metrics["actor/grad_norm"]) and metrics["actor/grad_norm"] > 0, (step, metrics["actor/grad_norm"])
assert bypass[1]["actor/old_logprobs_mismatch_count"] == 0, bypass[1]
assert math.isfinite(bypass[1]["actor/grad_norm"]) and bypass[1]["actor/grad_norm"] > 0, bypass[1]["actor/grad_norm"]
print("mismatch_count:", [recompute[step]["training/rollout_logprobs_mismatch_count"] for step in (1, 2)])
print("update-forward mismatch_count recompute/bypass:", [recompute[step]["actor/old_logprobs_mismatch_count"] for step in (1, 2)], bypass[1]["actor/old_logprobs_mismatch_count"])
print("grad_norm recompute/bypass:", recompute[1]["actor/grad_norm"], bypass[1]["actor/grad_norm"])
PY

echo "Qwen3-Omni Thinker true on-policy gate passed."
