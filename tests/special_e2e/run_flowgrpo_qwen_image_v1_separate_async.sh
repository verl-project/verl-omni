#!/usr/bin/env bash
# FlowGRPO diffusion v1 separate_async e2e smoke test (minimal runtime).
#
# Exercises:
#   parquet load -> standalone vllm_omni rollout -> jpeg rule reward ->
#   flow_grpo -> FSDP LoRA -> NCCL checkpoint-engine weight sync to
#   standalone replicas (PolicyGradientDiffusionTrainerV1SeparateAsync).
#
# GPU layout (separate async needs dedicated rollout GPUs):
#   - NUM_GPUS_ACTOR:      colocated actor + colocated rollout
#   - NUM_GPUS_STANDALONE: standalone rollout
#   Total visible GPUs must be >= NUM_GPUS_ACTOR + NUM_GPUS_STANDALONE.
#   Defaults split a 4-GPU allocation 2+2.
#
# Requires: vllm-omni, diffusers>=0.37, TransferQueue, cupy+pyzmq (nccl ckpt),
#   tiny Qwen-Image at ~/models/tiny-random/Qwen-Image
#
# Override via env: NUM_GPUS, NUM_GPUS_ACTOR, NUM_GPUS_STANDALONE, MODEL_PATH,
#                   DATA_DIR, TOTAL_TRAIN_STEPS, CKPT_BACKEND, ...
set -euo pipefail

NUM_GPUS=${NUM_GPUS:-4}
NUM_GPUS_ACTOR=${NUM_GPUS_ACTOR:-$((NUM_GPUS / 2))}
NUM_GPUS_STANDALONE=${NUM_GPUS_STANDALONE:-$((NUM_GPUS - NUM_GPUS_ACTOR))}
if [[ "${NUM_GPUS_ACTOR}" -lt 1 || "${NUM_GPUS_STANDALONE}" -lt 1 ]]; then
    echo "Need at least 1 actor GPU and 1 standalone rollout GPU" \
         "(NUM_GPUS=${NUM_GPUS}, NUM_GPUS_ACTOR=${NUM_GPUS_ACTOR}," \
         "NUM_GPUS_STANDALONE=${NUM_GPUS_STANDALONE})" >&2
    exit 2
fi

MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/Qwen-Image}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}/tokenizer}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_diffusion_v1_separate_async}
dummy_train_path=${TRAIN_FILES:-${DATA_DIR}/train.parquet}
dummy_test_path=${VAL_FILES:-${DATA_DIR}/test.parquet}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-2}
CKPT_BACKEND=${CKPT_BACKEND:-nccl}
ROLLOUT_TP=${ROLLOUT_TP:-1}

# Fully on-policy knobs so the short smoke finishes without waiting on
# off-policy backlog (sync_compatible pauses standalone during actor update).
NUM_WARMUP_BATCHES=${NUM_WARMUP_BATCHES:-0}
PARAMETER_SYNC_STEP=${PARAMETER_SYNC_STEP:-2}
MAX_OFF_POLICY_THRESHOLD=${MAX_OFF_POLICY_THRESHOLD:-1}
MAX_OFF_POLICY_STRATEGY=${MAX_OFF_POLICY_STRATEGY:-drop}
SYNC_COMPATIBLE=${SYNC_COMPATIBLE:-1}

ENGINE=vllm_omni
max_prompt_length=256

ATTN_BACKEND=_flash_3_varlen_hub
ROLLOUT_ATTN_BACKEND=FLASH_ATTN
if ! python3 -c 'from verl_omni.utils.diffusion_attention import fa3_available; raise SystemExit(0 if fa3_available() else 1)' >/dev/null 2>&1; then
    ATTN_BACKEND=native
    ROLLOUT_ATTN_BACKEND=TORCH_SDPA
fi

# Pre-flight: surface missing cupy/pyzmq before Ray spins up.
python3 - <<PY
import importlib
import sys

backend = "${CKPT_BACKEND}"
module_map = {
    "nccl": "verl.checkpoint_engine.nccl_checkpoint_engine",
    "nixl": "verl.checkpoint_engine.nixl_checkpoint_engine",
    "mooncake": "verl.checkpoint_engine.mooncake_checkpoint_engine",
    "hccl": "verl.checkpoint_engine.hccl_checkpoint_engine",
}
mod = module_map.get(backend)
if mod is None:
    sys.exit(0)
try:
    importlib.import_module(mod)
except Exception as e:
    print(f"Checkpoint backend '{backend}' could not be imported: {e}", file=sys.stderr)
    print("Install the required dependency. For 'nccl' that is typically:", file=sys.stderr)
    print("  pip install cupy-cuda12x   # or cupy-cuda13x matching your CUDA major", file=sys.stderr)
    print("  pip install pyzmq", file=sys.stderr)
    sys.exit(1)
PY

# Each outer step consumes PARAMETER_SYNC_STEP complete PPO mini-batches.
n_resp_per_prompt=2
micro_bsz_per_gpu=1
mini_bsz=$((micro_bsz_per_gpu * NUM_GPUS_ACTOR))
train_batch_size=$((PARAMETER_SYNC_STEP * mini_bsz))
# Enough rows for a couple of prompt groups across the short smoke.
synthetic_train_size=$((train_batch_size * TOTAL_TRAIN_STEPS * 2))
if [[ "${synthetic_train_size}" -lt 4 ]]; then
    synthetic_train_size=4
fi

python3 tests/special_e2e/create_dummy_diffusion_data.py \
    --local_save_dir "${DATA_DIR}" \
    --train_size "${synthetic_train_size}" \
    --val_size 4

python3 -m verl_omni.trainer.main_diffusion_v1 \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.nnodes=1 \
    actor_rollout_ref.rollout.n_gpus_per_node=${NUM_GPUS_STANDALONE} \
    actor_rollout_ref.rollout.checkpoint_engine.backend=${CKPT_BACKEND} \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
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
    reward.reward_model.enable=False \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=flowgrpo-diffusion-v1-separate-async-e2e \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node=${NUM_GPUS_ACTOR} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS} \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.num_warmup_batches=${NUM_WARMUP_BATCHES} \
    trainer.v1.separate_async.parameter_sync_step=${PARAMETER_SYNC_STEP} \
    trainer.v1.separate_async.sync_compatible=${SYNC_COMPATIBLE} \
    trainer.v1.sampler.max_off_policy_threshold=${MAX_OFF_POLICY_THRESHOLD} \
    trainer.v1.sampler.max_off_policy_strategy=${MAX_OFF_POLICY_STRATEGY} \
    "$@"

echo "FlowGRPO diffusion v1 separate_async e2e test passed (training completed successfully)."
