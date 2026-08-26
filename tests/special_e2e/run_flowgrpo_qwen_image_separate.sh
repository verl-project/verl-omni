#!/usr/bin/env bash
# FlowGRPO synchronous separate diffusion e2e smoke test (minimal runtime).
#
# Exercises the pinned repository stack end to end:
#   parquet load -> standalone vllm_omni rollout -> jpeg rule reward ->
#   flow_grpo -> FSDP full finetuning -> NCCL checkpoint-engine weight sync.
#
# GPU layout:
#   - NUM_GPUS_ACTOR:      pure Actor workers
#   - NUM_GPUS_STANDALONE: standalone rollout workers
#   Total visible GPUs must be >= NUM_GPUS_ACTOR + NUM_GPUS_STANDALONE.
#   Defaults split the 4-GPU diffusion smoke allocation 2+2.
#
# Requires: vllm-omni, diffusers>=0.37, cupy+pyzmq (NCCL checkpoint engine),
#   tiny Qwen-Image at ~/models/tiny-random/Qwen-Image.
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
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_diffusion_separate}
dummy_train_path=${TRAIN_FILES:-${DATA_DIR}/train.parquet}
dummy_test_path=${VAL_FILES:-${DATA_DIR}/test.parquet}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-2}
CKPT_BACKEND=${CKPT_BACKEND:-nccl}
ROLLOUT_TP=${ROLLOUT_TP:-1}

ENGINE=vllm_omni
max_prompt_length=256

ATTN_BACKEND=_flash_3_varlen_hub
ROLLOUT_ATTN_BACKEND=FLASH_ATTN
if ! python3 -c 'from verl_omni.utils.diffusion_attention import fa3_available; raise SystemExit(0 if fa3_available() else 1)' >/dev/null 2>&1; then
    ATTN_BACKEND=native
    ROLLOUT_ATTN_BACKEND=TORCH_SDPA
fi

# Surface a missing checkpoint backend dependency before Ray workers start.
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
module = module_map.get(backend)
if module is not None:
    try:
        importlib.import_module(module)
    except Exception as error:
        print(f"Checkpoint backend '{backend}' could not be imported: {error}", file=sys.stderr)
        sys.exit(1)
PY

n_resp_per_prompt=2
micro_bsz_per_gpu=1
mini_bsz=$((micro_bsz_per_gpu * NUM_GPUS_ACTOR))
train_batch_size=$((mini_bsz * n_resp_per_prompt))
synthetic_train_size=$((train_batch_size * TOTAL_TRAIN_STEPS))

python3 tests/special_e2e/create_dummy_diffusion_data.py \
    --local_save_dir "${DATA_DIR}" \
    --train_size "${synthetic_train_size}" \
    --val_size 4

# Let VisualRewardManager dispatch the data source to the JPEG scorer.
python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=${dummy_train_path} \
    data.val_files=${dummy_test_path} \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    actor_rollout_ref.separate=true \
    actor_rollout_ref.hybrid_engine=false \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.tokenizer_path=${TOKENIZER_PATH} \
    actor_rollout_ref.model.attn_backend=${ATTN_BACKEND} \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.lora_adapter_path=null \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.nnodes=1 \
    actor_rollout_ref.rollout.n_gpus_per_node=${NUM_GPUS_STANDALONE} \
    actor_rollout_ref.rollout.checkpoint_engine.backend=${CKPT_BACKEND} \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.rollout_attn_backend=${ROLLOUT_ATTN_BACKEND} \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.pipeline.height=256 \
    actor_rollout_ref.rollout.pipeline.width=256 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=4.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=${max_prompt_length} \
    actor_rollout_ref.rollout.algo.noise_level=1.0 \
    actor_rollout_ref.rollout.algo.sde_type=sde \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,4]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    reward.num_workers=1 \
    reward.reward_model.enable=False \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=flowgrpo-diffusion-separate-e2e \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node=${NUM_GPUS_ACTOR} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS} \
    trainer.use_v1=false \
    "$@"

echo "FlowGRPO synchronous separate diffusion e2e test passed."
