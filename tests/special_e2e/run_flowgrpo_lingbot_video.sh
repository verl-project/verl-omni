#!/usr/bin/env bash
# LingBot-Video Dense T2V FlowGRPO e2e smoke test (minimal runtime), vllm_omni rollout.
#
# Single pass covering:
#   parquet load (structured JSON captions) -> lingbot_dense_t2v_agent rollout
#   (vllm_omni) -> jpeg_compressibility rule reward (self-contained, no reward
#   server) -> flow_grpo -> FSDP LoRA on the `blocks.*` transformer -> weight sync.
#
# Requires: vllm-omni, diffusers>=0.37, the optional `lingbot-video` package, and
#   a tiny LingBot-Video Dense checkpoint at ~/models/tiny-random/lingbot-video-dense.
#   If that checkpoint is absent it is built here from LINGBOT_SOURCE_MODEL (a real
#   Dense checkpoint, used offline only for its processor/scheduler configs -- the
#   multi-GB weight shards are never loaded).
set -euo pipefail

# Override via env: NUM_GPUS, ROLLOUT_TP, MODEL_PATH, LINGBOT_SOURCE_MODEL, DATA_DIR,
#                   TOTAL_TRAIN_STEPS, TRAIN_FILES, VAL_FILES
NUM_GPUS=${NUM_GPUS:-4}
ROLLOUT_TP=${ROLLOUT_TP:-1}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/lingbot-video-dense}
LINGBOT_SOURCE_MODEL=${LINGBOT_SOURCE_MODEL:-${HOME}/models/lingbot-video-dense-1.3b}
DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_lingbot_video}
dummy_train_path=${TRAIN_FILES:-${DATA_DIR}/train.parquet}
dummy_test_path=${VAL_FILES:-${DATA_DIR}/test.parquet}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-1}

ENGINE=vllm_omni
# LingBot captions expand to the ~150-token rewrite template plus a short caption
# dict, so the tiny smoke prompts stay well under this budget.
max_prompt_length=512

# Build the tiny checkpoint on demand.  LingBot has no shared tiny-model provisioning
# step (unlike the Qwen-Image smoke tests), so the e2e self-provisions and is a no-op
# once the checkpoint exists.
if [[ ! -f "${MODEL_PATH}/model_index.json" ]]; then
    echo "Tiny LingBot-Video Dense checkpoint not found at ${MODEL_PATH}; building it..."
    python3 tests/special_e2e/build_lingbot_video_dense_tiny_random.py \
        --output-dir "${MODEL_PATH}" \
        --source-model "${LINGBOT_SOURCE_MODEL}"
fi

# This helper runs nvidia-smi in a background loop during training and
# fails if any vLLMOmniHttpServer process is left resident on GPU-0.
_LEAK_FILE=$(mktemp)
_LEAK_PID=""
cleanup_leak_monitor() {
    [[ -n "${_LEAK_PID}" ]] && kill "${_LEAK_PID}" 2>/dev/null || true
    rm -f "${_LEAK_FILE}"
}
trap cleanup_leak_monitor EXIT

start_leak_monitor() {
    : > "${_LEAK_FILE}"
    while true; do
        if nvidia-smi -i 0 2>&1 | grep -q "vLLMOmniHttpServer"; then
            echo "LEAK" >> "${_LEAK_FILE}"
        fi
        sleep 1
    done &
    _LEAK_PID=$!
}

check_leak_monitor() {
    kill "${_LEAK_PID}" 2>/dev/null || true
    _LEAK_PID=""
    if grep -q "LEAK" "${_LEAK_FILE}" 2>/dev/null; then
        echo ""
        echo "FAIL: unexpected vLLMOmniHttpServer process(es) detected on GPU-0 —"
        ray stop --force 2>/dev/null || true
        exit 1
    fi
}

n_resp_per_prompt=2
micro_bsz_per_gpu=1
micro_bsz=$((micro_bsz_per_gpu * NUM_GPUS))
mini_bsz=${micro_bsz}
train_batch_size=$((mini_bsz * n_resp_per_prompt))

python3 tests/special_e2e/create_dummy_lingbot_video_data.py \
    --local_save_dir "${DATA_DIR}" \
    --train_size "${train_batch_size}" \
    --val_size 4

start_leak_monitor
python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=${dummy_train_path} \
    data.val_files=${dummy_test_path} \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    'actor_rollout_ref.model.fsdp_layer_prefixes=["blocks."]' \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    "actor_rollout_ref.model.target_modules=['to_q','to_k','to_v','to_out','gate_proj','up_proj','down_proj']" \
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
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.agent.default_agent_loop=lingbot_dense_t2v_agent \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.pipeline.height=64 \
    actor_rollout_ref.rollout.pipeline.width=64 \
    actor_rollout_ref.rollout.pipeline.num_frames=5 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.pipeline.guidance_scale=3.0 \
    actor_rollout_ref.rollout.pipeline.shift=3.0 \
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
    trainer.experiment_name=flowgrpo-lingbot-video-e2e \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS} \
    "$@"
check_leak_monitor

echo "FlowGRPO LingBot-Video Dense T2V e2e test passed (training completed successfully)."
