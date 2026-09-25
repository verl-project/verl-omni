#!/usr/bin/env bash
# Boogu-Image DiffusionNFT LoRA RL, vllm_omni rollout.
#
# A Boogu sibling of examples/diffusionnft_trainer/qwen_image/run_qwen_image_ocr_lora.sh,
# differing in model path, LoRA targets / fsdp_layer_prefixes, the Boogu guidance knob
# (`pipeline.guidance_scale`, vs Qwen's `true_cfg_scale`), rollout TP=1, and the dataset.
#
# Run from the repo root: `reward.custom_reward_function.path` is repo-relative.
set -euo pipefail

# Set WORKSPACE to any writable directory; defaults to $HOME.
WORKSPACE=${WORKSPACE:-$HOME}

ocr_train_path=${TRAIN_FILES:-$WORKSPACE/data/ocr/boogu_image/train.parquet}
ocr_test_path=${VAL_FILES:-$WORKSPACE/data/ocr/boogu_image/test.parquet}

model_name=${MODEL_NAME:-Boogu/Boogu-Image-0.1-Base}
# Boogu's tokenizer lives under `processor/`, but an HF hub id allows only two
# segments, so resolve MODEL_NAME to a local dir and point AutoTokenizer at it.
if [[ -d "$model_name" ]]; then
    model_dir=$model_name
else
    model_dir=$(python3 - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download("${model_name}"))
PY
    )
fi
tokenizer_path=${TOKENIZER_PATH:-$model_dir/processor}
reward_model_name=${REWARD_MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}
reward_function_path=verl_omni/utils/reward_score/genrm_ocr.py

NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS:-4}
# BooguImagePipeline supports neither TP nor SP nor CFG-parallel.
ROLLOUT_TP=1
# Qwen3-VL-8B has 32 query / 8 KV heads, so REWARD_TP must be a power of two
# (TP=3 cannot partition them): the largest power of two dividing NUM_GPUS.
if [[ -z "${REWARD_TP:-}" ]]; then
    REWARD_TP=1
    while (( NUM_GPUS_ACTOR_ROLLOUT_REWARD % (REWARD_TP * 2) == 0 )); do
        REWARD_TP=$((REWARD_TP * 2))
    done
fi

ENGINE=vllm_omni
REWARD_ENGINE=vllm

# FA3 is unavailable here; default to the native/SDPA pair, which must be used
# together (enforced in diffusion_attention.py). ATTN_BACKEND=fa3 opts back in.
ATTN_BACKEND=${ATTN_BACKEND:-native}
if [[ "$ATTN_BACKEND" == "fa3" ]]; then
    MODEL_ATTN_BACKEND=_flash_3_varlen_hub
    ROLLOUT_ATTN_BACKEND=FLASH_ATTN_3_HUB
else
    MODEL_ATTN_BACKEND=native
    ROLLOUT_ATTN_BACKEND=TORCH_SDPA
fi

# The rollout and reward engines are co-located, so these shares are additive
# and must sum to ~0.75 (0.5+0.5 leaves the actor nothing and OOMs weight sync).
# TP=1 puts the full ~17GB Qwen3-VL replica on every GPU, so it needs more.
if [[ -z "${REWARD_GPU_MEM_UTIL:-}" ]]; then
    if (( REWARD_TP >= 4 )); then
        REWARD_GPU_MEM_UTIL=0.25
    else
        REWARD_GPU_MEM_UTIL=0.30
    fi
fi
if [[ -z "${ROLLOUT_GPU_MEM_UTIL:-}" ]]; then
    if (( REWARD_TP >= 4 )); then
        ROLLOUT_GPU_MEM_UTIL=0.5
    else
        ROLLOUT_GPU_MEM_UTIL=0.45
    fi
fi

IMAGE_RESOLUTION=${IMAGE_RESOLUTION:-512}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-10}
VAL_STEPS=${VAL_STEPS:-40}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-4.0}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
REQUEST_BATCH_MAX_WAIT_MS=${REQUEST_BATCH_MAX_WAIT_MS:-10}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-24}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-7200}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-256}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-30}

python3 -m verl_omni.trainer.main_diffusion \
    trainer.use_v1=false \
    data.train_files=$ocr_train_path \
    data.val_files=$ocr_test_path \
    data.train_max_samples=$TRAIN_MAX_SAMPLES \
    data.val_max_samples=$VAL_MAX_SAMPLES \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=256 \
    actor_rollout_ref.model.algorithm=diffusion_nft \
    actor_rollout_ref.model.model_type=diffusion_nft_model \
    actor_rollout_ref.model.path=$model_dir \
    actor_rollout_ref.model.tokenizer_path=$tokenizer_path \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.attn_backend=$MODEL_ATTN_BACKEND \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','img_to_q','img_to_k','img_to_v','img_out','instruct_to_q','instruct_to_k','instruct_to_v','instruct_out','feed_forward.linear_1','feed_forward.linear_2','feed_forward.linear_3','img_feed_forward.linear_1','img_feed_forward.linear_2','img_feed_forward.linear_3']" \
    actor_rollout_ref.model.fsdp_layer_prefixes="['double_stream_layers.','single_stream_layers.','context_refiner.','noise_refiner.','ref_image_refiner.']" \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=12 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=12 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=diffusion_nft \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.actor.diffusion_loss.mix_beta=0.1 \
    actor_rollout_ref.actor.diffusion_loss.ref_kl_coef=10.0 \
    actor_rollout_ref.actor.diffusion_loss.adv_clip_max=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.rollout_adapter=old \
    actor_rollout_ref.rollout.rollout_attn_backend=$ROLLOUT_ATTN_BACKEND \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL \
    actor_rollout_ref.rollout.max_num_seqs=$MAX_NUM_SEQS \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=$MAX_NUM_SEQS \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.request_batch_max_wait_ms=$REQUEST_BATCH_MAX_WAIT_MS \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=$ROLLOUT_STEPS \
    actor_rollout_ref.rollout.pipeline.guidance_scale=$GUIDANCE_SCALE \
    actor_rollout_ref.rollout.pipeline.height=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.width=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.algo.noise_level=0.0 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=null \
    actor_rollout_ref.rollout.algo.sde_window_range=null \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=$VAL_STEPS \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    algorithm.timestep_fraction=1.0 \
    algorithm.old_policy_decay_schedule=delayed_linear_to_0_999 \
    algorithm.old_policy_update_interval=2 \
    algorithm.adv_mode=continuous \
    reward.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / REWARD_TP)) \
    reward.reward_model.enable=True \
    reward.reward_model.model_path=$reward_model_name \
    reward.reward_model.rollout.name=$REWARD_ENGINE \
    reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
    reward.reward_model.rollout.gpu_memory_utilization=$REWARD_GPU_MEM_UTIL \
    reward.reward_model.rollout.max_model_len=8192 \
    reward.custom_reward_function.path=$reward_function_path \
    reward.custom_reward_function.name=compute_score_ocr \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=diffusion_nft \
    trainer.experiment_name=boogu_image_ocr_lora \
    trainer.log_val_generations=8 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=$TOTAL_TRAIN_STEPS "$@"
