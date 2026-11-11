#!/usr/bin/env bash
# MiniMax H3 image dense-text DiffusionNFT LoRA recipe.
# Reward: 0.5 * PickScore + 0.2 * (OCR x layout guard) + 0.3 * HPSv2.1 (ocr_preference_reward.py).
#
# Default cell: 512x512 generated directly (no upscale), dense-text tier t20.
#
# The rollout runs the release's 6-step PDD sampler unchanged; the Actor trains on exactly those 6 sigmas
# with the matching per-step head. No CFG, no negative prompt, no resolution upscale.
#
# Requires:
#   MODEL_PATH   native release root (transformer/, vae/, tokenizer/, pdd_config.json, pdd_fused_heads.safetensors)
#   ACTOR_PATH   Diffusers-layout transformer written by
#                python verl_omni/pipelines/minimax_h3_image_diffusion_nft/checkpoint.py convert \
#                    --native-root "$MODEL_PATH" --output "$ACTOR_PATH"
#   DATA_DIR     directory with train_t20.parquet / test_t20.parquet (or set TRAIN_FILES / VAL_FILES)
#   REWARD_MODEL_NAME  Qwen3-VL-8B-Instruct repo id or local path (served by the reward worker)
set -euo pipefail

export VERL_OMNI_H3_IMAGE_REPLICATE_FROZEN=1
export VERL_OMNI_H3_IMAGE_PACK_MICRO_BATCH=1
export GENRM_OCR_TEMPERATURE=0
export VLLM_ENGINE_READY_TIMEOUT_S=${VLLM_ENGINE_READY_TIMEOUT_S:-3600}
: "${PPOCR_MODEL_DIR:?Set PPOCR_MODEL_DIR to the PP-OCRv6 medium ONNX directory}"
: "${PICKSCORE_PATH:?Set PICKSCORE_PATH to yuvalkirstain/PickScore_v1}"
: "${CLIP_H14_PATH:?Set CLIP_H14_PATH to laion/CLIP-ViT-H-14-laion2B-s32B-b79K}"
: "${CUSTOM_REWARD_MODEL_PATH:?Set CUSTOM_REWARD_MODEL_PATH to HPS_v2.1_compressed.pt}"
export HPSV2_PRETRAINED_PATH=${HPSV2_PRETRAINED_PATH:-$CLIP_H14_PATH/open_clip_pytorch_model.bin}
export CUSTOM_REWARD_MODEL_PATH

export WANDB_MODE=${WANDB_MODE:-offline}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

export TRAIN_FILES=${TRAIN_FILES:-${DATA_DIR:?Set DATA_DIR or TRAIN_FILES and VAL_FILES}/train_t20.parquet}
export VAL_FILES=${VAL_FILES:-$DATA_DIR/test_t20.parquet}
: "${MODEL_PATH:?Set MODEL_PATH to the native MiniMax H3 image release root}"
: "${ACTOR_PATH:?Set ACTOR_PATH to the converted Diffusers-layout transformer directory}"
for required in "$MODEL_PATH/pdd_config.json" "$MODEL_PATH/pdd_fused_heads.safetensors" "$ACTOR_PATH/config.json"; do
    [[ -f "$required" ]] || { echo "Missing $required" >&2; exit 1; }
done
REWARD_MODEL_NAME=${REWARD_MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}
REWARD_TP=${REWARD_TP:-1}
REWARD_GPU_MEM_UTIL=${REWARD_GPU_MEM_UTIL:-0.15}

N_GPUS=${N_GPUS:-3}
export PREFERENCE_SCORER_GPU=${PREFERENCE_SCORER_GPU:-$N_GPUS}
RAY_GPU_ARGS=()
if python3 -c "import sys, torch; sys.exit(torch.version.hip is None)"; then
    # vLLM's ROCm platform copies a driver-level HIP_VISIBLE_DEVICES into CUDA_VISIBLE_DEVICES; Ray workers
    # inherit that while Ray sets a per-worker HIP_VISIBLE_DEVICES, and ROCm aborts on the mismatch.
    # Leave visibility unset and cap Ray at the first N_GPUS devices instead.
    unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
    RAY_GPU_ARGS=(+ray_kwargs.ray_init.num_gpus="$N_GPUS")
else
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS - 1)))
fi
if (( PREFERENCE_SCORER_GPU < N_GPUS )); then
    echo "PREFERENCE_SCORER_GPU=$PREFERENCE_SCORER_GPU must not be one of the $N_GPUS trainer GPUs" >&2
    exit 1
fi
ROLLOUT_TP=${ROLLOUT_TP:-1}
TEXT_ENCODER_TP=${TEXT_ENCODER_TP:-$ROLLOUT_TP}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-12}
ROLLOUT_N=${ROLLOUT_N:-8}
HEIGHT=${HEIGHT:-512}
WIDTH=${WIDTH:-512}
MAX_PROMPT_EMBEDS=${MAX_PROMPT_EMBEDS:-512}
ACTOR_ATTN_BACKEND=${ACTOR_ATTN_BACKEND:-native}
ROLLOUT_ATTN_BACKEND=${ROLLOUT_ATTN_BACKEND:-TORCH_SDPA}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-50}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-minimax_h3_image_ocr_lora}

script_path=$(readlink -f "$0")
reward_file="$(dirname "$script_path")/ocr_preference_reward.py"
output_dir=${OUTPUT_DIR:-$(dirname "$script_path")/outputs/$EXPERIMENT_NAME}
mkdir -p "$output_dir/checkpoints"

lora_warmstart_arg=()
if [[ -n "${LORA_WARMSTART_PATH:-}" ]]; then
    lora_warmstart_arg=(actor_rollout_ref.model.lora_adapter_path=$LORA_WARMSTART_PATH)
fi

${VERL_OMNI_ENTRY:-python3 -m verl_omni.trainer.main_diffusion} \
    trainer.use_v1=false \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    algorithm.adv_mode=continuous \
    algorithm.timestep_fraction=0.5 \
    algorithm.old_policy_decay_schedule=delayed_linear_to_0_999 \
    algorithm.old_policy_update_interval=2 \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_max_samples=64 \
    data.max_prompt_length="$MAX_PROMPT_EMBEDS" \
    data.truncation=error \
    data.seed=42 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.tokenizer_path="$MODEL_PATH/tokenizer" \
    actor_rollout_ref.model.config_path="$ACTOR_PATH" \
    +actor_rollout_ref.model.architecture=MiniMaxH3ImagePipeline \
    actor_rollout_ref.model.external_lib=verl_omni.pipelines.minimax_h3_image_diffusion_nft \
    actor_rollout_ref.model.algorithm=diffusion_nft \
    actor_rollout_ref.model.model_type=diffusion_nft_model \
    actor_rollout_ref.model.attn_backend="$ACTOR_ATTN_BACKEND" \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.lora_rank=66 \
    actor_rollout_ref.model.lora_alpha=132 \
    "${lora_warmstart_arg[@]}" \
    actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
    actor_rollout_ref.model.target_modules='["to_q","to_k","to_v","to_out.0","ff.net.0.proj","ff.net.2"]' \
    actor_rollout_ref.model.fsdp_layer_prefixes="['transformer_blocks.','token_refiner.refiner_blocks.']" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=1e-4 \
    actor_rollout_ref.actor.optim.betas="[0.9,0.999]" \
    actor_rollout_ref.actor.optim.override_optimizer_config="{eps: 1e-8}" \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=diffusion_nft \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.actor.diffusion_loss.mix_beta=0.1 \
    actor_rollout_ref.actor.diffusion_loss.ref_kl_coef=0.0001 \
    actor_rollout_ref.actor.diffusion_loss.adv_clip_max=5.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.save_lora_only=True \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.max_num_seqs=1 \
    actor_rollout_ref.rollout.rollout_attn_backend="$ROLLOUT_ATTN_BACKEND" \
    actor_rollout_ref.rollout.rollout_adapter=old \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.text_encoder_tp_size="$TEXT_ENCODER_TP" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.agent.num_workers=$((N_GPUS / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.agent.default_agent_loop=minimax_h3_diffusion_single_turn_agent \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.max_prompt_embed_length="$MAX_PROMPT_EMBEDS" \
    actor_rollout_ref.rollout.pipeline.height="$HEIGHT" \
    actor_rollout_ref.rollout.pipeline.width="$WIDTH" \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=6 \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length="$MAX_PROMPT_EMBEDS" \
    +actor_rollout_ref.rollout.pipeline.output_type=pt \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.height="$HEIGHT" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.width="$WIDTH" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=6 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.true_cfg_scale=1.0 \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.output_type=pt \
    reward.num_workers=$((N_GPUS / REWARD_TP)) \
    reward.reward_model.enable=True \
    reward.reward_model.model_path="$REWARD_MODEL_NAME" \
    reward.reward_model.rollout.name=vllm \
    reward.reward_model.rollout.tensor_model_parallel_size="$REWARD_TP" \
    reward.reward_model.rollout.gpu_memory_utilization="$REWARD_GPU_MEM_UTIL" \
    reward.reward_model.rollout.max_model_len=8192 \
    reward.reward_model.rollout.response_length=160 \
    reward.custom_reward_function.path=pkg://verl_omni.reward_loop.reward_manager.multi \
    reward.custom_reward_function.name=_multi_reward_placeholder \
    reward.reward_manager.name=MultiVisualRewardManager \
    reward.reward_manager.module.path=pkg://verl_omni.reward_loop.reward_manager \
    "+reward.reward_functions.pick.path=$reward_file" \
    +reward.reward_functions.pick.name=term_pickscore \
    +reward.reward_functions.pick.weight=0.5 \
    "+reward.reward_functions.ocr.path=$reward_file" \
    +reward.reward_functions.ocr.name=term_ocr_guarded \
    +reward.reward_functions.ocr.weight=0.2 \
    "+reward.reward_functions.aes.path=$reward_file" \
    +reward.reward_functions.aes.name=term_hpsv2 \
    +reward.reward_functions.aes.weight=0.3 \
    reward.aggregation=weighted_sum \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=diffusion_nft \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$output_dir/checkpoints" \
    trainer.validation_data_dir="$output_dir/validation_data" \
    trainer.rollout_data_dir="$output_dir/rollout_data" \
    trainer.rollout_data_save_freq=10 \
    trainer.log_val_generations=64 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    "${RAY_GPU_ARGS[@]}" \
    "$@"
