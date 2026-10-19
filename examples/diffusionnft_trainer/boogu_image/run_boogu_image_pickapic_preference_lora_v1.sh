#!/usr/bin/env bash
# Boogu-Image DiffusionNFT LoRA on Pick-a-Pic v1 prompts with the V1 sync trainer and
# HPSv2.1 + PickScore + CLIPScore (1:1:1).
#
# Default workload:
#   24 prompt groups/update * 16 images/group = 384 rollouts/update
#   10 training denoising steps, 40 validation denoising steps on 256 fixed prompts
#   all rollout timesteps trained (the standard DiffusionNFT setting)
#   CFG-free (guidance_scale=1.0): DiffusionNFT regresses the trained velocity onto
#   the policy's own samples, and the CFG-extrapolated velocity is not a proper flow,
#   so training it erodes the guidance and washes the images out.
#
# Hardware: four GPUs shared synchronously by actor/rollout and reward workers.
# Prepare data first:
#   python examples/flowgrpo_trainer/data_process/boogu_image_pickapic.py
set -euo pipefail

# vLLM 0.30's allocator extension uses the pip-provided CUDA 13 runtime even
# when PyTorch itself is a cu129 build. Make those shared libraries discoverable.
cuda13_lib=$(python3 - <<'PY'
from importlib.metadata import PackageNotFoundError, distribution

try:
    path = distribution("nvidia-cuda-runtime").locate_file("nvidia/cu13/lib")
    print(path if path.is_dir() else "")
except PackageNotFoundError:
    print("")
PY
)
if [[ -n "$cuda13_lib" ]]; then
    export LD_LIBRARY_PATH=$cuda13_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi

# Ray workers inherit this. Uncapped, every reward and actor worker spawns one CPU
# thread per core and they oversubscribe the host: reward scoring took 87-157 s per
# step instead of ~4 s, and the CPU-offloaded optimizer step slowed by ~6%.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

WORKSPACE=${WORKSPACE:-$HOME}
TRAIN_FILES=${TRAIN_FILES:-$WORKSPACE/data/pickapic_v1/boogu_image/train.parquet}
VAL_FILES=${VAL_FILES:-$WORKSPACE/data/pickapic_v1/boogu_image/test.parquet}

MODEL_NAME=${MODEL_NAME:-Boogu/Boogu-Image-0.1-Base}
if [[ -d "$MODEL_NAME" ]]; then
    model_dir=$MODEL_NAME
else
    model_dir=$(python3 - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download("${MODEL_NAME}"))
PY
    )
fi
TOKENIZER_PATH=${TOKENIZER_PATH:-$model_dir/processor}

: "${HPSV2_PRETRAINED_PATH:?Set HPSV2_PRETRAINED_PATH to the OpenCLIP ViT-H-14 checkpoint}"
: "${CUSTOM_REWARD_MODEL_PATH:?Set CUSTOM_REWARD_MODEL_PATH to the HPS_v2.1 reward checkpoint}"
[[ -f "$HPSV2_PRETRAINED_PATH" ]] || { echo "Missing HPSv2 base checkpoint: $HPSV2_PRETRAINED_PATH" >&2; exit 1; }
[[ -f "$CUSTOM_REWARD_MODEL_PATH" ]] || {
    echo "Missing HPSv2.1 reward checkpoint: $CUSTOM_REWARD_MODEL_PATH" >&2
    exit 1
}

TOTAL_GPUS=${NUM_GPUS:-4}
if (( TOTAL_GPUS <= 0 )); then
    echo "NUM_GPUS must be positive, got $TOTAL_GPUS." >&2
    exit 1
fi
NUM_GPUS_ACTOR_ROLLOUT=$TOTAL_GPUS
REWARD_DEVICES="["
for ((gpu_index = 0; gpu_index < TOTAL_GPUS; gpu_index++)); do
    (( gpu_index > 0 )) && REWARD_DEVICES+=","
    REWARD_DEVICES+="$gpu_index"
done
REWARD_DEVICES+="]"

ROLLOUT_TP=1
ENGINE=vllm_omni
ATTN_BACKEND=${ATTN_BACKEND:-native}
if [[ "$ATTN_BACKEND" == "fa3" ]]; then
    MODEL_ATTN_BACKEND=_flash_3_varlen_hub
    ROLLOUT_ATTN_BACKEND=FLASH_ATTN_3_HUB
else
    MODEL_ATTN_BACKEND=native
    ROLLOUT_ATTN_BACKEND=TORCH_SDPA
fi

IMAGE_RESOLUTION=${IMAGE_RESOLUTION:-512}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-256}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-10}
VAL_STEPS=${VAL_STEPS:-40}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-1.0}
TIMESTEP_FRACTION=${TIMESTEP_FRACTION:-1.0}
# Leave room for one resident HPSv2 + PickScore + CLIPScore replica per GPU.
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.45}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1}
REQUEST_BATCH_MAX_WAIT_MS=${REQUEST_BATCH_MAX_WAIT_MS:-0}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-24}
ROLLOUT_N=${ROLLOUT_N:-16}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-5760}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-256}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-60}
DATA_SEED=${DATA_SEED:-42}
VAL_SEED=${VAL_SEED:-42}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-10}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-3}

ACTOR_LR=${ACTOR_LR:-3e-4}
REF_KL_COEF=${REF_KL_COEF:-1e-4}
ADV_CLIP_MAX=${ADV_CLIP_MAX:-5.0}
REWARD_DTYPE=${REWARD_DTYPE:-bfloat16}
REWARD_BATCH_SIZE=${REWARD_BATCH_SIZE:-16}
PICKSCORE_MODEL=${PICKSCORE_MODEL:-yuvalkirstain/PickScore_v1}
PICKSCORE_PROCESSOR=${PICKSCORE_PROCESSOR:-laion/CLIP-ViT-H-14-laion2B-s32B-b79K}
CLIPSCORE_MODEL=${CLIPSCORE_MODEL:-openai/clip-vit-large-patch14}

HPSV2_WEIGHT=${HPSV2_WEIGHT:-1.0}
PICKSCORE_WEIGHT=${PICKSCORE_WEIGHT:-1.0}
CLIPSCORE_WEIGHT=${CLIPSCORE_WEIGHT:-1.0}
PICKSCORE_DIVISOR=${PICKSCORE_DIVISOR:-26.0}
CLIPSCORE_SCALE=${CLIPSCORE_SCALE:-2.5}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-boogu_image_pickapic_hpsv2_pickscore_clipscore_lora_v1}
OUTPUT_DIR=${OUTPUT_DIR:-$WORKSPACE/outputs/$EXPERIMENT_NAME}
mkdir -p "$OUTPUT_DIR/checkpoints" "$OUTPUT_DIR/validation"

echo "[workload] groups=$TRAIN_BATCH_SIZE group_size=$ROLLOUT_N images/update=$((TRAIN_BATCH_SIZE * ROLLOUT_N))" >&2
echo "[resources] $TOTAL_GPUS GPUs shared synchronously by actor/rollout and reward replicas" >&2

python3 -m verl_omni.trainer.main_diffusion_v1 \
    "data.train_files=$TRAIN_FILES" \
    "data.val_files=$VAL_FILES" \
    data.train_max_samples=$TRAIN_MAX_SAMPLES \
    data.val_max_samples=$VAL_MAX_SAMPLES \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.seed=$DATA_SEED \
    actor_rollout_ref.model.algorithm=diffusion_nft \
    actor_rollout_ref.model.model_type=diffusion_nft_model \
    "actor_rollout_ref.model.path=$model_dir" \
    "actor_rollout_ref.model.tokenizer_path=$TOKENIZER_PATH" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.attn_backend=$MODEL_ATTN_BACKEND \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','img_to_q','img_to_k','img_to_v','img_out','instruct_to_q','instruct_to_k','instruct_to_v','instruct_out','feed_forward.linear_1','feed_forward.linear_2','feed_forward.linear_3','img_feed_forward.linear_1','img_feed_forward.linear_2','img_feed_forward.linear_3']" \
    actor_rollout_ref.model.fsdp_layer_prefixes="['double_stream_layers.','single_stream_layers.','context_refiner.','noise_refiner.','ref_image_refiner.']" \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=12 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=12 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=diffusion_nft \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.actor.diffusion_loss.mix_beta=0.1 \
    actor_rollout_ref.actor.diffusion_loss.ref_kl_coef=$REF_KL_COEF \
    actor_rollout_ref.actor.diffusion_loss.adv_clip_max=$ADV_CLIP_MAX \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT / ROLLOUT_TP)) \
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
    actor_rollout_ref.rollout.pipeline.max_sequence_length=$MAX_PROMPT_LENGTH \
    actor_rollout_ref.rollout.algo.noise_level=0.0 \
    actor_rollout_ref.rollout.algo.sde_type=sde \
    actor_rollout_ref.rollout.algo.sde_window_size=null \
    actor_rollout_ref.rollout.algo.sde_window_range=null \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=$VAL_STEPS \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    +actor_rollout_ref.rollout.val_kwargs.seed=$VAL_SEED \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    algorithm.timestep_fraction=$TIMESTEP_FRACTION \
    algorithm.old_policy_decay_schedule=delayed_linear_to_0_999 \
    algorithm.old_policy_update_interval=2 \
    algorithm.adv_mode=continuous \
    reward.num_workers=$TOTAL_GPUS \
    reward.reward_model.enable=False \
    reward.reward_model.enable_resource_pool=False \
    reward.reward_model.n_gpus_per_node=$TOTAL_GPUS \
    reward.reward_model.nnodes=1 \
    reward.custom_reward_function.path=pkg://verl_omni.reward_loop.reward_manager.multi \
    reward.custom_reward_function.name=_multi_reward_placeholder \
    reward.reward_manager.name=MultiVisualRewardManager \
    reward.reward_manager.module.path=pkg://verl_omni.reward_loop.reward_manager \
    +reward.models.preference_mix.backend=native \
    +reward.models.preference_mix.offload=false \
    "+reward.models.preference_mix.placement.devices=$REWARD_DEVICES" \
    +reward.models.preference_mix.executor.model=pkg://verl_omni.utils.reward_score.preference_mix_reward:PreferenceMixNativeModel \
    "+reward.models.preference_mix.executor.kwargs.hpsv2_pretrained_path=$HPSV2_PRETRAINED_PATH" \
    "+reward.models.preference_mix.executor.kwargs.hpsv2_checkpoint_path=$CUSTOM_REWARD_MODEL_PATH" \
    "+reward.models.preference_mix.executor.kwargs.dtype=$REWARD_DTYPE" \
    "+reward.models.preference_mix.executor.kwargs.pickscore_model_path=$PICKSCORE_MODEL" \
    "+reward.models.preference_mix.executor.kwargs.pickscore_processor_path=$PICKSCORE_PROCESSOR" \
    "+reward.models.preference_mix.executor.kwargs.clipscore_model_path=$CLIPSCORE_MODEL" \
    +reward.models.preference_mix.executor.kwargs.max_batch_size=$REWARD_BATCH_SIZE \
    +reward.reward_functions.preference_mix.model=preference_mix \
    +reward.reward_functions.preference_mix.path=pkg://verl_omni.utils.reward_score.preference_mix_reward \
    +reward.reward_functions.preference_mix.name=compute_score_preference_mix \
    +reward.reward_functions.preference_mix.weight=1.0 \
    +reward.reward_functions.preference_mix.required=true \
    +reward.reward_functions.preference_mix.hpsv2_weight=$HPSV2_WEIGHT \
    +reward.reward_functions.preference_mix.pickscore_weight=$PICKSCORE_WEIGHT \
    +reward.reward_functions.preference_mix.clipscore_weight=$CLIPSCORE_WEIGHT \
    +reward.reward_functions.preference_mix.pickscore_divisor=$PICKSCORE_DIVISOR \
    +reward.reward_functions.preference_mix.clipscore_scale=$CLIPSCORE_SCALE \
    reward.aggregation=weighted_sum \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=diffusion_nft \
    trainer.experiment_name=$EXPERIMENT_NAME \
    "trainer.default_local_dir=$OUTPUT_DIR/checkpoints" \
    "trainer.validation_data_dir=$OUTPUT_DIR/validation" \
    trainer.log_val_generations=8 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=1 \
    trainer.total_training_steps=$TOTAL_TRAIN_STEPS \
    trainer.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP \
    trainer.resume_mode=auto \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=sync "$@"
