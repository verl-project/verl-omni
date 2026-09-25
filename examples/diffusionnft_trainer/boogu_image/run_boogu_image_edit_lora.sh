#!/usr/bin/env bash
# Boogu-Image DiffusionNFT LoRA RL, edit (TI2I) mode, vllm_omni rollout.
#
# The edit sibling of examples/diffusionnft_trainer/boogu_image/run_boogu_image_ocr_lora.sh.
# Boogu runs T2I and TI2I through the same DiffusionNFT pipeline, so only the dataset and the
# prompt budget differ: the rollout adapter reads the reference image from each row's `images`
# column and conditions on it as `condition_image_latents`. The edit prompt keeps its `<image>`
# placeholder, which expands well past the T2I recipe's 256-token budget, so both
# `data.max_prompt_length` and `actor_rollout_ref.rollout.pipeline.max_sequence_length` are
# raised to 512 together -- the e2e harness derives the latter from the former, and splitting
# them would silently truncate the instruction.
#
# Data: examples/flowgrpo_trainer/data_process/boogu_image_edit_ocr.py
#   python3 examples/flowgrpo_trainer/data_process/boogu_image_edit_ocr.py \
#       --input_dir ~/data/ocr_edit \
#       --output_dir "$HOME/data/ocr/boogu_image_edit_pickscore" \
#       --image_size 512
# The reward is PickScore -- the reward the verified TI2I recipe
# examples/flowgrpo_trainer/qwen_image_edit/run_qwen_image_edit_lora.sh uses. It scores the
# generated image against the edit instruction by CLIP similarity, so it reads the
# *instruction* from each row's `ground_truth`; that is why the dataset above is the
# `_pickscore` one, which the converter writes with the instruction in `ground_truth` rather
# than the target text the OCR GenRM needed. PickScore runs CLIP locally in the reward
# workers, so this recipe serves no reward model at all.
#
# Output resolution follows the reference image (`align_res`), which the converter pins by
# letterboxing onto a square `--image_size` canvas.
#
# Run from the repo root: `reward.custom_reward_function.path` is repo-relative.
set -euo pipefail

# Set WORKSPACE to any writable directory; defaults to $HOME.
WORKSPACE=${WORKSPACE:-$HOME}

ocr_train_path=${TRAIN_FILES:-$WORKSPACE/data/ocr/boogu_image_edit_pickscore/train.parquet}
ocr_test_path=${VAL_FILES:-$WORKSPACE/data/ocr/boogu_image_edit_pickscore/test.parquet}

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

NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS:-4}
# BooguImagePipeline supports neither TP nor SP nor CFG-parallel.
ROLLOUT_TP=1

ENGINE=vllm_omni

# --- reward function --------------------------------------------------------
# PickScore, as in the verified Qwen-Image-Edit TI2I recipe. It runs CLIP locally in the
# reward workers, so no reward model is served and the ~17GB/GPU GenRM replica the T2I
# recipe carries is absent here. One worker per GPU, since each worker loads its own CLIP
# onto the device it is placed on.
reward_function_path=${REWARD_FUNCTION_PATH:-pkg://verl_omni.utils.reward_score.pickscore_reward}
reward_function_name=${REWARD_FUNCTION_NAME:-compute_score_pickscore}
REWARD_WORKERS=${REWARD_WORKERS:-$NUM_GPUS_ACTOR_ROLLOUT_REWARD}
echo "[reward] fn=${reward_function_name} path=${reward_function_path} workers=${REWARD_WORKERS} data=$(dirname "$ocr_train_path")" >&2

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

# With PickScore there is no co-located reward engine, so the rollout no longer shares its
# device with an 8B GenRM replica. The T2I recipe fits that replica in 0.25 on top of a 0.5
# rollout; with the replica gone, 0.5 for the rollout is what this recipe already used and
# leaves strictly more headroom for the actor than before.
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}

# Edit output resolution follows the reference image (align_res); this is the
# fallback when a batch carries no reference.
IMAGE_RESOLUTION=${IMAGE_RESOLUTION:-512}
# Prompt budget: raised together with `data.max_prompt_length` below (see header).
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-10}
VAL_STEPS=${VAL_STEPS:-40}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-4.0}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
REQUEST_BATCH_MAX_WAIT_MS=${REQUEST_BATCH_MAX_WAIT_MS:-10}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-24}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-7200}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-256}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-30}
DATA_SEED=${DATA_SEED:-42}

python3 -m verl_omni.trainer.main_diffusion \
    trainer.use_v1=false \
    data.train_files=$ocr_train_path \
    data.val_files=$ocr_test_path \
    data.train_max_samples=$TRAIN_MAX_SAMPLES \
    data.val_max_samples=$VAL_MAX_SAMPLES \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.seed=$DATA_SEED \
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
    actor_rollout_ref.rollout.pipeline.max_sequence_length=$MAX_PROMPT_LENGTH \
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
    reward.num_workers=$REWARD_WORKERS \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path=$reward_function_path \
    reward.custom_reward_function.name=$reward_function_name \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=diffusion_nft \
    trainer.experiment_name=boogu_image_ocr_edit_lora \
    trainer.log_val_generations=8 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=$TOTAL_TRAIN_STEPS "$@"
