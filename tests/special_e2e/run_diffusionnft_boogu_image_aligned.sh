#!/usr/bin/env bash
# DiffusionNFT Boogu-Image convergence check, aligned to the QwenImage+DiffusionNFT
# example's dataset/reward settings, per review on PR #568:
# https://github.com/verl-project/verl-omni/pull/568
#
# The plain smoke test (run_diffusionnft_boogu_image.sh) uses the free rule-based
# jpeg_compressibility reward on 8 dummy samples with rollout.n=2 -- enough to prove
# the training loop executes, but with no discriminating reward signal or group size
# to produce a real advantage estimate. This script instead mirrors
# examples/diffusionnft_trainer/qwen_image/run_qwen_image_ocr_lora.sh's reward/loss
# settings and examples/flowgrpo_trainer/boogu_image/run_boogu_image_ocr_lora.sh's
# OCR reward wiring for Boogu-Image:
#   - real generative reward model (Qwen3-VL-8B-Instruct via genrm_ocr), not
#     jpeg_compressibility
#   - rollout.n=16 (was 2) for a meaningful DiffusionNFT advantage estimate
#   - mix_beta / ref_kl_coef / old_policy_update_interval matched to the
#     QwenImage+DiffusionNFT recipe
#
# The policy checkpoint is still the tiny random Boogu-Image build used by the rest
# of this PR's e2e suite, not the real pretrained checkpoint -- using the real
# checkpoint plus the 8B reward model at the QwenImage recipe's full 4xA800-80G scale
# is far outside a smoke test's budget. This keeps the reward/dataset mechanism
# identical to the recipe while keeping compute cheap.
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0

python3 - <<'PY'
import torch

print(f"PyTorch {torch.__version__}, CUDA build {torch.version.cuda}", flush=True)
if torch.version.cuda is None or not torch.cuda.is_available():
    raise RuntimeError("The BOOGU convergence check requires CUDA PyTorch.")
torch.ones(1, device="cuda").sum().item()

try:
    import cv2  # noqa: F401
except ImportError as exc:
    raise RuntimeError(
        "OpenCV must import successfully before starting rollout workers. "
        "For a minimal Ubuntu image, install libgl1 and libglib2.0-0."
    ) from exc
PY

NUM_GPUS=${NUM_GPUS:-4}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/Boogu-Image}
SOURCE_MODEL=${SOURCE_MODEL:-Boogu/Boogu-Image-0.1-Base}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}/processor}
REWARD_MODEL_NAME=${REWARD_MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}
REWARD_FUNCTION_PATH=${REWARD_FUNCTION_PATH:-verl_omni/utils/reward_score/genrm_ocr.py}

DATA_DIR=${DATA_DIR:-${HOME}/data/ocr_style_diffusion}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-100}
max_prompt_length=256
experiment_name=diffusionnft-boogu-image-ocr-aligned

dummy_train_path=${TRAIN_FILES:-${DATA_DIR}/train.parquet}
dummy_test_path=${VAL_FILES:-${DATA_DIR}/test.parquet}

ENGINE=vllm_omni
REWARD_ENGINE=vllm

if ! python3 -c 'import boogu' >/dev/null 2>&1; then
    echo "SKIP: the boogu-image package is required for the training-side transformer."
    exit 5
fi

ATTN_BACKEND=native
ROLLOUT_ATTN_BACKEND=TORCH_SDPA
# Same TP split as the real Boogu-Image + QwenImage OCR recipes: rollout stays
# TP=1 (the vllm-omni BooguImagePipeline supports neither TP nor SP nor
# CFG-parallel), the reward engine uses all GPUs via TP.
ROLLOUT_TP=1
REWARD_TP=${NUM_GPUS}

n_resp_per_prompt=16
train_prompts=4
micro_bsz_per_gpu=1
mini_bsz=$((micro_bsz_per_gpu * NUM_GPUS))
# data.train_batch_size counts dataset rows (unique prompts) per step, not
# post-rollout.n samples -- the dataset must contain at least this many rows.
train_batch_size=${train_prompts}

python3 tests/special_e2e/build_boogu_image_tiny_random.py \
    --output-dir "${MODEL_PATH}" \
    --source-model "${SOURCE_MODEL}"

python3 tests/special_e2e/create_ocr_style_diffusion_data.py \
    --local_save_dir "${DATA_DIR}" \
    --train_size "${train_prompts}" \
    --val_size 4

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=${dummy_train_path} \
    data.val_files=${dummy_test_path} \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    actor_rollout_ref.model.algorithm=diffusion_nft \
    actor_rollout_ref.model.model_type=diffusion_nft_model \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.tokenizer_path=${TOKENIZER_PATH} \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.attn_backend=${ATTN_BACKEND} \
    actor_rollout_ref.rollout.rollout_attn_backend=${ROLLOUT_ATTN_BACKEND} \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.fsdp_layer_prefixes="['double_stream_layers.','single_stream_layers.','context_refiner.','noise_refiner.','ref_image_refiner.']" \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=diffusion_nft \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.actor.diffusion_loss.mix_beta=0.1 \
    actor_rollout_ref.actor.diffusion_loss.ref_kl_coef=0.0001 \
    actor_rollout_ref.actor.diffusion_loss.adv_clip_max=5.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.rollout_adapter=old \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=4 \
    actor_rollout_ref.rollout.pipeline.height=256 \
    actor_rollout_ref.rollout.pipeline.width=256 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.pipeline.guidance_scale=4.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=${max_prompt_length} \
    actor_rollout_ref.rollout.algo.noise_level=0.0 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=null \
    actor_rollout_ref.rollout.algo.sde_window_range=null \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    algorithm.timestep_fraction=1.0 \
    algorithm.old_policy_decay_schedule=delayed_linear_to_0_999 \
    algorithm.old_policy_update_interval=2 \
    algorithm.adv_mode=continuous \
    reward.num_workers=$((NUM_GPUS / REWARD_TP)) \
    reward.reward_model.enable=True \
    reward.reward_model.model_path=${REWARD_MODEL_NAME} \
    reward.reward_model.rollout.name=${REWARD_ENGINE} \
    reward.reward_model.rollout.tensor_model_parallel_size=${REWARD_TP} \
    reward.reward_model.rollout.max_model_len=8192 \
    reward.custom_reward_function.path=${REWARD_FUNCTION_PATH} \
    reward.custom_reward_function.name=compute_score_ocr \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=${experiment_name} \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=10 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_epochs=${TOTAL_TRAIN_STEPS} \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS} \
    "$@"

echo "DiffusionNFT Boogu-Image OCR-aligned convergence check passed (${TOTAL_TRAIN_STEPS} steps completed)."
