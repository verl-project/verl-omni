#!/usr/bin/env bash
# DiffusionNFT Boogu-Image e2e smoke test (minimal runtime), vllm_omni rollout.
#
# MODE=t2i (default)  text-to-image path.
# MODE=edit           TI2I editing path; additionally exercises the reference
#                     latents (condition image -> ref_image_hidden_states
#                     refiner) branch.
#
# Single pass covering:
#   parquet load -> vllm_omni rollout (BooguImageDiffusionNFTPipeline,
#   deterministic ODE) -> jpeg_compressibility rule reward -> DiffusionNFT ->
#   FSDP LoRA -> sync.
#
# Requires: vllm-omni (>= Boogu support), the `boogu-image` package (canonical
#   transformer for the training side; the checkpoint's transformer_boogu.py
#   is a re-export shim), and a locally cached Boogu/Boogu-Image-0.1-Base --
#   the tiny checkpoint is built from it below (processor/scheduler are
#   copied verbatim). Base and Edit share the pipeline architecture, so one
#   tiny checkpoint serves both modes. Point SOURCE_MODEL at a local
#   directory when the base checkpoint is not in the Hugging Face cache.
set -euo pipefail

# Preserve worker diagnostics, including failures before the ready pipe is sent.
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0

# Image builders may have no GPU: uv --torch-backend=auto can install CPU
# wheels there even when the eventual training container has CUDA devices.
python3 - <<'PY'
import torch

print(f"PyTorch {torch.__version__}, CUDA build {torch.version.cuda}", flush=True)
if torch.version.cuda is None:
    raise RuntimeError(
        "The BOOGU smoke test requires CUDA PyTorch. When building the image "
        "without a GPU, select an explicit uv --torch-backend matching your "
        "CUDA stack instead of auto."
    )
if not torch.cuda.is_available():
    raise RuntimeError("CUDA PyTorch is installed, but no usable CUDA device is visible.")
torch.ones(1, device="cuda").sum().item()

# An OpenCV import failure can leave cv2/ on sys.path, causing spawned
# rollout workers to import cv2.typing instead of the standard-library typing
# module. Surface the missing system dependency before Ray hides it as EOF.
try:
    import cv2  # noqa: F401
except ImportError as exc:
    raise RuntimeError(
        "OpenCV must import successfully before starting rollout workers. "
        "For a minimal Ubuntu image, install libgl1 and libglib2.0-0."
    ) from exc
PY

# Override via env: MODE, NUM_GPUS, MODEL_PATH, SOURCE_MODEL, DATA_DIR,
#                   TOTAL_TRAIN_STEPS, TRAIN_FILES, VAL_FILES
MODE=${MODE:-t2i}
NUM_GPUS=${NUM_GPUS:-4}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/Boogu-Image}
SOURCE_MODEL=${SOURCE_MODEL:-Boogu/Boogu-Image-0.1-Base}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}/processor}

case "${MODE}" in
    t2i)
        DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_diffusion}
        TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-4}
        max_prompt_length=256
        experiment_name=diffusionnft-boogu-image-e2e
        ;;
    edit)
        DATA_DIR=${DATA_DIR:-${HOME}/data/dummy_boogu_image_edit}
        TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-2}
        max_prompt_length=512
        experiment_name=diffusionnft-boogu-image-edit-e2e
        ;;
    *)
        echo "FAIL: unknown MODE='${MODE}' (expected 't2i' or 'edit')."
        exit 1
        ;;
esac

dummy_train_path=${TRAIN_FILES:-${DATA_DIR}/train.parquet}
dummy_test_path=${VAL_FILES:-${DATA_DIR}/test.parquet}

ENGINE=vllm_omni

# boogu-image is an optional third-party dependency. Exit 5 (SKIP) rather than
# fail so its absence does not redden the shared smoke suite for everyone.
if ! python3 -c 'import boogu' >/dev/null 2>&1; then
    echo "SKIP: the boogu-image package is required for the training-side transformer."
    echo "Install it with: pip install 'boogu-image @ git+https://github.com/boogu-project/Boogu-Image.git'"
    exit 5
fi

# The Boogu path has no FA3 requirement; native/SDPA works everywhere and
# avoids the Hub fetch of kernels-community/flash-attn3 in sandboxes.
ATTN_BACKEND=native
ROLLOUT_ATTN_BACKEND=TORCH_SDPA

n_resp_per_prompt=2
micro_bsz_per_gpu=1
micro_bsz=$((micro_bsz_per_gpu * NUM_GPUS))
mini_bsz=${micro_bsz}
train_batch_size=$((mini_bsz * n_resp_per_prompt))

# Idempotent: a no-op when the tiny checkpoint is already present.
python3 tests/special_e2e/build_boogu_image_tiny_random.py \
    --output-dir "${MODEL_PATH}" \
    --source-model "${SOURCE_MODEL}"

if [[ "${MODE}" == "edit" ]]; then
    python3 tests/special_e2e/create_dummy_image_edit_data.py \
        --local_save_dir "${DATA_DIR}" \
        --train_size "${train_batch_size}" \
        --val_size 4 \
        --image-width 256 \
        --image-height 256
else
    python3 tests/special_e2e/create_dummy_diffusion_data.py \
        --local_save_dir "${DATA_DIR}" \
        --train_size "${train_batch_size}" \
        --val_size 4
fi

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
    actor_rollout_ref.actor.diffusion_loss.mix_beta=0.5 \
    actor_rollout_ref.actor.diffusion_loss.ref_kl_coef=0.001 \
    actor_rollout_ref.actor.diffusion_loss.adv_clip_max=5.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
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
    algorithm.old_policy_update_interval=1 \
    algorithm.adv_mode=continuous \
    reward.num_workers=1 \
    reward.reward_model.enable=False \
    trainer.logger=console \
    trainer.project_name=verl-test \
    trainer.experiment_name=${experiment_name} \
    trainer.log_val_generations=0 \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS} \
    "$@"

echo "DiffusionNFT Boogu-Image (MODE=${MODE}) e2e test passed (training completed successfully)."
