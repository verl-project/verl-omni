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
#                   TOTAL_TRAIN_STEPS, TOTAL_EPOCHS, TRAIN_FILES, VAL_FILES
MODE=${MODE:-t2i}
NUM_GPUS=${NUM_GPUS:-4}
MODEL_PATH=${MODEL_PATH:-${HOME}/models/tiny-random/Boogu-Image}
SOURCE_MODEL=${SOURCE_MODEL:-Boogu/Boogu-Image-0.1-Base}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}/processor}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}

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

# Boogu's rollout transformer carries only the attention projections and the
# feed-forward halves. FSDP layered-summon does not transport top-level modules
# (`x_embedder`, `caption_embedder`, the patch embedders) to the rollout, so
# `all-linear` names targets that can never bind -- and the adapter's
# validate_boogu_lora_targets() rejects such a list outright, which is how this
# harness came to report success while every weight sync failed. Same list as the
# recipe: examples/diffusionnft_trainer/boogu_image/run_boogu_image_ocr_lora.sh.
BOOGU_LORA_TARGETS="['to_q','to_k','to_v','to_out.0','img_to_q','img_to_k','img_to_v','img_out','instruct_to_q','instruct_to_k','instruct_to_v','instruct_out','feed_forward.linear_1','feed_forward.linear_2','feed_forward.linear_3','img_feed_forward.linear_1','img_feed_forward.linear_2','img_feed_forward.linear_3']"

# Training stdout is tee'd here so the LoRA sync can be asserted afterwards.
TRAIN_LOG="$(mktemp "${TMPDIR:-/tmp}/boogu_nft_e2e.XXXXXX.log")"
trap 'rm -f "${TRAIN_LOG}"' EXIT

n_resp_per_prompt=2
micro_bsz_per_gpu=1
rollout_tp=1
# Validate before the arithmetic below: `expected_engine_workers` divides by `rollout_tp`,
# so a non-dividing override would truncate and the sync assertion would check the wrong
# worker count instead of failing.
if (( NUM_GPUS < 1 || micro_bsz_per_gpu < 1 )); then
    echo "FAIL: NUM_GPUS (${NUM_GPUS}) and micro_bsz_per_gpu (${micro_bsz_per_gpu}) must be positive."
    exit 1
fi
if (( NUM_GPUS % rollout_tp != 0 )); then
    echo "FAIL: NUM_GPUS (${NUM_GPUS}) must be divisible by rollout_tp (${rollout_tp})."
    exit 1
fi
micro_bsz=$((micro_bsz_per_gpu * NUM_GPUS))
mini_bsz=${micro_bsz}
train_batch_size=$((mini_bsz * n_resp_per_prompt))
# Size the dummy set for the requested epochs, as the QwenImage smoke test does:
# `train_batch_size` alone only yields one epoch's worth of prompts.
steps_per_epoch=$(((TOTAL_TRAIN_STEPS + TOTAL_EPOCHS - 1) / TOTAL_EPOCHS))
synthetic_train_size=$((train_batch_size * steps_per_epoch))

# Idempotent: a no-op when the tiny checkpoint is already present.
python3 tests/special_e2e/build_boogu_image_tiny_random.py \
    --output-dir "${MODEL_PATH}" \
    --source-model "${SOURCE_MODEL}"

if [[ "${MODE}" == "edit" ]]; then
    # Mirror the real edit recipe's data contract: `boogu_image_edit_ocr.py` emits a text-only
    # negative prompt (guided TI2I does not feed the reference image to the negative branch), so
    # the row references fewer media than it carries. Generating the `with-image` form here would
    # keep the harness green while the real recipe's parquet failed to load -- the placeholder
    # count would match the image count and mask the loader's exact-count check.
    python3 tests/special_e2e/create_dummy_image_edit_data.py \
        --local_save_dir "${DATA_DIR}" \
        --train_size "${synthetic_train_size}" \
        --val_size 4 \
        --image-width 256 \
        --image-height 256 \
        --negative-prompt-mode text-only
else
    python3 tests/special_e2e/create_dummy_diffusion_data.py \
        --local_save_dir "${DATA_DIR}" \
        --train_size "${synthetic_train_size}" \
        --val_size 4
fi

# Guard the fixture contract the edit mode depends on. The row must reference fewer media than
# it carries, i.e. carry a negative prompt with no `<image>` placeholder. If the generator ever
# reverts to the `with-image` form, the placeholder count equals the image count, the loader's
# exact-count check is satisfied for the wrong reason, and this harness silently stops covering
# the row shape the real edit recipe trains on -- which is precisely how it stayed green while
# `data/ocr/boogu_image_edit` could not be loaded.
if [[ "${MODE}" == "edit" ]]; then
    python3 - "${dummy_train_path}" <<'PY'
import sys

import pandas as pd

negative = pd.read_parquet(sys.argv[1]).iloc[0]["negative_prompt"][1]["content"]
assert "<image>" not in negative, (
    f"edit fixture must emit a text-only negative prompt, got {negative!r}; the harness would no "
    "longer cover the real edit recipe's data contract (see --negative-prompt-mode above)"
)
PY
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
    actor_rollout_ref.model.target_modules="${BOOGU_LORA_TARGETS}" \
    actor_rollout_ref.model.fsdp_layer_prefixes="['double_stream_layers.','single_stream_layers.','context_refiner.','noise_refiner.','ref_image_refiner.']" \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_bsz_per_gpu} \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.actor.diffusion_loss.mix_beta=0.1 \
    actor_rollout_ref.actor.diffusion_loss.ref_kl_coef=10.0 \
    actor_rollout_ref.actor.diffusion_loss.adv_clip_max=1.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
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
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.total_training_steps=${TOTAL_TRAIN_STEPS} \
    "$@" 2>&1 | tee "${TRAIN_LOG}"

# Guard the argument-list contract above. If a comment is ever added inside that
# backslash-continued list, the command terminates at the comment and every later
# override is silently dropped -- surfacing much later as a confusing resource error.
# `experiment_name` is the last `trainer.*` argument, so it is the cheapest witness that
# the whole list was parsed.
if ! grep -q "'experiment_name': '${experiment_name}'" "${TRAIN_LOG}"; then
    echo "FAIL: the trailing Hydra overrides never reached the trainer (its resolved config"
    echo "      does not report experiment_name='${experiment_name}'). The argument list was"
    echo "      truncated -- look for a comment line inside the backslash-continued list."
    exit 1
fi

# Training exiting 0 is not evidence that the actor's deltas reached the rollout.
# vllm-omni only raises when *no* target binds, so a partial name/target miss
# stays silent while the actor keeps training modules the rollout never receives
# (issue #658) -- and the engine pins VLLM_LOGGING_LEVEL=WARN, so vllm-omni's own
# INFO line about a loaded adapter never reaches this output. The mapper therefore
# reports its binding outcome once per engine process at WARNING level; assert
# that positive evidence rather than trusting the exit code.
#
# The report is emitted once per engine process, so a single match is not enough:
# if three of four workers dropped every delta, `grep -q` would still pass and the
# run would look healthy. Count the reports and require exactly one per engine.
expected_engine_workers=$((NUM_GPUS / rollout_tp))
bind_reports=$(grep -cE "Boogu-Image LoRA sync: bound [1-9][0-9]* actor delta modules to vllm-omni, 0 dropped" "${TRAIN_LOG}" || true)
if [[ "${bind_reports}" -ne "${expected_engine_workers}" ]]; then
    echo "FAIL: expected ${expected_engine_workers} LoRA sync report(s) (one per engine"
    echo "      process), got ${bind_reports}."
    echo "      Expected one line per engine process of the form:"
    echo "        Boogu-Image LoRA sync: bound <N> actor delta modules to vllm-omni, 0 dropped (<M> wrapped target modules)."
    echo "      A missing report means that engine's deltas did not reach the rollout,"
    echo "      so this run never exercised the DiffusionNFT update path and its pass"
    echo "      would be vacuous. Look above for \"unsupported targets\" or"
    echo "      \"update_weights_from_ipc' failed\"."
    exit 1
fi

echo "DiffusionNFT Boogu-Image (MODE=${MODE}) e2e test passed (training completed successfully)."
