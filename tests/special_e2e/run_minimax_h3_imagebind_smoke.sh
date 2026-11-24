#!/usr/bin/env bash
# Run one H3 training step with the production recipe's named ImageBind binding.
set -eo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
cd "$script_dir/../.."
export IMAGEBIND_SMOKE_DEVICE=${1:?expected cuda or npu}
shift
model_root="$HOME/models/tiny-random/minimax-h3"
export OUTPUT_DIR
OUTPUT_DIR=$(mktemp -d "${TMPDIR:-/tmp}/h3-imagebind.XXXXXX")
export DATA_DIR="$OUTPUT_DIR/data" IMAGEBIND_MODEL_PATH="$OUTPUT_DIR/imagebind.pth"
export VERL_FILE_LOGGER_PATH="$OUTPUT_DIR/metrics.jsonl"

case "$IMAGEBIND_SMOKE_DEVICE" in
    cuda)
        python3 "$script_dir/build_minimax_h3_tiny_random.py" --output-dir "$model_root"
        # Use the venv's cuDNN, as in the existing H3 tiny trainer smoke.
        cudnn_dir=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/cudnn/lib")')
        if [[ -d "$cudnn_dir" ]]; then
            export LD_LIBRARY_PATH="$cudnn_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        fi
        launcher=examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_t2va_lora_v1.sh
        device_args=("$@")
        ;;
    npu)
        cache="${H3_TINY_MODEL_DIR:-$model_root}"
        model_root="$OUTPUT_DIR/minimax-h3"
        python3 "$script_dir/build_minimax_h3_tiny_random.py" --output-dir "$model_root" --offline-cache "$cache"
        source "$script_dir/lib_npu_e2e.sh"
        export NUM_GPUS="$NUM_NPUS"
        launcher=examples/flowgrpo_trainer/minimax_h3/run_minimax_h3_t2va_lora_v1_npu.sh
        # Diffusers 0.41 implements Ulysses backward for native SDPA.
        device_args=("${NPU_RUNTIME_ARGS[@]}" actor_rollout_ref.model.attn_backend=native)
        ;;
    *) echo "Expected cuda or npu" >&2; exit 2 ;;
esac

export MODEL_PATH="$model_root/FL2VA" ACTOR_CONFIG_PATH="$model_root/transformer"
export VERL_OMNI_MINIMAX_H3_TINY_TEXT_CONFIG="$MODEL_PATH/text_encoder/config.json"
export PYTHONPATH="$script_dir/tiny_imagebind:$script_dir/minimax_h3_tiny_patch:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HEIGHT=160 WIDTH=288 NUM_FRAMES=97 INFER_STEPS=4 TOTAL_TRAINING_STEPS=1 TEXT_ENCODER_TP=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 FLASHINFER_DISABLE_VERSION_CHECK=1
python3 "$script_dir/create_dummy_h3_data.py" --task t2va --local-save-dir "$DATA_DIR" --train-size "$NUM_GPUS" --val-size 2
python3 -c 'import os, torch; torch.save({"direction": torch.tensor([3., 4.])}, os.environ["IMAGEBIND_MODEL_PATH"])'

bash "$launcher" "${device_args[@]}" \
    '~reward.reward_functions.clap' \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.trust_remote_code=True \
    +actor_rollout_ref.model.architecture=MiniMaxH3Pipeline \
    actor_rollout_ref.model.lora_rank=8 actor_rollout_ref.model.lora_alpha=16 \
    "data.train_batch_size=$NUM_GPUS" "actor_rollout_ref.actor.ppo_mini_batch_size=$NUM_GPUS" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    data.max_prompt_length=256 actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.pipeline.task=t2va \
    'actor_rollout_ref.rollout.algo.sde_window_range=[0,3]' actor_rollout_ref.rollout.algo.sde_window_size=2 \
    'trainer.logger=[console,file]' trainer.val_before_train=False trainer.test_freq=-1 trainer.save_freq=-1 \
    trainer.resume_mode=disable trainer.log_val_generations=0 trainer.rollout_data_save_freq=0 \
    "ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS:-40}" "hydra.run.dir=$OUTPUT_DIR" hydra.output_subdir=.hydra

python3 - "$OUTPUT_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

from omegaconf import OmegaConf

output = Path(sys.argv[1])
reward = OmegaConf.load(output / ".hydra/config.yaml").reward
assert reward.models.imagebind.backend == "native"
assert reward.models.imagebind.executor.model == "verl_omni.utils.reward_score.imagebind:ImageBindModel"
assert reward.reward_functions.imagebind.model == "imagebind"
assert reward.reward_functions.imagebind.name == "compute_score_native"
steps = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
(metrics,) = [row["data"] for row in steps if row["step"] == 1 and "actor/grad_norm" in row["data"]]
score = metrics["critic/reward/imagebind/mean"]
assert 0.70 < score < 0.90, score
assert not metrics.get("critic/reward/imagebind/errors/count", 0)
assert math.isclose(metrics["critic/rewards/mean"], score, abs_tol=1e-6)
assert metrics["critic/rewards/zero_std_ratio"] < 1, "Every rollout group received identical rewards"
grad_norm = metrics["actor/grad_norm"]
assert math.isfinite(grad_norm) and grad_norm > 0, grad_norm
print(f"Step 1: managed ImageBind mean = {score:.6f}")
PY
