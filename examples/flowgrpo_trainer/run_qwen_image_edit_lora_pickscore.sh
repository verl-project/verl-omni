# Qwen-Image-Edit-2511 LoRA RL on ShareGPT-4o-Image-Mini, PickScore reward
# PickScore: local CLIP-based preference model, NO vLLM needed for reward
set -x

export RAY_DEDUP_LOGS=0

model_name=${MODEL_PATH:-Qwen/Qwen-Image-Edit-2511}
reward_function_path=${REWARD_FUNCTION_PATH:-verl_omni/utils/reward_score/pickscore.py}

NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS_ACTOR_ROLLOUT_REWARD:-4}
ACTOR_SP=${ACTOR_SP:-1}
ROLLOUT_TP=${ROLLOUT_TP:-1}
IMAGE_RESOLUTION=${IMAGE_RESOLUTION:-512}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}

ENGINE=vllm_omni

# Qwen-Image-Edit stores its processor in a `processor/` subdirectory that lacks
# a config.json with `model_type`, causing verl's hf_processor (AutoConfig) to fail.
# Patch: write a minimal config.json into the processor dir so AutoConfig can
# resolve the model type. This is idempotent and does not affect model weights.
python3 - <<'PATCH_PROCESSOR'
import json, pathlib, subprocess, sys

model_id = "Qwen/Qwen-Image-Edit-2511"

# Resolve the snapshot directory from HF cache
try:
    from huggingface_hub import snapshot_download
    local = snapshot_download(model_id, local_files_only=True)
except Exception:
    import os
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    slug = "models--" + model_id.replace("/", "--")
    refs = pathlib.Path(cache) / slug / "refs" / "main"
    if refs.exists():
        sha = refs.read_text().strip()
        local = str(pathlib.Path(cache) / slug / "snapshots" / sha)
    else:
        print("WARNING: could not locate Qwen-Image-Edit-2511 in HF cache; skipping processor patch", file=sys.stderr)
        sys.exit(0)

proc_dir = pathlib.Path(local) / "processor"
cfg_file = proc_dir / "config.json"
if proc_dir.exists() and not cfg_file.exists():
    cfg_file.write_text(json.dumps({"model_type": "qwen2_vl"}))
    print(f"Patched processor config: {cfg_file}")
else:
    print(f"Processor config already present or processor dir missing: {cfg_file}")
PATCH_PROCESSOR

# Optional reproducibility (yaml defaults are null / unseeded):
#   data.seed=42
#   actor_rollout_ref.rollout.seed=42

script_path=$(readlink -f "$0")
script_name=$(basename "$script_path" .sh)
repo_root=$(dirname "$script_path")
while [[ "$repo_root" != "/" && ! -f "$repo_root/LICENSE" ]]; do
    repo_root=$(dirname "$repo_root")
done
if [[ ! -f "$repo_root/LICENSE" ]]; then
    echo "Unable to locate repo root from $script_path: no LICENSE found" >&2
    exit 1
fi

# Set WORKSPACE to any writable directory; defaults to the repository root.
WORKSPACE=${WORKSPACE:-$repo_root}
train_path=${TRAIN_FILES:-$WORKSPACE/data/sharegpt4o_image_mini_qwen_image_edit/train.parquet}
test_path=${VAL_FILES:-$WORKSPACE/data/sharegpt4o_image_mini_qwen_image_edit/test.parquet}

output_dir=$repo_root/outputs/$script_name
checkpoint_dir=$output_dir/checkpoints
run_timestamp=$(date +"%Y%m%d_%H%M")
log_file=$output_dir/logs/$run_timestamp/${NODE_RANK:-0}.log
rollout_data_dir=$output_dir/logs/$run_timestamp/rollout_images
mkdir -p "$checkpoint_dir" "$(dirname "$log_file")"
exec > >(tee -a "$log_file") 2>&1
echo "Logging to $log_file"

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=$train_path \
    data.val_files=$test_path \
    data.train_batch_size=32 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out','img_mlp.net.0.proj','img_mlp.net.2','txt_mlp.net.0.proj','txt_mlp.net.2']" \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=$ACTOR_SP \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.prompt_length=$MAX_PROMPT_LENGTH \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=4.0 \
    actor_rollout_ref.rollout.pipeline.height=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.width=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=$MAX_PROMPT_LENGTH \
    actor_rollout_ref.rollout.algo.noise_level=1.2 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,5]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=28 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    reward.num_workers=1 \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path=$reward_function_path \
    reward.custom_reward_function.name=compute_score \
    trainer.logger='["console", "tensorboard"]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=qwen_image_edit_sharegpt4o_image_mini_lora_pickscore \
    trainer.default_local_dir=$checkpoint_dir \
    +trainer.rollout_data_dir=$rollout_data_dir \
    trainer.log_val_generations=8 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=300 \
    trainer.resume_mode=auto "$@"
