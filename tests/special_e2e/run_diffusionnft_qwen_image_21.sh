#!/usr/bin/env bash
# Qwen-Image 2.1 NFT：V1 sync + FSDP2 + vLLM-Omni 在线训练 smoke。
# LoRA；NUM_REFERENCE_IMAGES=0..4，0 为 T2I。
# 随机权重、合成数据和已有的内容均值 reward；无需下载真实模型。
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
NUM_GPUS=${NUM_GPUS:-4}
NUM_REFERENCE_IMAGES=${NUM_REFERENCE_IMAGES:-0}
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-2}
for value in "$NUM_GPUS" "$TOTAL_TRAIN_STEPS"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "NUM_GPUS and TOTAL_TRAIN_STEPS must be positive integers." >&2
        exit 2
    fi
done
if [[ ! "$NUM_REFERENCE_IMAGES" =~ ^[0-4]$ ]]; then
    echo "NUM_REFERENCE_IMAGES must be between 0 and 4." >&2
    exit 2
fi

work_dir=$(mktemp -d)
trap 'rm -rf -- "$work_dir"' EXIT
model_path="$work_dir/model"
python3 -m tests.special_e2e.build_qwen_image_21_tiny_random --output-dir "$model_path"

# 复用已有 dummy prompts 和图片生成，只适配单 user 及多图占位符。
python3 - "$work_dir" "$NUM_GPUS" "$TOTAL_TRAIN_STEPS" "$NUM_REFERENCE_IMAGES" <<'PY'
import sys
from pathlib import Path

import pandas as pd

from tests.special_e2e.create_dummy_diffusion_data import build_rows
from tests.special_e2e.create_dummy_image_edit_data import _create_dummy_image

root, batch, steps, refs = Path(sys.argv[1]), *map(int, sys.argv[2:])
for split, size in (("train", batch * steps), ("test", batch)):
    rows = build_rows(split, size, ["jpeg_compressibility"])
    for index, row in enumerate(rows):
        row["prompt"] = row["prompt"][-1:]
        row["prompt"][0]["content"] = "<image>" * refs + row["prompt"][0]["content"]
        row["negative_prompt"] = [{"role": "user", "content": "<image>" * refs + " "}]
        if refs:
            row["images"] = [{"bytes": _create_dummy_image(seed=index * 4 + image)} for image in range(refs)]
    pd.DataFrame(rows).to_parquet(root / f"{split}.parquet")
PY

python3 -m verl_omni.trainer.main_diffusion_v1 \
    "data.train_files=$work_dir/train.parquet" \
    "data.val_files=$work_dir/test.parquet" \
    data.train_batch_size="$NUM_GPUS" data.max_prompt_length=32768 \
    "actor_rollout_ref.model.path=$model_path" \
    "actor_rollout_ref.model.tokenizer_path=$model_path/processor" \
    actor_rollout_ref.model.algorithm=diffusion_nft \
    actor_rollout_ref.model.model_type=diffusion_nft_model \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.model.lora_rank=8 actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
    actor_rollout_ref.model.target_modules='["to_q","to_k","to_v","to_out.0"]' \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size="$NUM_GPUS" \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.offload_policy=true \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$NUM_GPUS" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=diffusion_nft \
    actor_rollout_ref.actor.diffusion_loss.mix_beta=0.1 \
    actor_rollout_ref.actor.diffusion_loss.ref_kl_coef=0.0001 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.agent.default_agent_loop=qwen_image_21_diffusion_single_turn_agent \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=2 actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.max_num_seqs=1 \
    actor_rollout_ref.rollout.max_prompt_embed_length=32768 \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.calculate_log_probs=false \
    actor_rollout_ref.rollout.rollout_adapter=old \
    actor_rollout_ref.rollout.checkpoint_engine.backend=naive \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=2 \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.height=64 actor_rollout_ref.rollout.pipeline.width=64 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=8192 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=2 \
    algorithm.trainer_type=direct_preference algorithm.sample_source=online \
    algorithm.timestep_fraction=1.0 algorithm.adv_mode=continuous \
    algorithm.old_policy_decay_schedule=delayed_linear_to_0_999 \
    algorithm.old_policy_update_interval=1 \
    reward.num_workers=1 reward.reward_model.enable=false \
    reward.custom_reward_function.path=tests/special_e2e/wan22_dummy_reward.py \
    reward.custom_reward_function.name=compute_score \
    trainer.logger=console trainer.project_name=verl-test \
    "trainer.experiment_name=qwen-image-21-nft-lora-${NUM_REFERENCE_IMAGES}refs" \
    "trainer.default_local_dir=$work_dir/checkpoints" \
    trainer.n_gpus_per_node="$NUM_GPUS" trainer.nnodes=1 \
    trainer.val_before_train=false trainer.test_freq=-1 trainer.save_freq=-1 \
    trainer.resume_mode=disable trainer.total_epochs=1 \
    trainer.total_training_steps="$TOTAL_TRAIN_STEPS" \
    trainer.use_v1=true trainer.v1.trainer_mode=sync "$@"
