#!/usr/bin/env bash
# 单图/多参考图编辑沿用 OCR 的 LoRA、FSDP2、V1 同步配置。
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/run_qwen_image_21_ocr_lora_v1.sh" \
    actor_rollout_ref.rollout.max_prompt_embed_length=32768 \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/image_edit_reward.py \
    reward.custom_reward_function.name=compute_score_image_edit \
    ++reward.reward_model.rollout.engine_kwargs.vllm.limit_mm_per_prompt.image=5 \
    reward.reward_model.rollout.temperature=0.0 \
    reward.reward_model.rollout.top_p=1.0 \
    reward.reward_model.rollout.response_length=512 \
    trainer.experiment_name=edit_lora_v1 "$@"
