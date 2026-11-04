#!/usr/bin/env bash
# Lance LoRA RL (text-to-image), vllm_omni rollout (FlowGRPO) with PickScore reward
#
# Lance shares BAGEL's transformer, so this mirrors the BAGEL PickScore
# recipe.  What differs is the checkpoint (3B), the deploy config and the
# architecture name; the sigma shift (3.5) lives in the adapters.
#
# Prerequisite: preprocess the PickScore dataset for Lance:
#   python examples/flowgrpo_trainer/data_process/lance_pickscore.py \
#       --model_path ~/models/bytedance-research/Lance \
#       --input_dir ~/data/pickscore \
#       --output_dir ~/data/pickscore/lance
#
# Raw dataset (train.txt / test.txt) from:
#   https://github.com/yifan123/flow_grpo/tree/main/dataset/pickscore
set -euo pipefail

# Set WORKSPACE to any writable directory; defaults to $HOME
WORKSPACE=${WORKSPACE:-$HOME}

pickscore_train_path=$WORKSPACE/data/pickscore/lance/train.parquet
pickscore_test_path=$WORKSPACE/data/pickscore/lance/test.parquet

LANCE_DEPLOY_CONFIG=${LANCE_DEPLOY_CONFIG:-"$(dirname "$0")/lance_deploy_config.yaml"}

# Bundle root; the adapters resolve the Lance_3B subdirectory.
model_name=${MODEL_PATH:-"$WORKSPACE/models/bytedance-research/Lance"}
# The released tokenizer lives inside the checkpoint directory rather than at
# the bundle root, and designates neither eos nor pad, which the agent loop
# needs the moment it pads a batch.  The data prep writes a repaired copy
# beside the parquet; point verl at that, since it loads this path as given
# and only the model adapters resolve the bundle root.
tokenizer_name=$WORKSPACE/data/pickscore/lance/tokenizer
# That tokenizer ships no chat_template, and the agent loop renders every
# prompt through one.  A user turn in Qwen's ChatML layout is what the rollout
# adapter's _extract_prompt_text pulls the caption back out of.
# Literal newlines must survive both shell and Hydra parsing.
custom_chat_template='{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}<|im_start|>user
{{ message['\''content'\''] }}<|im_end|>
{% endif %}{% endfor %}'
reward_function_path=verl_omni/utils/reward_score/pickscore_reward.py

NUM_GPUS_ACTOR_ROLLOUT_REWARD=4
ROLLOUT_TP=1

ENGINE=vllm_omni

# enable reward model on 0'th gpu, it is a temporary workaround
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

python3 -m verl_omni.trainer.main_diffusion \
    "data.train_files=$pickscore_train_path" \
    "data.val_files=$pickscore_test_path" \
    data.train_batch_size=48 \
    data.max_prompt_length=256 \
    data.filter_overlong_prompts=True \
    +data.apply_chat_template_kwargs.chat_template="\"$custom_chat_template\"" \
    data.trust_remote_code=True \
    algorithm.global_std=False \
    "actor_rollout_ref.model.path=$model_name" \
    "actor_rollout_ref.model.tokenizer_path=$tokenizer_name" \
    actor_rollout_ref.model.custom_chat_template="\"$custom_chat_template\"" \
    +actor_rollout_ref.model.architecture=OmniLanceForConditionalGeneration \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.model.target_modules="['q_proj_moe_gen','k_proj_moe_gen','v_proj_moe_gen','o_proj_moe_gen','mlp_moe_gen.gate_proj','mlp_moe_gen.up_proj','mlp_moe_gen.down_proj']" \
    actor_rollout_ref.model.fsdp_layer_prefixes="['layers.']" \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=24 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=12 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.diffusion_loss.clip_ratio=1e-5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=12 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=15 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.algo.noise_level=1.3 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,7]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=30 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    "+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.deploy_config=$LANCE_DEPLOY_CONFIG" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=12 \
    reward.num_workers=1 \
    "reward.custom_reward_function.path=$reward_function_path" \
    reward.custom_reward_function.name=compute_score_pickscore \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=lance_pickscore_lora \
    trainer.log_val_generations=8 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=300 "$@"
