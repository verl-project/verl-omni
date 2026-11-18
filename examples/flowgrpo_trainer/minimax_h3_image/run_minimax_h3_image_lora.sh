#!/usr/bin/env bash
set -euo pipefail

# H3 Image needs Diffusers >=0.43.0; do not change the repository-wide minimum.
python -c 'from importlib.metadata import version; from packaging.version import Version; assert Version(version("diffusers")) >= Version("0.43.0"), "H3 Image FlowGRPO requires diffusers>=0.43.0"'

: "${MODEL_PATH:?Set MODEL_PATH to the native MiniMax-H3-Image release}"
: "${TRAIN_FILES:?Set TRAIN_FILES to a prompt parquet file}"
: "${VAL_FILES:?Set VAL_FILES to a validation parquet file}"
ACTOR_PATH=${ACTOR_PATH:-$MODEL_PATH/transformer_diffusers}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/minimax_h3_image_flowgrpo}
DEVICE=${DEVICE:-cuda}
NUM_DEVICES=${NUM_DEVICES:-4}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ACTOR_ATTN_BACKEND=${ACTOR_ATTN_BACKEND:-native}
if [[ "$DEVICE" == npu && "$ACTOR_ATTN_BACKEND" == native ]]; then
    ACTOR_ATTN_BACKEND=_native_npu
fi
if ! [[ "$NUM_DEVICES" =~ ^[1-9][0-9]*$ && "$ROLLOUT_TP" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_DEVICES and ROLLOUT_TP must be positive integers." >&2
    exit 1
fi
if (( NUM_DEVICES % ROLLOUT_TP != 0 )); then
    echo "NUM_DEVICES must be positive and divisible by ROLLOUT_TP." >&2
    exit 1
fi

python -m verl_omni.trainer.main_diffusion_v1 \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=sync \
    "trainer.device=$DEVICE" \
    algorithm.adv_estimator=flow_grpo \
    algorithm.global_std=True \
    "data.train_files=$TRAIN_FILES" \
    "data.val_files=$VAL_FILES" \
    data.train_batch_size=8 \
    data.val_max_samples=16 \
    data.max_prompt_length=512 \
    data.truncation=error \
    "actor_rollout_ref.model.path=$MODEL_PATH" \
    "actor_rollout_ref.model.config_path=$ACTOR_PATH" \
    actor_rollout_ref.model.algorithm=flow_grpo \
    "actor_rollout_ref.model.attn_backend=$ACTOR_ATTN_BACKEND" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    'actor_rollout_ref.model.target_modules=[to_q,to_k,to_v,to_out.0,ff.net.0.proj,ff.net.2]' \
    'actor_rollout_ref.model.fsdp_layer_prefixes=[transformer_blocks.,token_refiner.refiner_blocks.]' \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP" \
    "actor_rollout_ref.rollout.text_encoder_tp_size=$ROLLOUT_TP" \
    "actor_rollout_ref.rollout.agent.num_workers=$((NUM_DEVICES / ROLLOUT_TP))" \
    actor_rollout_ref.rollout.agent.default_agent_loop=minimax_h3_diffusion_single_turn_agent \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.max_num_seqs=1 \
    actor_rollout_ref.rollout.max_prompt_embed_length=512 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.pipeline.height=512 \
    actor_rollout_ref.rollout.pipeline.width=512 \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=6 \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=512 \
    +actor_rollout_ref.rollout.pipeline.output_type=pt \
    actor_rollout_ref.rollout.algo.sde_type=cps \
    actor_rollout_ref.rollout.algo.noise_level=0.8 \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    'actor_rollout_ref.rollout.algo.sde_window_range=[0,5]' \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=6 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    reward.num_workers=1 \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path=pkg://verl_omni.utils.reward_score.jpeg_compressibility \
    reward.custom_reward_function.name=compute_score \
    'trainer.logger=[console]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=minimax_h3_image \
    "trainer.default_local_dir=$OUTPUT_DIR/checkpoints" \
    "trainer.validation_data_dir=$OUTPUT_DIR/validation" \
    "trainer.n_gpus_per_node=$NUM_DEVICES" \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-5}" \
    "$@"
