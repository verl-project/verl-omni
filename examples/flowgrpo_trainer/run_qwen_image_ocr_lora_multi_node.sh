# Qwen-Image lora RL, vllm_omni rollout
set -x

# Set WORKSPACE to any writable directory; defaults to $HOME
WORKSPACE=${WORKSPACE:-$HOME}

ocr_train_path=$WORKSPACE/data/ocr/qwen_image/train.parquet
ocr_test_path=$WORKSPACE/data/ocr/qwen_image/test.parquet

model_name=Qwen/Qwen-Image
reward_model_name=Qwen/Qwen3-VL-8B-Instruct
reward_function_path=verl_omni/utils/reward_score/genrm_ocr.py

# ---- Cluster topology --------------------------------------------------------
NNODES=${NNODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))     # 8

# ---- Parallelism -------------------------------------------------------------
# Rollout: TP=1, DP=1 -> 16 independent vLLMOmniReplica instances (one per GPU).
# Each replica's nnodes = (TP*DP)/GPUS_PER_NODE = 1, so run_headless is bypassed.
ROLLOUT_TP=1
# Reward: keep TP=4 so one reward server fits on 4 GPUs on a single node.
# With TOTAL_GPUS=16 this yields 4 reward replicas (2 per node).
REWARD_TP=4

ENGINE=vllm_omni
REWARD_ENGINE=vllm

# ---- Batch sizing ------------------------------------------------------------
# Single-node baseline (run_qwen_image_ocr_lora.sh): 4 GPUs, train_batch=32,
# ppo_mini=16. Linearly scale by TOTAL_GPUS/4 = 4x to keep per-GPU work the
# same while increasing global batch size, matching the RFC's "scale-up"
# validation goal (equivalent per-GPU throughput, larger global batch).
TRAIN_BATCH_SIZE=$((32 * TOTAL_GPUS / 4))   # 256
PPO_MINI_BATCH_SIZE=$((16 * TOTAL_GPUS / 4)) # 128
PPO_MICRO_BATCH_PER_GPU=16                   # unchanged

# Number of AgentLoopWorker actors that fan out prompts in parallel and call
# the HTTP servers. Following the existing convention (one client per
# rollout replica): TOTAL_GPUS / ROLLOUT_TP. NOTE: this knob controls the
# *clients*, not the number of replicas (replicas = total_gpus / TP / DP).
ROLLOUT_NUM_WORKERS=$((TOTAL_GPUS / ROLLOUT_TP))   # 16

# Optional reproducibility (yaml defaults are null / unseeded):
#   data.seed=42
#   actor_rollout_ref.rollout.seed=42

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=$ocr_train_path \
    data.val_files=$ocr_test_path \
    data.train_batch_size=$TRAIN_BATCH_SIZE  \
    data.max_prompt_length=256 \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out','img_mlp.net.0.proj','img_mlp.net.2','txt_mlp.net.0.proj','txt_mlp.net.2']" \
    actor_rollout_ref.model.attn_backend="_flash_3_varlen_hub" \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_PER_GPU \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=$GPUS_PER_NODE \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=False \
    actor_rollout_ref.actor.fsdp_config.offload_policy=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.agent.num_workers=$ROLLOUT_NUM_WORKERS \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.true_cfg_scale=4.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.algo.noise_level=1.2 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,5]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=50 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    reward.num_workers=$((TOTAL_GPUS / REWARD_TP)) \
    reward.reward_model.enable=True \
    reward.reward_model.model_path=$reward_model_name \
    reward.reward_model.rollout.name=$REWARD_ENGINE \
    reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
    reward.custom_reward_function.path=$reward_function_path \
    reward.custom_reward_function.name=compute_score_ocr \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=qwen_image_ocr_lora_multinode_${NNODES}x${GPUS_PER_NODE} \
    trainer.log_val_generations=8 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=300 "$@"
