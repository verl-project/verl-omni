# SD3 offline DPO training on pre-ranked image pairs
set -x

# Set WORKSPACE to any writable directory; defaults to $HOME.
WORKSPACE=${WORKSPACE:-$HOME}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

offline_train_path=${OFFLINE_DPO_TRAIN_PATH:-$SCRIPT_DIR/../../datasets/offline_dpo/train.parquet}
offline_test_path=${OFFLINE_DPO_TEST_PATH:-$SCRIPT_DIR/../../datasets/offline_dpo/test.parquet}

model_name=stabilityai/stable-diffusion-3.5-medium

NUM_GPUS_ACTOR=1

python3 -m verl_omni.trainer.main_diffusion --config-name=offline_dpo_trainer \
    data.train_files=$offline_train_path \
    data.val_files=$offline_test_path \
    trainer.resume_mode=disable \
    data.train_batch_size=16 \
    data.max_prompt_length=256 \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.algorithm=dpo \
    actor_rollout_ref.actor.diffusion_loss.dpo_beta=100.0 \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out']" \
    actor_rollout_ref.actor.optim.lr=2e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=dpo \
    reward.reward_model.enable=False \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=offline_dpo \
    trainer.experiment_name=sd3_offline_dpo_lora \
    trainer.test_freq=-1 \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.total_epochs=300 \
    trainer.total_training_steps=1000 "$@"
