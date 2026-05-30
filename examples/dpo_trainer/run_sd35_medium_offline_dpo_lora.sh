# SD3 offline DPO training on pre-ranked image pairs
set -x

# Set WORKSPACE to any writable directory; defaults to $HOME.
WORKSPACE=${WORKSPACE:-$HOME}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

offline_train_path=${OFFLINE_DPO_TRAIN_PATH:-$SCRIPT_DIR/../../datasets/offline_dpo/train.parquet}
offline_test_path=${OFFLINE_DPO_TEST_PATH:-$SCRIPT_DIR/../../datasets/offline_dpo/test.parquet}

model_name=stabilityai/stable-diffusion-3.5-medium
custom_chat_template='{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}{{ message['\''content'\''] }}{% endif %}{% endfor %}'

NUM_GPUS_ACTOR=1

python3 -m verl_omni.trainer.main_diffusion \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=offline \
    data.train_files=$offline_train_path \
    data.val_files=$offline_test_path \
    data.train_batch_size=16 \
    data.max_prompt_length=256 \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.offline_dpo_dataset \
    data.custom_cls.name=OfflineDPODataset \
    data.custom_cls.collate_fn=offline_dpo_collate_fn \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.algorithm=dpo \
    actor_rollout_ref.model.model_type=diffusion_dp_model \
    actor_rollout_ref.model.custom_chat_template="\"$custom_chat_template\"" \
    actor_rollout_ref.model.external_lib=verl_omni.pipelines.sd3_dpo \
    actor_rollout_ref.model.pipeline.guidance_scale=4.0 \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out']" \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=dpo \
    actor_rollout_ref.actor.diffusion_loss.dpo_beta=100.0 \
    actor_rollout_ref.actor.optim.lr=2e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm_omni \
    trainer.resume_mode=disable \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=offline_dpo \
    trainer.experiment_name=sd3_offline_dpo_lora \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR \
    trainer.save_freq=30 \
    trainer.total_epochs=300 \
    trainer.total_training_steps=1000 "$@"
