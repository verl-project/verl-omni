# BAGEL Uni-COT supervised fine-tuning with LoRA + FSDP.
#
# Expected Uni-COT rows contain:
#   image_list: [problem_image, step1_diagram, step2_diagram, ...]
#   instruction_list: [instruction text]
#   output_text_list: interleaved reasoning text with <image_start>/<image_end>
set -x

WORKSPACE=${WORKSPACE:-$HOME}
BAGEL_EXAMPLE_DIR=${BAGEL_EXAMPLE_DIR:-$WORKSPACE/data/bagel_example}
UNICOT_TRAIN_FILE=${UNICOT_TRAIN_FILE:-$BAGEL_EXAMPLE_DIR/unicot_sft/train.jsonl}
UNICOT_VAL_FILE=${UNICOT_VAL_FILE:-$BAGEL_EXAMPLE_DIR/unicot_sft/val.jsonl}
UNICOT_DATA_CONFIG=${UNICOT_DATA_CONFIG:-"$(dirname "$0")/unicot_data_config.yaml"}

model_name=${BAGEL_MODEL_PATH:-~/models/ByteDance-Seed/BAGEL-7B-MoT}

NUM_GPUS=${NUM_GPUS:-4}

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=$UNICOT_TRAIN_FILE \
    data.val_files=$UNICOT_VAL_FILE \
    data.train_batch_size=8 \
    data.max_prompt_length=8192 \
    data.trust_remote_code=True \
    data.custom_cls.path=pkg://verl_omni.utils.dataset.unicot_sft_dataset \
    data.custom_cls.name=UniCOTSFTDataset \
    data.custom_cls.collate_fn=unicot_sft_collate_fn \
    data.custom_cls.dataset_config_file=$UNICOT_DATA_CONFIG \
    data.custom_cls.train_split=train \
    data.custom_cls.val_split=train \
    algorithm.trainer_type=sft \
    algorithm.sample_source=offline \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.tokenizer_path=$model_name \
    actor_rollout_ref.model.model_type=bagel_sft_model \
    actor_rollout_ref.model.algorithm=bagel_sft \
    +actor_rollout_ref.model.architecture=OmniBagelForConditionalGeneration \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.lora_rank=256 \
    actor_rollout_ref.model.lora_alpha=512 \
    actor_rollout_ref.model.lora_dtype=float32 \
    actor_rollout_ref.model.target_modules="['q_proj_moe_gen','k_proj_moe_gen','v_proj_moe_gen','o_proj_moe_gen','mlp_moe_gen.gate_proj','mlp_moe_gen.up_proj','mlp_moe_gen.down_proj']" \
    actor_rollout_ref.model.fsdp_layer_prefixes="['layers.']" \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=bagel_sft \
    actor_rollout_ref.actor.diffusion_loss.ce_weight=1.0 \
    actor_rollout_ref.actor.diffusion_loss.mse_weight=1.0 \
    actor_rollout_ref.actor.optim.lr=2e-5 \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=bagel_sft \
    trainer.experiment_name=bagel_unicot_lora \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=500 \
    trainer.test_freq=0 \
    trainer.total_epochs=3 \
    trainer.total_training_steps=3000 "$@"
