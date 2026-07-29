# BAGEL example supervised fine-tuning with LoRA + FSDP.
#
# This script follows the official BAGEL TRAIN.md example-data layout:
#   bagel_example/
#   ├── t2i/
#   ├── editing/seedxedit_multi/
#   ├── editing/parquet_info/
#   └── vlm/{images,llava_ov_si.jsonl}
#
# It uses verl_omni.utils.dataset.bagel_sft_dataset through
# examples/sft_trainer/bagel/bagel_example_sft_dataset.py.
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
WORKSPACE=${WORKSPACE:-$REPO_ROOT}

BAGEL_EXAMPLE_URL=${BAGEL_EXAMPLE_URL:-https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/bagel_example.zip}
BAGEL_EXAMPLE_ZIP=${BAGEL_EXAMPLE_ZIP:-$WORKSPACE/data/bagel_example.zip}
BAGEL_EXAMPLE_DIR=${BAGEL_EXAMPLE_DIR:-$WORKSPACE/data/bagel_example}
BAGEL_DATA_CONFIG=${BAGEL_DATA_CONFIG:-$SCRIPT_DIR/bagel_example_data_config.yaml}
BAGEL_DATASET_ADAPTER=${BAGEL_DATASET_ADAPTER:-$SCRIPT_DIR/bagel_example_sft_dataset.py}
BAGEL_PARQUET_INFO_FILE=${BAGEL_PARQUET_INFO_FILE:-$BAGEL_EXAMPLE_DIR/editing/parquet_info/seedxedit_multi.json}

model_name=${BAGEL_MODEL_PATH:-$HOME/models/ByteDance-Seed/BAGEL-7B-MoT}

NUM_GPUS=${NUM_GPUS:-4}
NUM_WORKERS=${NUM_WORKERS:-1}
EXPECTED_NUM_TOKENS=${EXPECTED_NUM_TOKENS:-10240}
MAX_NUM_TOKENS=${MAX_NUM_TOKENS:-11520}
MAX_NUM_TOKENS_PER_SAMPLE=${MAX_NUM_TOKENS_PER_SAMPLE:-10240}
NUM_PACKED_BATCHES=${NUM_PACKED_BATCHES:-1000}

if [ ! -d "$BAGEL_EXAMPLE_DIR" ]; then
    mkdir -p "$(dirname "$BAGEL_EXAMPLE_ZIP")"
    if [ ! -f "$BAGEL_EXAMPLE_ZIP" ]; then
        wget -O "$BAGEL_EXAMPLE_ZIP" "$BAGEL_EXAMPLE_URL"
    fi
    unzip "$BAGEL_EXAMPLE_ZIP" -d "$(dirname "$BAGEL_EXAMPLE_DIR")"
fi

require_path() {
    local kind=$1
    local path=$2
    if [ "$kind" = "dir" ] && [ ! -d "$path" ]; then
        echo "Missing required BAGEL example directory: $path" >&2
        exit 1
    fi
    if [ "$kind" = "file" ] && [ ! -f "$path" ]; then
        echo "Missing required BAGEL example file: $path" >&2
        exit 1
    fi
}

require_path dir "$BAGEL_EXAMPLE_DIR/t2i"
require_path dir "$BAGEL_EXAMPLE_DIR/editing/seedxedit_multi"
require_path file "$BAGEL_PARQUET_INFO_FILE"
require_path dir "$BAGEL_EXAMPLE_DIR/vlm/images"
require_path file "$BAGEL_EXAMPLE_DIR/vlm/llava_ov_si.jsonl"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export BAGEL_EXAMPLE_DIR

python3 -m verl_omni.trainer.main_diffusion \
    "data.train_files=$BAGEL_EXAMPLE_DIR" \
    "data.val_files=$BAGEL_EXAMPLE_DIR" \
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    data.max_prompt_length=8192 \
    data.dataloader_num_workers=$NUM_WORKERS \
    data.trust_remote_code=True \
    "data.custom_cls.path=file://$BAGEL_DATASET_ADAPTER" \
    data.custom_cls.name=BagelExampleSFTDataset \
    data.custom_cls.collate_fn=bagel_example_sft_collate_fn \
    "+data.custom_cls.bagel_example_dir=$BAGEL_EXAMPLE_DIR" \
    "+data.custom_cls.dataset_config_file=$BAGEL_DATA_CONFIG" \
    "+data.custom_cls.parquet_info_file=$BAGEL_PARQUET_INFO_FILE" \
    +data.custom_cls.expected_num_tokens=$EXPECTED_NUM_TOKENS \
    +data.custom_cls.max_num_tokens=$MAX_NUM_TOKENS \
    +data.custom_cls.max_num_tokens_per_sample=$MAX_NUM_TOKENS_PER_SAMPLE \
    +data.custom_cls.prefer_buffer_before=$EXPECTED_NUM_TOKENS \
    +data.custom_cls.max_buffer_size=50 \
    +data.custom_cls.max_latent_size=64 \
    +data.custom_cls.use_flex=True \
    +data.custom_cls.num_packed_batches=$NUM_PACKED_BATCHES \
    algorithm.trainer_type=sft \
    algorithm.sample_source=offline \
    "actor_rollout_ref.model.path=$model_name" \
    "actor_rollout_ref.model.tokenizer_path=$model_name" \
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
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=bagel_sft \
    trainer.experiment_name=bagel_example_lora \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=500 \
    trainer.test_freq=0 \
    trainer.total_epochs=3 \
    trainer.total_training_steps=3000 "$@"
