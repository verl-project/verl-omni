# Boogu-Image lora RL, vllm_omni rollout
#
# Prerequisites (on top of the standard install):
#   pip install "boogu-image @ git+https://github.com/boogu-project/Boogu-Image.git"
# The training engine loads the checkpoint's canonical transformer through
# diffusers AutoModel + trust_remote_code; the checkpoint's transformer_boogu.py
# is a shim that re-exports the class from the boogu package.
#
# Data: examples/flowgrpo_trainer/data_process/boogu_image_ocr.py
# (its chat template replicates the upstream Boogu system prompts verbatim,
# including the empty-negative-prompt quirk — do not edit them).
#
# fsdp_layer_prefixes below is required, not tuning: Boogu's DiT blocks are not
# named transformer_blocks.*, so LoRA weight-sync collection gathers 0 params
# without it.
#
# Reference throughput, 4xA800-80G, this configuration: 716 s/step for
# 32x16 = 512 images, plus 2900 s per validation over the full 1018-prompt test
# set. The rollout dominates until request-level packing is on -- without
# max_num_seqs the engine clamps to one image per forward and gen alone costs 4x
# more. Each checkpoint is ~21 GB; budget disk against save_freq.
#
# Validation images are written nowhere by default: log_val_generations only
# builds the W&B table, and under WANDB_MODE=offline that cell holds the string
# "Image" rather than the file. Set trainer.validation_data_dir=<path> to get
# them on disk -- worth doing, because a rising OCR score says nothing about
# whether the glyphs came out right.
#
# Left at defaults on purpose — whether they are needed depends on the box,
# not on Boogu:
#   attn_backend=native, rollout_attn_backend=TORCH_SDPA
#       No FA3 requirement on this path; set these if the transformer fails to
#       fetch kernels-community/flash-attn3 from the Hub (no egress / untrusted
#       remote code). Where the kernel loads, the default is faster.
#   reward.reward_model.rollout.max_model_len=8192
#   actor_rollout_ref.rollout.gpu_memory_utilization
#       Qwen3-VL advertises a 262144 context, and the reward engine shares these
#       GPUs with the rollout. Cap the context, or lower the rollout's memory
#       share, if either engine cannot reserve what it asks for.
#   actor.fsdp_config.param_offload / optimizer_offload
#       Both are ON below. Offload is usually described as a GPU-poor tradeoff,
#       but here it costs nothing measurable: update_actor took 1265s with it and
#       1200s without, so this step is compute-bound, not bound by parameter
#       movement. Keeping it on is what leaves room for the colocated 8B GenRM.
set -x

# Set WORKSPACE to any writable directory; defaults to $HOME
WORKSPACE=${WORKSPACE:-$HOME}

ocr_train_path=$WORKSPACE/data/ocr/boogu_image/train.parquet
ocr_test_path=$WORKSPACE/data/ocr/boogu_image/test.parquet

model_name=Boogu/Boogu-Image-0.1-Base
reward_model_name=Qwen/Qwen3-VL-8B-Instruct
reward_function_path=verl_omni/utils/reward_score/genrm_ocr.py

NUM_GPUS_ACTOR_ROLLOUT_REWARD=4
# The vllm-omni BooguImagePipeline supports neither TP nor SP nor CFG-parallel.
ROLLOUT_TP=1
REWARD_TP=4

ENGINE=vllm_omni
REWARD_ENGINE=vllm
# Request-level packing. Set it: the config default is 1024, and a T2I rollout
# that packs 1024 images into one forward runs out of memory long before it
# finishes. Keep it equal to log_prob_micro_batch_size_per_gpu below -- rollout
# and the old-log-prob recompute then see the same bf16 batch shape, and
# FlowGRPO's ratio stays meaningful.
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
REQUEST_BATCH_MAX_WAIT_MS=${REQUEST_BATCH_MAX_WAIT_MS:-10}
# Matches the Qwen-Image recipes, which train at the 512 pipeline default. The
# OCR reward model reads the image far below 1024, so the extra latent tokens
# buy resolution the reward never sees while costing them in every rollout,
# old-log-prob, and actor forward.
IMAGE_RESOLUTION=${IMAGE_RESOLUTION:-512}

python3 -m verl_omni.trainer.main_diffusion \
    data.train_files=$ocr_train_path \
    data.val_files=$ocr_test_path \
    data.train_batch_size=32 \
    data.max_prompt_length=256 \
    actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.tokenizer_path=$model_name/processor \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=128 \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','img_to_q','img_to_k','img_to_v','img_out','instruct_to_q','instruct_to_k','instruct_to_v','instruct_out','feed_forward.linear_1','feed_forward.linear_2','feed_forward.linear_3','img_feed_forward.linear_1','img_feed_forward.linear_2','img_feed_forward.linear_3']" \
    actor_rollout_ref.model.fsdp_layer_prefixes="['double_stream_layers.','single_stream_layers.','context_refiner.','noise_refiner.','ref_image_refiner.']" \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${MAX_NUM_SEQS} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=${MAX_NUM_SEQS} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.request_batch_max_wait_ms=${REQUEST_BATCH_MAX_WAIT_MS} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.pipeline.height=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.width=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.guidance_scale=4.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.algo.noise_level=1.2 \
    actor_rollout_ref.rollout.algo.sde_type="sde" \
    actor_rollout_ref.rollout.algo.sde_window_size=2 \
    actor_rollout_ref.rollout.algo.sde_window_range="[0,5]" \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=50 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${MAX_NUM_SEQS} \
    reward.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / REWARD_TP)) \
    reward.reward_model.enable=True \
    reward.reward_model.model_path=$reward_model_name \
    reward.reward_model.rollout.name=$REWARD_ENGINE \
    reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
    reward.custom_reward_function.path=$reward_function_path \
    reward.custom_reward_function.name=compute_score_ocr \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=flow_grpo \
    trainer.experiment_name=boogu_image_ocr_lora \
    trainer.log_val_generations=8 \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=300 "$@"
