#!/usr/bin/env bash
# Two reward terms sharing one judge deployment, each with its own user prompt.
#
# Same recipe as run_sd35_medium_ocr_lora_v1.sh, but the Qwen2.5-VL judge is
# registered once as `judge` and two genrm_ocr terms point at it:
#   ocr_default: the scorer's default user prompt
#   ocr_custom:  a custom user prompt set on the term
# Each term is logged as critic/reward/<term>/mean. Set SHARED_JUDGE=0 to run
# the single-term baseline (one `ocr` term on the same judge) for comparison.
#
# Defaults fit 1 actor/rollout GPU + 1 judge GPU (32 GB each).
set -x

WORKSPACE=${OCR_WORKSPACE:-${WORKSPACE:-$HOME}}

ocr_train_path=$WORKSPACE/data/ocr/sd3/train.parquet
ocr_test_path=$WORKSPACE/data/ocr/sd3/test.parquet

model_name=${MODEL_PATH:-stabilityai/stable-diffusion-3.5-medium}
reward_model_name=${REWARD_MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}
reward_function_path=verl_omni/utils/reward_score/genrm_ocr.py
custom_chat_template='{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}{{ message['\''content'\''] }}{% endif %}{% endfor %}'
custom_user_prompt=${CUSTOM_USER_PROMPT:-"Read the text in this image and output it exactly, with nothing else."}

NUM_GPUS_ACTOR_ROLLOUT=${NUM_GPUS_ACTOR_ROLLOUT:-1}
NUM_GPUS_REWARD=${NUM_GPUS_REWARD:-1}
IMAGE_RESOLUTION=${IMAGE_RESOLUTION:-384}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-20}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
SHARED_JUDGE=${SHARED_JUDGE:-1}

if [ "$SHARED_JUDGE" = "1" ]; then
    experiment_name=sd35_medium_ocr_shared_judge
    reward_terms=(
        +reward.reward_functions.ocr_default.model=judge
        +reward.reward_functions.ocr_default.path=$reward_function_path
        +reward.reward_functions.ocr_default.name=compute_score_ocr
        +reward.reward_functions.ocr_default.weight=0.5
        +reward.reward_functions.ocr_default.required=True
        +reward.reward_functions.ocr_default.use_rollout_sampling_params=True
        +reward.reward_functions.ocr_custom.model=judge
        +reward.reward_functions.ocr_custom.path=$reward_function_path
        +reward.reward_functions.ocr_custom.name=compute_score_ocr
        +reward.reward_functions.ocr_custom.weight=0.5
        +reward.reward_functions.ocr_custom.required=True
        +reward.reward_functions.ocr_custom.use_rollout_sampling_params=True
        "+reward.reward_functions.ocr_custom.user_prompt=\"$custom_user_prompt\""
    )
else
    experiment_name=sd35_medium_ocr_single_judge
    reward_terms=(
        +reward.reward_functions.ocr.model=judge
        +reward.reward_functions.ocr.path=$reward_function_path
        +reward.reward_functions.ocr.name=compute_score_ocr
        +reward.reward_functions.ocr.weight=1.0
        +reward.reward_functions.ocr.required=True
        +reward.reward_functions.ocr.use_rollout_sampling_params=True
    )
fi

python3 -m verl_omni.trainer.main_diffusion_v1 \
    data.train_files="$ocr_train_path" \
    data.val_files="$ocr_test_path" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_max_samples=32 \
    data.max_prompt_length=512 \
    data.truncation=error \
    data.seed=42 \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    algorithm.global_std=True \
    algorithm.train_timestep_range="[0,7]" \
    algorithm.train_timestep_count=4 \
    algorithm.old_policy_decay_schedule=linear_to_0_3 \
    algorithm.old_policy_update_interval=1 \
    actor_rollout_ref.model.algorithm=dgpo \
    actor_rollout_ref.model.model_type=diffusion_nft_model \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.custom_chat_template="\"$custom_chat_template\"" \
    'actor_rollout_ref.model.extra_tokenizers={clip: {path: tokenizer, max_length: 77}, t5: {path: tokenizer_3, max_length: 256}}' \
    actor_rollout_ref.model.attn_backend=native \
    actor_rollout_ref.rollout.rollout_attn_backend=TORCH_SDPA \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out']" \
    actor_rollout_ref.actor.diffusion_loss.dgpo_beta=100.0 \
    actor_rollout_ref.actor.diffusion_loss.dgpo_clip_range=0.01 \
    actor_rollout_ref.actor.diffusion_loss.ref_kl_coef=0.02 \
    actor_rollout_ref.actor.diffusion_loss.adv_clip_max=5.0 \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$N_RESP_PER_PROMPT \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_no_sync_for_gradient_accumulation=true \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.n=$N_RESP_PER_PROMPT \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.agent.num_workers=$NUM_GPUS_ACTOR_ROLLOUT \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.rollout_adapter=old \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.pipeline.height=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.width=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=4.5 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.max_prompt_embed_length=333 \
    actor_rollout_ref.rollout.algo.noise_level=0.0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=64 \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=28 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    reward.num_workers=$NUM_GPUS_REWARD \
    reward.reward_model.enable=False \
    reward.reward_model.enable_resource_pool=True \
    reward.reward_model.nnodes=1 \
    reward.reward_model.n_gpus_per_node=$NUM_GPUS_REWARD \
    reward.reward_manager.name=MultiVisualRewardManager \
    +reward.models.judge.backend=engine \
    +reward.models.judge.offload=False \
    +reward.models.judge.model_path=$reward_model_name \
    +reward.models.judge.n_gpus_per_node=$NUM_GPUS_REWARD \
    +reward.models.judge.nnodes=1 \
    +reward.models.judge.rollout.name=vllm \
    +reward.models.judge.rollout.gpu_memory_utilization=0.85 \
    +reward.models.judge.rollout.tensor_model_parallel_size=1 \
    +reward.models.judge.rollout.enforce_eager=False \
    +reward.models.judge.rollout.max_model_len=8192 \
    "${reward_terms[@]}" \
    trainer.logger='["console"]' \
    trainer.project_name=dgpo \
    trainer.experiment_name=$experiment_name \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=sync \
    trainer.v1.sampler.drop_incomplete_groups=True "$@"
