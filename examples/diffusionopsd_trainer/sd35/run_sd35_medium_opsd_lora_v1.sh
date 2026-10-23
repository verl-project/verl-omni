#!/usr/bin/env bash
# SD3.5-Medium LoRA recipe for DiffusionOPSD (V1 trainer, sync mode).
#
# DiffusionOPSD is an online self-distillation method: the frozen `old` LoRA
# adapter rolls out deterministically, the actor builds explicit clean-output
# targets by ascending/descending a differentiable reward in a relative L2
# trust region around an on-policy anchor, and a detached two-branch
# forward-process loss fits the current policy to those targets.
#
# This is the paper-v1 configuration: 512^2, 10 deterministic CFG-free rollout
# steps, query noise sigma*=0.278, rho=0.10, M_tgt=2, beta=1.0, c_adv=5.
# Data, LoRA targets and trainer entry point follow the DGPO SD3.5 recipe; the
# old-policy EMA uses `linear_to_0_5`, which is exactly
# eta_beh = min(1e-3 * update, 0.5).
#
# Status: the loss, config, the `diffusion_opsd_model` engine, the SD3.5 adapters,
# CPU tests and an offline e2e smoke are in place, and this recipe was validated
# on GPU at 512x512 for 100 steps (see examples/diffusionopsd_trainer for the
# measured curve).
#
# The endpoint omega reward is the native PickScore named model configured below.
# It must use the same checkpoint as the differentiable target reward
# (`diffusion_loss.opsd_reward_model_path`): the endpoint score goes through the
# native PIL preprocessing while target construction resizes bicubically, so the
# two paths differ by a small, bounded amount rather than being identical.
set -x

# Set WORKSPACE to any writable directory; defaults to $HOME.
WORKSPACE=${WORKSPACE:-$HOME}

pickscore_train_path=$WORKSPACE/data/pickscore_sfw/sd3/train.parquet
pickscore_test_path=$WORKSPACE/data/pickscore_sfw/sd3/test.parquet

model_name=${OPSD_MODEL_PATH:-stabilityai/stable-diffusion-3.5-medium}
# PickScore v1 checkpoint shared by the endpoint reward and the differentiable
# target reward.
pickscore_model_path=${PICKSCORE_MODEL_PATH:-yuvalkirstain/PickScore_v1}
custom_chat_template='{% for message in messages %}{% if message['\''role'\''] == '\''user'\'' %}{{ message['\''content'\''] }}{% endif %}{% endfor %}'

NUM_GPUS_ACTOR_ROLLOUT=${NUM_GPUS_ACTOR_ROLLOUT:-2}
NUM_GPUS_REWARD=${NUM_GPUS_REWARD:-1}
ROLLOUT_TP=1
REWARD_TP=1
IMAGE_RESOLUTION=${IMAGE_RESOLUTION:-512}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
# The epoch budget must not run out before TOTAL_TRAINING_STEPS: a small prompt set gives only
# a few steps per epoch, so the cap is raised well above it (100 steps / 3 per epoch = 34).
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1000}
ATTN_BACKEND=native
ROLLOUT_ATTN_BACKEND=TORCH_SDPA
MAX_NUM_SEQS=256
# One complete PickScore replica per native reward GPU bundle.
NATIVE_REWARD_DEVICES=$(seq -s, 0 $((NUM_GPUS_REWARD - 1)))
# Paper v1 samples K=24 images per prompt; 8 keeps the first GPU validation
# affordable while preserving group-normalized endpoint weights.
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}

if [ "${FA3:-0}" = "1" ]; then
    ATTN_BACKEND="_flash_3_varlen_hub"
    ROLLOUT_ATTN_BACKEND=FLASH_ATTN_3_HUB
fi

ENGINE=vllm_omni
reward_function_path=pkg://verl_omni.utils.reward_score.pickscore_reward

python3 -m verl_omni.trainer.main_diffusion_v1 \
    data.train_files="$pickscore_train_path" \
    data.val_files="$pickscore_test_path" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_max_samples=32 \
    data.max_prompt_length=512 \
    data.truncation=error \
    data.seed=42 \
    algorithm.trainer_type=direct_preference \
    algorithm.sample_source=online \
    algorithm.global_std=True \
    algorithm.old_policy_decay_schedule=linear_to_0_5 \
    algorithm.old_policy_update_interval=1 \
    actor_rollout_ref.model.algorithm=opsd \
    actor_rollout_ref.model.model_type=diffusion_opsd_model \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.custom_chat_template="\"$custom_chat_template\"" \
    'actor_rollout_ref.model.extra_tokenizers={clip: {path: tokenizer, max_length: 77}, t5: {path: tokenizer_3, max_length: 256}}' \
    actor_rollout_ref.model.attn_backend=$ATTN_BACKEND \
    actor_rollout_ref.rollout.rollout_attn_backend=$ROLLOUT_ATTN_BACKEND \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.policy_state_adapters='["default","old"]' \
    actor_rollout_ref.model.target_modules="['to_q','to_k','to_v','to_out.0','add_q_proj','add_k_proj','add_v_proj','to_add_out']" \
    actor_rollout_ref.actor.diffusion_loss.loss_mode=opsd \
    actor_rollout_ref.actor.diffusion_loss.opsd_rho=0.10 \
    actor_rollout_ref.actor.diffusion_loss.opsd_n_ascent=2 \
    actor_rollout_ref.actor.diffusion_loss.opsd_eta=1.0 \
    actor_rollout_ref.actor.diffusion_loss.opsd_branch_beta=1.0 \
    actor_rollout_ref.actor.diffusion_loss.opsd_query_sigma=0.278 \
    actor_rollout_ref.actor.diffusion_loss.opsd_eps_gamma=1e-5 \
    actor_rollout_ref.actor.diffusion_loss.opsd_reward_kind=pickscore \
    actor_rollout_ref.actor.diffusion_loss.opsd_reward_model_path=$pickscore_model_path \
    actor_rollout_ref.actor.diffusion_loss.opsd_target_micro_batch=6 \
    actor_rollout_ref.actor.diffusion_loss.adv_clip_max=5.0 \
    actor_rollout_ref.actor.optim.lr=3e-4 \
    actor_rollout_ref.actor.optim.weight_decay=0.0001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$N_RESP_PER_PROMPT \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_no_sync_for_gradient_accumulation=true \
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.n=$N_RESP_PER_PROMPT \
    actor_rollout_ref.rollout.seed=42 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT / ROLLOUT_TP)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.rollout_adapter=old \
    actor_rollout_ref.rollout.pipeline.height=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.width=$IMAGE_RESOLUTION \
    actor_rollout_ref.rollout.pipeline.num_inference_steps=10 \
    actor_rollout_ref.rollout.pipeline.guidance_scale=1.0 \
    actor_rollout_ref.rollout.pipeline.max_sequence_length=256 \
    actor_rollout_ref.rollout.max_prompt_embed_length=333 \
    actor_rollout_ref.rollout.algo.noise_level=0.0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=$MAX_NUM_SEQS \
    actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=28 \
    actor_rollout_ref.rollout.val_kwargs.algo.noise_level=0.0 \
    reward.num_workers=$((NUM_GPUS_REWARD / REWARD_TP)) \
    reward.reward_model.enable=False \
    reward.reward_model.enable_resource_pool=True \
    reward.reward_model.nnodes=1 \
    reward.reward_model.n_gpus_per_node=$NUM_GPUS_REWARD \
    reward.reward_manager.name=MultiVisualRewardManager \
    +reward.models.pickscore.backend=native \
    +reward.models.pickscore.offload=False \
    +reward.models.pickscore.model_path=$pickscore_model_path \
    "+reward.models.pickscore.placement.devices=[$NATIVE_REWARD_DEVICES]" \
    +reward.models.pickscore.executor.model=verl_omni.utils.reward_score.pickscore_reward:PickScoreNativeModel \
    +reward.reward_functions.pickscore.path=$reward_function_path \
    +reward.reward_functions.pickscore.name=compute_score_pickscore_native \
    +reward.reward_functions.pickscore.weight=1.0 \
    +reward.reward_functions.pickscore.required=True \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name=opsd \
    trainer.experiment_name=sd35_medium_opsd_lora_v1 \
    trainer.log_val_generations=8 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.use_v1=true \
    trainer.v1.trainer_mode=sync \
    trainer.v1.sampler.drop_incomplete_groups=True "$@"
