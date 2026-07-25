#!/usr/bin/env bash
# Qwen3-Omni Thinker GSPO full-param RL — video→text (NExT-QA video reasoning).
# Hardware: Atlas 800T A3 (16x NPUs). tf5 + video rope patch.
# Pure video: data has only videos, prompt contains <video> placeholder.
# Uses default RLHFDataset (has_visual auto video frame extraction via qwen_vl_utils).
# Anti-collapse: lr=2e-6, batch=32, clip=1e-3, max_resp=4096, shuffle=true.
# reward=choice_reward (<answer>X</answer> letter exact match, reused from AVQA recipe).
# video item dict {fps:2.0, max_frames:32} caps video tokens (~8-10k) to fit max_model_len=16384.
# filter_overlong_prompts=false: datasets.map(num_proc) stalls on heavy video decode;
#   max_frames already bounds prompt length, so filtering is unnecessary.
# mm_processor_kwargs.fps=2.0 keeps processor's video_second_per_grid consistent with qwen_vl_utils fps.
set -x

export CPATH=/usr/include${CPATH:+:$CPATH}
export VLLM_ASCEND_ENABLE_NZ=0
export VERL_USE_EXTERNAL_MODULES=verl_omni,verl_omni.models.transformers.qwen3_omni_thinker

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-Omni-30B-A3B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/video2text/train.parquet"}
VAL_FILE=${VAL_FILE:-"$HOME/data/video2text/validation.parquet"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_CONFIG="${SCRIPT_DIR}/qwen3_omni_thinker_only_npu.yaml"

NUM_GPUS_ACTOR_ROLLOUT_REWARD=16
ROLLOUT_TP=2

python3 -m verl.trainer.main_ppo \
    --config-path="${SCRIPT_DIR}/config" \
    --config-name=qwen3_omni_thinker_gspo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.shuffle=true \
    data.max_response_length=4096 \
    data.max_prompt_length=12000 \
    data.train_batch_size=32 \
    data.filter_overlong_prompts=false \
    data.filter_overlong_prompts_workers=8 \
    ++data.mm_processor_kwargs.fps=2.0 \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.external_lib=verl_omni.models.transformers.qwen3_omni_thinker \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_configs_path="${STAGE_CONFIG}" \
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP)) \
    reward.custom_reward_function.path=verl_omni/utils/reward_score/choice_reward.py \
    reward.custom_reward_function.name=compute_score \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.clip_ratio_low=1e-3 \
    actor_rollout_ref.actor.clip_ratio_high=1e-3 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=qwen3_omni_thinker_rl \
    trainer.experiment_name=gspo_video2text_npu \
    trainer.n_gpus_per_node=${NUM_GPUS_ACTOR_ROLLOUT_REWARD} \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=25 \
    trainer.val_before_train=true \
    "$@"
