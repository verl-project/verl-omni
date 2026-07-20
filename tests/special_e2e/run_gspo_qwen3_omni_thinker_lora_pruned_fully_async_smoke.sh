#!/usr/bin/env bash
# Pruned Qwen3-Omni Thinker GSPO + LoRA fully-async smoke.
#
# Default split on one 4-GPU node:
#   - 2 GPUs: FSDP LoRA actor/ref trainer
#   - 2 GPUs: standalone vLLM-Omni AR rollout
set -xeuo pipefail

export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
export VLLM_USE_V1=${VLLM_USE_V1:-0}
export VLLM_DISABLE_COMPILE_CACHE=${VLLM_DISABLE_COMPILE_CACHE:-1}
export VERL_USE_EXTERNAL_MODULES=${VERL_USE_EXTERNAL_MODULES:-verl_omni,verl_omni.models.transformers.qwen3_omni_thinker}
export VERL_OMNI_SKIP_PIPELINES=${VERL_OMNI_SKIP_PIPELINES:-1}
export VERL_OMNI_SKIP_REWARD_LOOP=${VERL_OMNI_SKIP_REWARD_LOOP:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}
export CPATH=/usr/include${CPATH:+:$CPATH}

CONDA_ENV=${CONDA_ENV:-/nfs/ml-training-ssd/users/liuwei/verl_mega_async}
source "${CONDA_ENV}/bin/activate"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASYNC_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
VERL_ROOT="${ASYNC_ROOT}/verl"
export PYTHONPATH="${REPO_ROOT}:${VERL_ROOT}:${PYTHONPATH:-}"
cd "${VERL_ROOT}"

NUM_GPUS=${NUM_GPUS:-4}
TRAIN_GPUS=${TRAIN_GPUS:-2}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export CUDA_VISIBLE_DEVICES

MODEL_PATH=${MODEL_PATH:-/nfs/ml-training-ssd/users/liuwei/models/Qwen3-Omni-30B-A3B-Instruct-smoke-chattemplate}
TRAIN_FILE=${TRAIN_FILE:-/nfs/ml-training-ssd/users/liuwei/data/gsm8k_verl_prompt/train.parquet}
VAL_FILE=${VAL_FILE:-/nfs/ml-training-ssd/users/liuwei/data/gsm8k_verl_prompt/test.parquet}
STAGE_CONFIG=${STAGE_CONFIG:-${REPO_ROOT}/tests/special_e2e/qwen3_omni_thinker_only_tp2_pruned_async_smoke.yaml}

CACHE_ROOT=${CACHE_ROOT:-/nfs/ml-training-ssd/users/liuwei/verl_mega_async_pruned_smoke_cache}
export HF_HOME=${HF_HOME:-${CACHE_ROOT}/hf_home}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
export TORCH_HOME=${TORCH_HOME:-${CACHE_ROOT}/torch}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torchinductor}
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOG_ROOT=${LOG_ROOT:-/nfs/ofs-llab-hdd/users/liuwei/omni/qwen3_omni_megatron_rl/async_fullmodel/outputs/logs}
mkdir -p "${LOG_ROOT}"

EXCLUDE_MODULES=".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*"

python3 -m verl.experimental.fully_async_policy.fully_async_main \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=256 \
    data.max_response_length=64 \
    data.gen_batch_size=1 \
    data.train_batch_size=0 \
    data.val_max_samples=2 \
    data.return_raw_chat=True \
    data.trust_remote_code=True \
    \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.external_lib=verl_omni.models.transformers.qwen3_omni_thinker \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.lora_rank=8 \
    actor_rollout_ref.model.lora_alpha=16 \
    actor_rollout_ref.model.target_modules="all-linear" \
    actor_rollout_ref.model.exclude_modules="${EXCLUDE_MODULES}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    ++actor_rollout_ref.actor.freeze_vision_tower=True \
    \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_GPUS}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256 \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_configs_path="${STAGE_CONFIG}" \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar \
    \
    actor_rollout_ref.ref.strategy=fsdp \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=True \
    \
    reward.reward_manager.name=dapo \
    \
    trainer.logger=console \
    trainer.project_name=verl-qwen3-omni-async-smoke \
    trainer.experiment_name="qwen3-omni-pruned-fully-async-smoke-${RUN_ID}" \
    trainer.n_gpus_per_node="${TRAIN_GPUS}" \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${LOG_ROOT}/ckpts/qwen3-omni-pruned-fully-async-smoke-${RUN_ID}" \
    \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node="${ROLLOUT_GPUS}" \
    rollout.total_rollout_steps=2 \
    \
    async_training.staleness_threshold=0 \
    async_training.trigger_parameter_sync_step=1 \
    async_training.require_batches=1 \
    async_training.partial_rollout=True \
    async_training.use_trainer_do_validate=False \
    "$@"
