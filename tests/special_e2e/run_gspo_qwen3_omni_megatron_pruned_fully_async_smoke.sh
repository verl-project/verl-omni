#!/usr/bin/env bash
# Pruned Qwen3-Omni Thinker + Megatron full-model trainer + standalone
# vLLM-Omni rollout, using verl fully_async_policy resource split.
set -xeuo pipefail

export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
export VLLM_USE_V1=${VLLM_USE_V1:-0}
export VLLM_DISABLE_COMPILE_CACHE=${VLLM_DISABLE_COMPILE_CACHE:-1}
export VERL_USE_EXTERNAL_MODULES=${VERL_USE_EXTERNAL_MODULES:-verl_omni,verl_omni.models.transformers.qwen3_omni_thinker}
export VERL_OMNI_SKIP_PIPELINES=${VERL_OMNI_SKIP_PIPELINES:-1}
export VERL_OMNI_SKIP_REWARD_LOOP=${VERL_OMNI_SKIP_REWARD_LOOP:-1}
export VERL_FORCE_SHM_WEIGHT_TRANSFER=${VERL_FORCE_SHM_WEIGHT_TRANSFER:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=${RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO:-0}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export CPATH=/usr/include${CPATH:+:$CPATH}

CONDA_ENV=${CONDA_ENV:-/nfs/ml-training-ssd/users/liuwei/verl_mega_async}
source "${CONDA_ENV}/bin/activate"

export CUDA_HOME=${VERL_CUDA_HOME:-/usr/local/cuda-12.6}
_clean_ld_parts=()
IFS=: read -ra _ld_parts <<<"${LD_LIBRARY_PATH:-}"
for _ld_part in "${_ld_parts[@]}"; do
  if [[ "${_ld_part}" == *"/nvidia/cu13/lib" ]]; then
    continue
  fi
  _clean_ld_parts+=("${_ld_part}")
done
LD_LIBRARY_PATH="$(IFS=:; echo "${_clean_ld_parts[*]}")"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
unset _clean_ld_parts _ld_parts _ld_part

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASYNC_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
VERL_ROOT="${ASYNC_ROOT}/verl"
VLLM_OMNI_ROOT="${ASYNC_ROOT}/vllm-omni"
MBRIDGE_ROOT="${MBRIDGE_ROOT_OVERRIDE:-${ASYNC_ROOT}/megatron-bridge}"
MCORE_ROOT="${MCORE_ROOT_OVERRIDE:-${MBRIDGE_ROOT}/3rdparty/Megatron-LM}"
export PYTHONPATH="${REPO_ROOT}:${VERL_ROOT}:${VLLM_OMNI_ROOT}:${MBRIDGE_ROOT}/src:${MCORE_ROOT}:${PYTHONPATH:-}"
cd "${VERL_ROOT}"

NUM_GPUS=${NUM_GPUS:-4}
TRAIN_GPUS=${TRAIN_GPUS:-2}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-2}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export CUDA_VISIBLE_DEVICES

MODEL_PATH=${MODEL_PATH:-/nfs/ml-training-ssd/users/liuwei/models/Qwen3-Omni-30B-A3B-Instruct-smoke-chattemplate}
TRAIN_FILES=${TRAIN_FILES:-/nfs/ml-training-ssd/users/liuwei/data/gsm8k_verl_prompt/train.parquet}
VAL_FILES=${VAL_FILES:-/nfs/ml-training-ssd/users/liuwei/data/gsm8k_verl_prompt/test.parquet}
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
LOG_FILE=${LOG_FILE:-${LOG_ROOT}/qwen3_omni_megatron_pruned_fully_async_smoke_${RUN_ID}.log}
if [[ "${VERL_OMNI_LOG_TEE_INITIALIZED:-0}" != "1" ]]; then
  export VERL_OMNI_LOG_TEE_INITIALIZED=1
  exec > >(tee -a "${LOG_FILE}") 2>&1
fi
echo "[info] LOG_FILE=${LOG_FILE}"

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-256}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-64}
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
TOTAL_ROLLOUT_STEPS=${TOTAL_ROLLOUT_STEPS:-2}

ACTOR_TP=${ACTOR_TP:-1}
ACTOR_PP=${ACTOR_PP:-1}
ACTOR_CP=${ACTOR_CP:-1}
ACTOR_EP=${ACTOR_EP:-1}
ACTOR_ETP=${ACTOR_ETP:-1}
REF_TP=${REF_TP:-${ACTOR_TP}}
REF_PP=${REF_PP:-${ACTOR_PP}}
REF_CP=${REF_CP:-${ACTOR_CP}}
REF_EP=${REF_EP:-${ACTOR_EP}}
REF_ETP=${REF_ETP:-${ACTOR_ETP}}

ROLLOUT_TP=${ROLLOUT_TP:-${ROLLOUT_GPUS}}
OFFLOAD=${OFFLOAD:-False}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-False}
SEQUENCE_PARALLEL=${SEQUENCE_PARALLEL:-False}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-local}
MASKED_SOFTMAX_FUSION=${MASKED_SOFTMAX_FUSION:-False}
MOE_PERMUTE_FUSION=${MOE_PERMUTE_FUSION:-False}
GRADIENT_ACCUMULATION_FUSION=${GRADIENT_ACCUMULATION_FUSION:-False}

python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name=fully_async_ppo_megatron_trainer \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=True \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.prompt_key=prompt \
    data.dataloader_num_workers=0 \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.gen_batch_size=1 \
    data.train_batch_size=0 \
    data.val_max_samples=2 \
    data.return_raw_chat=True \
    data.trust_remote_code=True \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.external_lib=verl_omni.models.transformers.qwen3_omni_thinker \
    actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora.rank=0 \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.lr_decay_steps=1 \
    actor_rollout_ref.actor.optim.total_training_steps=1 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False \
    actor_rollout_ref.actor.megatron.use_remove_padding="${USE_REMOVE_PADDING}" \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${ACTOR_TP}" \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${ACTOR_PP}" \
    actor_rollout_ref.actor.megatron.context_parallel_size="${ACTOR_CP}" \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size="${ACTOR_EP}" \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size="${ACTOR_ETP}" \
    actor_rollout_ref.actor.megatron.sequence_parallel="${SEQUENCE_PARALLEL}" \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel="${SEQUENCE_PARALLEL}" \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion="${MASKED_SOFTMAX_FUSION}" \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion="${MOE_PERMUTE_FUSION}" \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion="${GRADIENT_ACCUMULATION_FUSION}" \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend="${ATTENTION_BACKEND}" \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.actor.megatron.param_offload="${OFFLOAD}" \
    actor_rollout_ref.actor.megatron.optimizer_offload="${OFFLOAD}" \
    actor_rollout_ref.actor.megatron.grad_offload="${OFFLOAD}" \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.data_parallel_size=1 \
    actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_seqs=4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.load_format=dummy \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=256 \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_configs_path="${STAGE_CONFIG}" \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_init_timeout=1200 \
    actor_rollout_ref.ref.strategy=megatron \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.megatron.use_mbridge=True \
    actor_rollout_ref.ref.megatron.vanilla_mbridge=False \
    actor_rollout_ref.ref.megatron.use_remove_padding="${USE_REMOVE_PADDING}" \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${REF_TP}" \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="${REF_PP}" \
    actor_rollout_ref.ref.megatron.context_parallel_size="${REF_CP}" \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size="${REF_EP}" \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size="${REF_ETP}" \
    actor_rollout_ref.ref.megatron.sequence_parallel="${SEQUENCE_PARALLEL}" \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.sequence_parallel="${SEQUENCE_PARALLEL}" \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.masked_softmax_fusion="${MASKED_SOFTMAX_FUSION}" \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.moe_permute_fusion="${MOE_PERMUTE_FUSION}" \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.gradient_accumulation_fusion="${GRADIENT_ACCUMULATION_FUSION}" \
    ++actor_rollout_ref.ref.megatron.override_transformer_config.attention_backend="${ATTENTION_BACKEND}" \
    actor_rollout_ref.ref.megatron.param_offload="${OFFLOAD}" \
    reward.num_workers=1 \
    reward.reward_manager.name=naive \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=qwen3_omni_megatron_async_smoke \
    trainer.experiment_name="qwen3_omni_megatron_pruned_fully_async_smoke_${RUN_ID}" \
    trainer.n_gpus_per_node="${TRAIN_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.resume_mode=disable \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.default_local_dir="${LOG_ROOT}/ckpts/qwen3_omni_megatron_pruned_fully_async_smoke_${RUN_ID}" \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node="${ROLLOUT_GPUS}" \
    rollout.n=2 \
    rollout.total_rollout_steps="${TOTAL_ROLLOUT_STEPS}" \
    async_training.staleness_threshold=0 \
    async_training.trigger_parameter_sync_step=1 \
    async_training.require_batches=1 \
    async_training.partial_rollout=True \
    async_training.use_trainer_do_validate=False \
    "$@"
