#!/usr/bin/env bash
# RPCO stage 3: multi-task agentic GRPO on UniCoT reflect + plan rows.
#
# Start the frozen image and judge sidecars first, then launch this script on
# trainer GPUs:
#   CUDA_VISIBLE_DEVICES=2,3 N_GPUS=2 \
#     bash examples/agenticllmgrpo_trainer/agent_llm/run_agentic_rpco.sh
set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VERLOMNI_ROOT="${REPO_ROOT}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
N_GPUS="${N_GPUS:-2}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.30}"
TOTAL_STEPS="${TOTAL_STEPS:-200}"
ROLLOUT_N="${ROLLOUT_N:-8}"

if (( N_GPUS % ROLLOUT_TP != 0 )); then
  echo "[ERROR] N_GPUS=${N_GPUS} must be divisible by ROLLOUT_TP=${ROLLOUT_TP}" >&2
  exit 2
fi
if (( TRAIN_BATCH_SIZE % N_GPUS != 0 || PPO_MINI_BATCH_SIZE % N_GPUS != 0 )); then
  echo "[ERROR] train and mini-batch sizes must be divisible by N_GPUS=${N_GPUS}" >&2
  exit 2
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a CUDA_DEVS <<< "${CUDA_VISIBLE_DEVICES}"
  if (( ${#CUDA_DEVS[@]} != N_GPUS )); then
    echo "[ERROR] CUDA_VISIBLE_DEVICES exposes ${#CUDA_DEVS[@]} GPU(s), expected ${N_GPUS}" >&2
    exit 2
  fi
fi

AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-$((N_GPUS / ROLLOUT_TP))}"
RPCO_INIT_CKPT="${RPCO_INIT_CKPT:-}"
MODEL_PATH="${RPCO_INIT_CKPT:-${MODEL_PATH:-Qwen/Qwen3-VL-2B-Instruct}}"

HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}"
UNICOT_BREAKDOWN_DIR="${UNICOT_BREAKDOWN_DIR:-${HF_HUB_CACHE}/datasets--Fr0zencr4nE--UniCoT-Breakdown-3K}"
UNICOT_REFLECTION_DIR="${UNICOT_REFLECTION_DIR:-${HF_HUB_CACHE}/datasets--Fr0zencr4nE--UniCoT-Self-Reflection-6K}"
UNICOT_MIX_RATIO="${UNICOT_MIX_RATIO:-0.5}"
UNICOT_VAL_RATIO="${UNICOT_VAL_RATIO:-0.05}"
UNICOT_SPLIT_SEED="${UNICOT_SPLIT_SEED:-42}"
TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/outputs/data/agentic_unicot/train.parquet}"
VAL_FILE="${VAL_FILE:-${REPO_ROOT}/outputs/data/agentic_unicot/val.parquet}"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-agentic_rpco_${RUN_TS}}"
CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/checkpoints/verl_omni_agentic/${EXPERIMENT_NAME}}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-8192}"
MAX_RESP_LEN="${MAX_RESP_LEN:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LEN + MAX_RESP_LEN))}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-16}"
MAX_USER_TURNS="${MAX_USER_TURNS:-16}"

export AGENTIC_E2E_ROOT="${AGENTIC_E2E_ROOT:-${REPO_ROOT}/outputs/e2e}"
export AGENTIC_E2E_RUN_NAME="${EXPERIMENT_NAME}"
export AGENTIC_VAL_VIZ="${AGENTIC_VAL_VIZ:-1}"
export AGENTIC_VLLM_OMNI_URL="${AGENTIC_VLLM_OMNI_URL:-http://127.0.0.1:8092}"
export AGENTIC_VLLM_URL="${AGENTIC_VLLM_URL:-http://127.0.0.1:8093}"
export UNICOT_BREAKDOWN_DIR UNICOT_REFLECTION_DIR UNICOT_MIX_RATIO UNICOT_VAL_RATIO UNICOT_SPLIT_SEED

# VisionCreator-R1 defaults all active dimensions to equal weight. This
# branch's builder uses ``tool``; accept the anchor's ``tool_call`` spelling as
# a compatibility alias.
if [[ -z "${RPCO_W_TOOL:-}" && -n "${RPCO_W_TOOL_CALL:-}" ]]; then
  export RPCO_W_TOOL="${RPCO_W_TOOL_CALL}"
fi
for DIM in REFLECT PLAN FORMAT TOOL RESULT; do
  KEY="RPCO_W_${DIM}"
  if [[ -z "${!KEY:-}" ]]; then
    export "${KEY}=1.0"
  fi
done

DATA_ARGS=(
  --breakdown_dir "$UNICOT_BREAKDOWN_DIR"
  --reflection_dir "$UNICOT_REFLECTION_DIR"
  --local_save_dir "$(dirname "$TRAIN_FILE")"
  --mix_ratio "$UNICOT_MIX_RATIO"
  --val_ratio "$UNICOT_VAL_RATIO"
  --seed "$UNICOT_SPLIT_SEED"
)
if [[ -n "${UNICOT_TRAIN_SIZE:-}" ]]; then
  DATA_ARGS+=(--train_size "$UNICOT_TRAIN_SIZE")
fi
if [[ -n "${UNICOT_VAL_SIZE:-}" ]]; then
  DATA_ARGS+=(--val_size "$UNICOT_VAL_SIZE")
fi
if [[ "${REBUILD_UNICOT:-0}" == "1" || ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  python3 -m verl_omni.utils.dataset.visual_reflection.build_unicot_agentic_rl "${DATA_ARGS[@]}"
else
  echo "[INFO] reusing ${TRAIN_FILE} and ${VAL_FILE}; set REBUILD_UNICOT=1 to rebuild"
fi

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LEN" \
  data.max_response_length="$MAX_RESP_LEN" \
  data.filter_overlong_prompts=true \
  data.truncation=left \
  data.return_raw_chat=true \
  data.seed="$UNICOT_SPLIT_SEED" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=16 \
  actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.trust_remote_code=true \
  actor_rollout_ref.model.use_remove_padding=true \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-4}" \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.optim.clip_grad=1.0 \
  actor_rollout_ref.actor.ppo_epochs=2 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.actor.fsdp_config.use_orig_params=true \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-0.8}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
  actor_rollout_ref.rollout.enable_chunked_prefill=true \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.rollout.layered_summon=false \
  actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS:-16}" \
  actor_rollout_ref.rollout.max_model_len="$MAX_MODEL_LEN" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_ROLLOUT_N:-1}" \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=false \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_ASSISTANT_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="$MAX_USER_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=2048 \
  actor_rollout_ref.rollout.agent.default_agent_loop=agentic_tool_agent \
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=verl_omni.agent_loop.agentic_metrics_manager.AgenticMetricsAgentLoopManager \
  actor_rollout_ref.rollout.agent.num_workers="$AGENT_NUM_WORKERS" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.ref.fsdp_config.use_orig_params=true \
  reward.custom_reward_function.path=pkg://verl_omni.utils.reward_score.agentic_multidim_reward \
  reward.custom_reward_function.name=compute_score \
  trainer.val_before_train="${VAL_BEFORE_TRAIN:-true}" \
  trainer.test_freq="${TEST_FREQ:-10}" \
  trainer.save_freq="${SAVE_FREQ:-5}" \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="$N_GPUS" \
  trainer.total_training_steps="$TOTAL_STEPS" \
  trainer.total_epochs="$TOTAL_STEPS" \
  trainer.resume_mode="${RESUME_MODE:-disable}" \
  trainer.default_local_dir="$CKPT_DIR" \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${WANDB_PROJECT:-verl_omni_agentic}" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  "$@"
