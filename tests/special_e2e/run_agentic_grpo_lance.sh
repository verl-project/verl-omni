#!/usr/bin/env bash
# ============================================================================
# Lance-3B Agentic GRPO GPU Smoke (#329 ST-1 / AC1)
# ============================================================================
#
# ST-1 (AC1): 1-step toy training completes — no OOM, finite non-zero loss,
#             stock HF + vLLM path.
#
# Usage (from verl-omni repo root). Set MODEL_PATH to a prepared Lance_3B_hf_und
# export (see tests/special_e2e/prepare_lance_hf_und.py). Machine-local env
# (CUDA/Ray LD_LIBRARY_PATH, GPU ids, WANDB, NCCL, VERL_USE_EXTERNAL_MODULES)
# belongs in the operator shell — not this script — then:
#   bash tests/special_e2e/run_agentic_grpo_lance.sh
# Do NOT point MODEL_PATH at raw Lance_3B (no chat_template → empty dataset).
#
# Output (override with OUTPUT_DIR):
#   outputs/agentic_grpo_lance_smoke/agentic_grpo_onestep.log
# ============================================================================
set -euo pipefail

# ---- Config -----------------------------------------------------------------
# Lance MoT HF layout (Lance_3B) is incomplete: no config.json / chat_template.
# Smoke uses the prepared understanding-only export (prepare_lance_hf_und.py).
# Do NOT point MODEL_PATH at raw Lance_3B — tokenizer has no chat_template and
# every prompt is skipped → filter dataset len: 0.
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a prepared HF understanding export (see prepare_lance_hf_und.py)}"

DATA_DIR="${DATA_DIR:-$HOME/data/agentic}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/agentic_grpo_lance_smoke}"
# Merge gate always uses its own toy parquet unless ST1_USE_ENV_DATA=1
# (operator env often exports overfit TRAIN_FILE/VAL_FILE).
if [[ "${ST1_USE_ENV_DATA:-0}" == "1" ]]; then
  TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
  VAL_FILE="${VAL_FILE:-$DATA_DIR/val.parquet}"
else
  TRAIN_FILE="$DATA_DIR/train.parquet"
  VAL_FILE="$DATA_DIR/val.parquet"
fi
# Prefer CUDA_VISIBLE_DEVICES count when the operator set it; else portable default.
if [[ -z "${N_GPUS:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a _cuda_devs <<< "${CUDA_VISIBLE_DEVICES// /}"
    N_GPUS="${#_cuda_devs[@]}"
  else
    N_GPUS=2
  fi
fi

# Always use the active venv interpreter when present (bare `python3` may be
# miniconda/base and would install TransferQueue into the wrong env).
PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Optional diagnostics only (not passed into HFModelConfig — it has no
# architecture/freeze fields; those belonged to the removed Omni agentic path).
if [[ -z "${MODEL_ARCHITECTURE:-}" && -f "$MODEL_PATH/config.json" ]]; then
  MODEL_ARCHITECTURE="$("$PYTHON_BIN" -c "import json; print(json.load(open('$MODEL_PATH/config.json'))['architectures'][0])")"
fi
MODEL_ARCHITECTURE="${MODEL_ARCHITECTURE:-Qwen2ForCausalLM}"

mkdir -p "$OUTPUT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
_pass()  { echo -e "${GREEN}[PASS]${NC} $*"; }
_fail() { echo -e "${RED}[FAIL]${NC} $*"; }
_info() { echo -e "${YELLOW}[INFO]${NC} $*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Hermes Jinja lives next to the recipe (upstream-style packaging). Prefer a
# prepared und export that already baked it via prepare_lance_hf_und.py — same
# as using an Instruct tokenizer. ToolAgentLoop parsing uses format=hermes.
TOOL_CHAT_TEMPLATE="${TOOL_CHAT_TEMPLATE:-$SCRIPT_DIR/qwen2_tool_chat_template.jinja2}"
if [[ ! -f "$TOOL_CHAT_TEMPLATE" ]]; then
  _fail "missing Hermes tool chat template: $TOOL_CHAT_TEMPLATE"
  exit 2
fi

# Colocated FSDP+vLLM: util is a fraction of *total* VRAM after actor init.
# 0.12 is often too low for KV once FSDP is resident; 0.15 works with
# max_model_len=4096 on free GPUs. Override via GPU_MEM_UTIL if needed.
_GPU_MEM_UTIL_WAS_SET=0
if [[ -n "${GPU_MEM_UTIL+x}" ]]; then
  _GPU_MEM_UTIL_WAS_SET=1
fi
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.15}"
if command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t _gpu_free_mib < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a _vis <<< "${CUDA_VISIBLE_DEVICES// /}"
  else
    _vis=("${!_gpu_free_mib[@]}")
  fi
  for _i in "${_vis[@]}"; do
    _free_gib="$(awk -v mib="${_gpu_free_mib[_i]:-0}" 'BEGIN { printf "%.1f", mib/1024 }')"
    if awk -v f="$_free_gib" 'BEGIN { exit !(f+0 < 24) }'; then
      _fail "ST-1 needs >=24GiB free on GPU$_i (free=${_free_gib}GiB); pick free CUDA_VISIBLE_DEVICES"
      exit 2
    fi
  done
fi

# Mode (2a) + toy sizing inlined for the 1-step GPU smoke.
# Multi-step Lance e2e / overfit recipes are out of scope here.
SMOKE_OVERRIDES=(
  algorithm.adv_estimator=grpo

  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.load_format=safetensors

  +actor_rollout_ref.model.override_config.tie_word_embeddings=false
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa

  actor_rollout_ref.rollout.multi_turn.enable=true
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5
  actor_rollout_ref.rollout.multi_turn.max_user_turns=5
  # Upstream default; set explicitly so ToolAgentLoop uses HermesToolParser.
  actor_rollout_ref.rollout.multi_turn.format=hermes
  actor_rollout_ref.rollout.multi_turn.function_tool_path=verl_omni/agent_loop/diffusion_tool.py

  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
  actor_rollout_ref.rollout.agent.agent_loop_config_path=null
  # train_batch_size must divide agent.num_workers (DataProto.chunk).
  actor_rollout_ref.rollout.agent.num_workers=2

  reward.custom_reward_function.path=pkg://verl_omni.utils.reward_score.agentic_reward
  # Deterministic length heuristic gives cold und reward variance for GRPO.
  reward.custom_reward_function.name=compute_score

  data.train_batch_size=4
  data.max_prompt_length=1024
  data.max_response_length=1024
  data.filter_overlong_prompts=true
  data.truncation=left
  # Cap below HF config max (128k) so colocated util~0.15 has enough KV.
  actor_rollout_ref.rollout.max_model_len=4096

  actor_rollout_ref.model.lora_rank=64
  actor_rollout_ref.model.lora_alpha=32
  "actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]"
  actor_rollout_ref.model.enable_gradient_checkpointing=true

  actor_rollout_ref.actor.ppo_mini_batch_size=4
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.fsdp_config.param_offload=true
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
  actor_rollout_ref.actor.fsdp_config.use_orig_params=true

  actor_rollout_ref.rollout.n=2
  actor_rollout_ref.rollout.temperature=0.8
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL}"
  actor_rollout_ref.rollout.enable_chunked_prefill=true
  actor_rollout_ref.rollout.enforce_eager=true
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2

  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
  actor_rollout_ref.ref.fsdp_config.param_offload=true
  actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
  actor_rollout_ref.ref.fsdp_config.use_orig_params=true

  trainer.val_before_train=false
  trainer.nnodes=1
)
if [[ "$N_GPUS" -eq 1 ]]; then
  # Single-GPU hybrid disables FSDP offload → less free VRAM for vLLM KV.
  # Keep layered_summon enabled: LoRA FSDP save asserts on embed_tokens when it
  # is forced off (seen on 1-GPU ST-1). Prefer CUDA_VISIBLE_DEVICES with 2 GPUs.
  if [[ "$_GPU_MEM_UTIL_WAS_SET" -eq 0 ]]; then
    GPU_MEM_UTIL=0.20
    for _i in "${!SMOKE_OVERRIDES[@]}"; do
      if [[ "${SMOKE_OVERRIDES[$_i]}" == actor_rollout_ref.rollout.gpu_memory_utilization=* ]]; then
        SMOKE_OVERRIDES[$_i]="actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL}"
      fi
    done
  fi
  SMOKE_OVERRIDES+=(
    actor_rollout_ref.actor.fsdp_config.param_offload=false
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
    actor_rollout_ref.ref.fsdp_config.param_offload=false
  )
  _info "N_GPUS=1: offload disabled; gpu_memory_utilization=${GPU_MEM_UTIL} (prefer 2 GPUs)"
fi

# ---- Pre-flight -------------------------------------------------------------
# pr-fredfork omni V1 path imports TransferQueue at package load (verl ppo.v1).
if ! "$PYTHON_BIN" -c "import transfer_queue" >/dev/null 2>&1; then
  _info "Installing TransferQueue into $($PYTHON_BIN -c 'import sys; print(sys.executable)') ..."
  "$PYTHON_BIN" -m pip install 'TransferQueue==0.1.8'
fi

_info "=== Agentic GRPO GPU Smoke (ST-1) ==="
_info "Python: $($PYTHON_BIN -c 'import sys; print(sys.executable)')"
_info "Model:  $MODEL_PATH"
_info "Data:   $DATA_DIR"
_info "GPUs:   $N_GPUS  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
_info "vLLM gpu_memory_utilization=${GPU_MEM_UTIL}"
_info "Output: $OUTPUT_DIR"

# Generate / refresh toy agentic data if missing or still on the pre-Hermes schema.
NEED_DATA=0
if [[ ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  NEED_DATA=1
elif ! "$PYTHON_BIN" -c "
import sys
import pandas as pd
df = pd.read_parquet(sys.argv[1])
prompt = df.iloc[0]['prompt']
ok = isinstance(prompt, list) and len(prompt) >= 4 and any(
    isinstance(m, dict) and m.get('role') == 'assistant' and '<tool_call>' in str(m.get('content', ''))
    for m in prompt
)
raise SystemExit(0 if ok else 1)
" "$TRAIN_FILE"; then
  NEED_DATA=1
fi
if [[ "$NEED_DATA" -eq 1 ]]; then
  _info "Generating Hermes-format toy agentic parquet data ..."
  "$PYTHON_BIN" tests/special_e2e/create_dummy_agentic_data.py \
    --local_save_dir "$DATA_DIR" --train_size 8 --val_size 4
fi

ST1_FAIL=0
ST1_LOG="$OUTPUT_DIR/agentic_grpo_onestep.log"
ST1_CKPT="$OUTPUT_DIR/agentic_grpo_onestep_ckpt"

# Record a failed assertion without putting FAIL= on the same line as _fail
# (avoids brittle `; VAR=1` parsing after echo -e / ANSI).
_record_fail() {
  _fail "$1"
  ST1_FAIL=1
}

# ============================================================================
# ST-1: 1-step toy training completes (AC1)
# ============================================================================
_info ""
_info "=== ST-1: 1-Step Agentic GRPO Training ==="
_info "Log:   $ST1_LOG"
_info "Arch:  $MODEL_ARCHITECTURE  rollout=vllm"

# Fresh smoke: never resume a prior ST-1 ckpt (LoRA/FSDP keys can mismatch across runs).
rm -rf "$ST1_CKPT"
mkdir -p "$ST1_CKPT"

set +e  # do not exit on failure so we can report
"$PYTHON_BIN" -m verl.trainer.main_ppo \
    'hydra.run.dir='"$OUTPUT_DIR" \
    'hydra.sweep.dir='"$OUTPUT_DIR" \
    hydra.output_subdir=null \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    "${SMOKE_OVERRIDES[@]}" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.resume_mode=disable \
    trainer.default_local_dir="$ST1_CKPT" \
    trainer.logger=console \
    "$@" 2>&1 | tee "$ST1_LOG"
ST1_EXIT=$?
set -e

# --- ST-1 assertions ---
if [[ "$ST1_EXIT" -ne 0 ]]; then
  _record_fail "ST-1: Training exited with code $ST1_EXIT"
fi
# Real CUDA/torch OOM only — do not match FSDP warnings like "risks CPU OOM".
if grep -Eiq 'cuda\s*out\s*of\s*memory|torch\.OutOfMemoryError|OutOfMemoryError:\s*CUDA' "$ST1_LOG"; then
  _record_fail "ST-1: OOM detected"
fi
if ! grep -q "actor/loss" "$ST1_LOG"; then
  _record_fail "ST-1: No 'actor/loss' metric in log"
fi
# Guard: recipe must stay on stock HF + vLLM (no removed custom agentic/Omni path).
if grep -Eq "AgenticLLMFSDPEngine|model_type.?.?agentic_llm|vllm_omni_model" "$ST1_LOG"; then
  _record_fail "ST-1: Unexpected custom worker/model path detected"
else
  _pass "ST-1: Stock language-model worker/rollout path used"
fi
# Non-zero finite loss ⇒ actor gradients flowed (AC: agent LLM weights update).
LOSS_CHECK_EXIT=0
"$PYTHON_BIN" - "$ST1_LOG" <<'PY' || LOSS_CHECK_EXIT=$?
import math
import re
import sys

loss_re = re.compile(
    r"actor/loss(?:[:\s=]+|:)(?:np\.(?:float64|float32)\()?([-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?|nan|[-]?inf)\)?",
    re.IGNORECASE,
)
with open(sys.argv[1], encoding="utf-8", errors="replace") as f:
    for line in f:
        m = loss_re.search(line)
        if not m:
            continue
        raw = m.group(1)
        try:
            val = float(raw)
        except ValueError:
            continue
        if math.isfinite(val) and val != 0.0:
            sys.exit(0)
        if not math.isfinite(val):
            sys.exit(2)
sys.exit(1)
PY
if [[ "$LOSS_CHECK_EXIT" -eq 2 ]]; then
  _record_fail "ST-1: NaN/Inf in actor/loss"
elif [[ "$LOSS_CHECK_EXIT" -ne 0 ]]; then
  _record_fail "ST-1: Could not extract non-zero actor/loss"
else
  _pass "ST-1: Non-zero actor/loss confirms gradients flowed"
fi
# Multi-turn evidence on the toy agentic dataset (AC: rewrites across iterations).
MULTITURN_EXIT=0
python3 - "$ST1_LOG" <<'PY' || MULTITURN_EXIT=$?
import re, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
# Match num_turns/... metrics with value >= 2 (plain float or numpy wrapper).
for m in re.finditer(
    r"num_turns[^0-9\n]{0,48}?(?:np\.\w+\()?([2-9]\d*(?:\.\d+)?)",
    text,
):
    sys.exit(0)
sys.exit(1)
PY
if [[ "$MULTITURN_EXIT" -ne 0 ]]; then
  _record_fail "ST-1: No multi-turn evidence (num_turns >= 2) in log"
else
  _pass "ST-1: Multi-turn num_turns >= 2 observed"
fi

if [[ "$ST1_FAIL" -eq 0 ]]; then
  _pass "ST-1: PASSED"
else
  _fail "ST-1: FAILED - ${ST1_FAIL} assertion group(s) failed"
fi

echo ""
echo "================================================================================"
echo "   Merge Gate: GPU Smoke Test Results"
echo "================================================================================"
printf "ST-1 (AC1: 1-step training):     %s\n" "$([ "$ST1_FAIL" -eq 0 ] && echo '✅ PASS' || echo '❌ FAIL')"
echo "--------------------------------------------------------------------------------"
echo "  Log: $ST1_LOG"
echo "================================================================================"

if [ "$ST1_FAIL" -eq 0 ]; then
  echo ""
  echo "  ✅ GPU SMOKE (ST-1): PASSED"
  echo ""
  exit 0
else
  echo ""
  echo "  ❌ GPU SMOKE (ST-1): FAILED"
  echo ""
  exit 1
fi
