# Bagel UND+GEN Co-RL (joint training) recipe, PR1 correctness.
# Entry: ${_PY} -m verl_omni.trainer.main_omni
#
# Fail-closed: UND must be the published Bagel checkpoint (Hermes tool-call), never
# Qwen3-VL; the UniCoT parquet must carry extra_info.reference_image_path; vllm-omni >=
# 0.24 requires model.lora.merge=True (fused MoT gen experts cannot bind *_moe_gen
# adapters). Do not edit the Mode (2a) recipe run_agenticrpco_grpo_lora.sh.
# Shell tracing is opt-in: DEBUG=1 adds the "+ ..." lines.
if [[ "${DEBUG:-}" == "1" ]]; then
  set -x
fi
_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../../.." && pwd)"
# Same interpreter convention as run_bagel_und_ar_serve.sh: prefer this repo's venv, fall
# back to PATH. Resolved *before* PYTHONPATH because the sibling-verl probe below needs it.
_PY="${_REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${_PY}" ]]; then
  _PY="$(command -v python3)"
fi
# Prefer the sibling ``verl`` checkout (its tokenizer package layout predates the venv's
# flat ``tokenizer.py``), but only when that checkout actually satisfies this tree's
# imports. A *stale* sibling is worse than none: it takes PYTHONPATH precedence and
# shadows the venv's installed ``verl``, so the run dies at import before any GPU is
# touched -- measured 2026-10-12 on this box, whose ``../verl`` predates
# ``build_rollout_dist_profiler`` being re-exported from ``verl.utils.profiler``:
#   ImportError: cannot import name 'build_rollout_dist_profiler' from 'verl.utils.profiler'
# The probe reproduces exactly that shadow (candidate first, venv site-packages behind it),
# so a compatible sibling keeps working and an incompatible one degrades to the venv copy.
_LOCAL_VERL="${_LOCAL_VERL:-${_REPO_ROOT}/../verl}"
if [[ -d "${_LOCAL_VERL}/verl" ]] \
  && PYTHONPATH="${_LOCAL_VERL}" "${_PY}" -c "from verl.utils.profiler import build_rollout_dist_profiler" >/dev/null 2>&1; then
  export PYTHONPATH="${_LOCAL_VERL}:${_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
else
  echo "[INFO] sibling verl checkout ${_LOCAL_VERL} is absent or too old for this tree; using the venv's installed verl" >&2
  export PYTHONPATH="${_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

_HF_HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"

TOTAL_STEPS="${TOTAL_STEPS:-200}"

# Flattened, self-describing Hydra run dir (RUN_TS is computed once; override it to
# re-launch into an existing folder). Checkpoints stay in the shared
# checkpoints/<project_name>/<experiment_name> unless DEFAULT_LOCAL_DIR is set.
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-bagel_corl_${RUN_TS}}"
RUN_DIR="${RUN_DIR:-${_REPO_ROOT}/outputs/${RUN_NAME}}"
DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-}"
echo "[INFO] bagel_corl run dir: ${RUN_DIR} (RUN_TS=${RUN_TS})"

# UND AR sampling: the published Bagel checkpoint degenerates under vLLM defaults
# (temp 1.0, top_p 1, top_k -1, no repetition penalty), so pin anti-repetition sampling.
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.9}"
ROLLOUT_TOP_K="${ROLLOUT_TOP_K:-50}"
ROLLOUT_REPETITION_PENALTY="${ROLLOUT_REPETITION_PENALTY:-1.05}"

export BAGEL_MODEL_PATH="${BAGEL_MODEL_PATH:-$HOME/models/ByteDance-Seed/BAGEL-7B-MoT}"
WORKSPACE=${WORKSPACE:-$HOME}
# Default to the same cards as run_bagel_und_ar_serve.sh (0,1,2,3) and never
# hard-assign: an unconditional export would clobber a caller's job allocation.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# flashinfer JIT-compiles MM/attention kernels and needs nvcc. The CUDA toolkits on this
# box live under /cm/shared, not the /usr/local/cuda default, so without this the engine
# dies with:
#   RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
# It also has to be a *recent* nvcc, which is the part a first-found-wins loop gets wrong.
# flashinfer 0.6.18 builds its kernels with ``--compress-mode=size`` (an nvcc 12.8+ flag),
# and pointing that build at 12.4 aborts the reward engine's init (measured 2026-10-14,
# cold flashinfer cache under a vllm 0.30.0 / flashinfer 0.6.18.post1 venv):
#   nvcc fatal   : Unknown option '--compress-mode=size'
#   ninja: build stopped: subcommand failed.
#   RuntimeError: Worker failed with error 'Ninja build failed. ...'
# So do not stop at the first ``toolkit/current``: ``/cm/shared/apps/cuda-latest`` is stale
# (its ``current`` symlink still points at 12.4.1), while 13.0 sits at
# ``cuda13.0/toolkit/13.0`` with no ``current`` symlink for a probe to find -- and 13.0 is
# the toolkit this venv's torch is built for (``torch 2.13.0+cu130``). Enumerate every
# installed toolkit and keep the newest nvcc that accepts the flag flashinfer passes, i.e.
# the property that actually decides whether the JIT build can succeed.
_nvcc_release() {
  "$1" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1
}
_nvcc_accepts_flashinfer_flags() {
  "$1" --dryrun --compress-mode=size -c /dev/null -o "${TMPDIR:-/tmp}/.nvcc-compress-probe.o" 2>&1 \
    | grep -q 'Unknown option' && return 1
  return 0
}
if [[ -z "${CUDA_HOME:-}" ]]; then
  while read -r _nvcc_ver _cand; do
    if [[ -n "${_cand}" ]] && _nvcc_accepts_flashinfer_flags "${_cand}/bin/nvcc"; then
      export CUDA_HOME="${_cand}"
      break
    fi
  done < <(
    ls -d /cm/shared/apps/cuda*/toolkit/*/ /usr/local/cuda 2>/dev/null \
      | while read -r _d; do
          [[ -x "${_d%/}/bin/nvcc" ]] && printf '%s %s\n' "$(_nvcc_release "${_d%/}/bin/nvcc")" "${_d%/}"
        done \
      | sort -rV
  )
fi
if [[ -n "${CUDA_HOME:-}" && ":${PATH}:" != *":${CUDA_HOME}/bin:"* ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi

# Same JIT-toolchain theme as nvcc above, but for ``ninja``: put the venv ``_PY`` was
# resolved from on PATH.
#
# ``vllm``'s engine warm-up shells out to ``ninja`` to compile kernels:
#   vllm/v1/engine/core.py:356 _initialize_kv_caches
#     -> self.model_executor.compile_or_warm_up_model()
#   FileNotFoundError: [Errno 2] No such file or directory: 'ninja'
#   RuntimeError: Worker failed with error '[Errno 2] No such file or directory: 'ninja''
# which surfaces as "Engine core initialization failed" and costs the entire startup
# (measured 2026-10-14: ~6 min, at the reward engine, REWARD_TP=4). The venv ships
# ``.venv/bin/ninja``, and nothing in this recipe ever added that directory to PATH --
# the judge sidecar below sets PATH for itself only. Interactive launches had it by
# accident (an activated venv), so the hole only opens for a launch from a bare shell
# (a watchdog, ``nohup``/``setsid``, CI, or a login shell without ``activate``).
if [[ -x "${_REPO_ROOT}/.venv/bin/ninja" && ":${PATH}:" != *":${_REPO_ROOT}/.venv/bin:"* ]]; then
  export PATH="${_REPO_ROOT}/.venv/bin:${PATH}"
fi

# The driver tops out at CUDA 12.2 while the venv's torch is cu13: prepend the cuda-compat
# shim + cu13 runtime when present, and let torch answer is_available() from NVML.
_cuda_compat="${_REPO_ROOT}/.venv/cuda-compat"
_cu13_lib="${_REPO_ROOT}/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
if [[ -d "${_cuda_compat}" ]]; then
  export LD_LIBRARY_PATH="${_cuda_compat}:${_cu13_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export PYTORCH_NVML_BASED_CUDA_CHECK="${PYTORCH_NVML_BASED_CUDA_CHECK:-1}"
fi

model_name=${BAGEL_MODEL_PATH:-$HOME/models/ByteDance-Seed/BAGEL-7B-MoT}
reward_model_name=${REWARD_MODEL:-${_HF_HUB}/models--CodeGoat24--UnifiedReward-2.0-qwen3vl-2b/snapshots/64b2f51d68a2fa583b194784b6cce3aeb12ed7df}

# These MUST keep their :- defaults: without them the parquet build aborts on empty
# --mix_ratio/--reflect_ratio/--seed from a clean shell.
UNICOT_REFLECTION_DIR=${UNICOT_REFLECTION_DIR:-${_HF_HUB}/datasets--Fr0zencr4nE--UniCoT-Self-Reflection-6K}
UNICOT_BREAKDOWN_DIR=${UNICOT_BREAKDOWN_DIR:-${_HF_HUB}/datasets--Fr0zencr4nE--UniCoT-Breakdown-3K}
UNICOT_MIX_RATIO=${UNICOT_MIX_RATIO:-0.5}
UNICOT_VAL_RATIO=${UNICOT_VAL_RATIO:-0.05}
UNICOT_SPLIT_SEED=${UNICOT_SPLIT_SEED:-42}
UNICOT_TRAIN_SIZE=${UNICOT_TRAIN_SIZE:-}
UNICOT_VAL_SIZE=${UNICOT_VAL_SIZE:-}

TRAIN_FILE=${UNICOT_TRAIN:-"${_REPO_ROOT}/outputs/data/agentic_unicot/train.parquet"}
VAL_FILE=${UNICOT_TEST:-"${_REPO_ROOT}/outputs/data/agentic_unicot/val.parquet"}
# Build the mixed UniCoT train/val parquet unless both exist (REBUILD_UNICOT=1 forces a rebuild).
if [[ "${REBUILD_UNICOT:-1}" == "1" || ! -f "$TRAIN_FILE" || ! -f "$VAL_FILE" ]]; then
  "${_PY}" -m verl_omni.utils.dataset.visual_reflection.build_unicot_agentic_rl \
      --breakdown_dir "$UNICOT_BREAKDOWN_DIR" \
      --reflection_dir "$UNICOT_REFLECTION_DIR" \
      --local_save_dir "$(dirname "$TRAIN_FILE")" \
      --mix_ratio "$UNICOT_MIX_RATIO" \
      --val_ratio "$UNICOT_VAL_RATIO" \
      --seed "$UNICOT_SPLIT_SEED" \
      ${UNICOT_TRAIN_SIZE:+--train_size "$UNICOT_TRAIN_SIZE"} \
      ${UNICOT_VAL_SIZE:+--val_size "$UNICOT_VAL_SIZE"} || {
    echo "[ERROR] UniCoT parquet build failed (traceback above). Refusing to launch:" >&2
    echo "[ERROR] without 'set -e' the trainer would otherwise keep going and train on a" >&2
    echo "[ERROR] stale ${TRAIN_FILE} (or none at all) while silently ignoring any dataset change." >&2
    exit 1
  }
else
  echo "[INFO] reusing existing UniCoT parquet: $TRAIN_FILE / $VAL_FILE (set REBUILD_UNICOT=1 to rebuild)"
fi

# All visible cards join the actor+GEN pool; UND AR is colocated on the same PG.
_count_visible_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local n=0 id
    IFS=',' read -ra _ids <<< "${CUDA_VISIBLE_DEVICES}"
    for id in "${_ids[@]}"; do
      id="${id#"${id%%[![:space:]]*}"}"
      id="${id%"${id##*[![:space:]]}"}"
      [[ -n "$id" ]] && n=$((n + 1))
    done
    echo "$n"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l
    return
  fi
  echo 1
}
N_VISIBLE="$(_count_visible_gpus)"
N_VISIBLE="${N_VISIBLE//[[:space:]]/}"
if [[ -z "$N_VISIBLE" || "$N_VISIBLE" -lt 1 ]]; then
  echo "Need at least 1 visible GPU (got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})." >&2
  exit 1
fi
NUM_GPUS_ACTOR_ROLLOUT_REWARD=${NUM_GPUS_ACTOR_ROLLOUT_REWARD:-$N_VISIBLE}
# UND AR takes the first UND_N_GPUS slices of the actor pool and must span TWO cards:
# bagel_corl_deploy_ar.yaml puts Thinker on logical 0 and DiT on logical 1. It must also
# divide the pool (split_resource_pool asserts an exact split). Raise UND_N_GPUS only
# together with the stage devices in that yaml; the mismatch is caught at launch.
UND_N_GPUS=${UND_N_GPUS:-2}
# UND AR Thinker KV budget; below ~0.393 vLLM aborts init with "No available memory for the cache blocks". Cannot buy GEN headroom.
UND_GPU_MEM_UTIL=${UND_GPU_MEM_UTIL:-0.40}
BAGEL_UND_DEPLOY_CONFIG=${BAGEL_UND_DEPLOY_CONFIG:-"$(dirname "$0")/bagel_corl_deploy_ar.yaml"}
# Fail-closed: reject a stale UND_N_GPUS<2 / UND_GPU_MEM_UTIL<0.40 from an earlier shell (bypass with UND_AR_ALLOW_TIGHT=1).
if [[ "${UND_AR_ALLOW_TIGHT:-0}" != "1" && "${NUM_GPUS_ACTOR_ROLLOUT_REWARD}" -ge 2 ]]; then
  if (( UND_N_GPUS < 2 )); then
    echo "UND_N_GPUS=${UND_N_GPUS} cannot cover bagel_corl_deploy_ar.yaml (Thinker on logical \"0\", DiT on \"1\")." >&2
    echo "Use UND_N_GPUS=2, or put both stages on \"0\" in the yaml for a 1-GPU smoke test." >&2
    exit 1
  fi
  if awk "BEGIN {exit !(${UND_GPU_MEM_UTIL} < 0.40)}"; then
    echo "UND_GPU_MEM_UTIL=${UND_GPU_MEM_UTIL} is below the ~0.393 AR engine-init floor" >&2
    echo "(28.16GiB consumed + 2.96GiB peak over a 27.54GiB checkpoint on a 79.66GiB card)." >&2
    echo "0.30 and 0.35 abort _init_und with 'No available memory for the cache blocks'." >&2
    exit 1
  fi
fi
# Optional live Hermes proof: STOP run_bagel_und_ar_serve.sh first — its AR engine pins GPU 0, which the colocated UND AR needs.
BAGEL_UND_AR_SERVING_READY=${BAGEL_UND_AR_SERVING_READY:-1}
echo "bagel_corl 4-device e2e: actor+GEN=${NUM_GPUS_ACTOR_ROLLOUT_REWARD} UND_AR_colocated=${UND_N_GPUS} visible=${N_VISIBLE} und_ready=${BAGEL_UND_AR_SERVING_READY} und_util=${UND_GPU_MEM_UTIL}"
# ROLLOUT_TP and the GEN deploy yaml's per-replica device template move together: the
# stage's devices list length must equal its world_size ("0,1" -> replica 0: 0,1; replica 1: 2,3).
if [[ "${NUM_GPUS_ACTOR_ROLLOUT_REWARD}" -ge 4 ]]; then
  ROLLOUT_TP=${ROLLOUT_TP:-2}
  _GEN_DEPLOY_DEFAULT="$(dirname "$0")/bagel_corl_deploy_tp2.yaml"
else
  ROLLOUT_TP=${ROLLOUT_TP:-1}
  _GEN_DEPLOY_DEFAULT="$(dirname "$0")/bagel_corl_deploy.yaml"
fi
BAGEL_DEPLOY_CONFIG=${BAGEL_DEPLOY_CONFIG:-$_GEN_DEPLOY_DEFAULT}
# One colocated RM tensor-parallel across the pool (not N copies of Qwen). Override REWARD_TP=1 for per-GPU workers.
REWARD_TP=${REWARD_TP:-$NUM_GPUS_ACTOR_ROLLOUT_REWARD}
REWARD_ENGINE=${REWARD_ENGINE:-vllm}
if (( NUM_GPUS_ACTOR_ROLLOUT_REWARD % ROLLOUT_TP != 0 )); then
  echo "n_gpus_per_node=${NUM_GPUS_ACTOR_ROLLOUT_REWARD} must be divisible by ROLLOUT_TP=${ROLLOUT_TP}." >&2
  exit 1
fi
if (( NUM_GPUS_ACTOR_ROLLOUT_REWARD % REWARD_TP != 0 )); then
  echo "n_gpus_per_node=${NUM_GPUS_ACTOR_ROLLOUT_REWARD} must be divisible by REWARD_TP=${REWARD_TP}." >&2
  exit 1
fi
# split_resource_pool asserts an exact split, so UND_N_GPUS must divide the actor pool.
if (( NUM_GPUS_ACTOR_ROLLOUT_REWARD % UND_N_GPUS != 0 )); then
  echo "n_gpus_per_node=${NUM_GPUS_ACTOR_ROLLOUT_REWARD} must be divisible by UND_N_GPUS=${UND_N_GPUS}" >&2
  echo "(the UND AR slice comes from split_resource_pool, which requires an exact split)." >&2
  exit 1
fi
# The GEN deploy yaml's stage must name exactly ROLLOUT_TP devices; read it back, do not trust the pairing above.
"${_PY}" - "$BAGEL_DEPLOY_CONFIG" "$ROLLOUT_TP" <<'PY' || exit 1
import sys

import yaml

path, tp = sys.argv[1], int(sys.argv[2])
try:
    stages = yaml.safe_load(open(path)).get("stages") or []
except OSError as exc:
    sys.exit(f"GEN deploy config {path!r} is unreadable: {exc}")
if not stages:
    sys.exit(f"GEN deploy config {path!r} declares no stages")
for stage in stages:
    devices = [d for d in str(stage.get("devices", "")).split(",") if d.strip()]
    if len(devices) != tp:
        sys.exit(
            f"GEN deploy config {path!r}: stage {stage.get('stage_id')} names {len(devices)} "
            f"device(s) {devices} but ROLLOUT_TP={tp}. A diffusion stage's device list is a "
            f"per-replica template whose length must equal tensor_parallel_size; pass a "
            f"matching yaml (bagel_corl_deploy.yaml for TP=1, bagel_corl_deploy_tp2.yaml for "
            f"TP=2) or clear ROLLOUT_TP."
        )
PY
# RM default: off for 1-2 GPUs (actor+Omni+UND cannot fit), on for >=4 so GEN gets an rm_score; ENABLE_RM=0 forces it off.
#
# ``GEN_N_DEFAULT`` is the sibling count (``rollout.n``) for this GPU budget. N is what
# makes a GRPO advantage *estimable*: the baseline is the group mean over N siblings of the
# same prompt, so a larger N buys a less noisy baseline on the same task, while
# ``TRAIN_BSZ`` buys a more diverse task mix. Both are raised on the >=4-GPU branch, which
# is the one whose batch was measured to have no trend.
GEN_N_DEFAULT=2
if [[ "$NUM_GPUS_ACTOR_ROLLOUT_REWARD" -eq 1 ]]; then
  ENABLE_RM=${ENABLE_RM:-0}
  REWARD_GPU_MEM_UTIL=${REWARD_GPU_MEM_UTIL:-0.08}
  ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.18}
  TRAIN_BSZ=${TRAIN_BSZ:-1}
  MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-512}
  GEN_STEPS=${GEN_STEPS:-4}
  GEN_HW=${GEN_HW:-256}
  CFG_TEXT_SCALE=${CFG_TEXT_SCALE:-1.0}
  # Canonical image guidance; inert while CFG_TEXT_SCALE <= 1.0, pinned anyway to keep both sides equal.
  CFG_IMG_SCALE=${CFG_IMG_SCALE:-1.5}
  LORA_RANK=${LORA_RANK:-4}
  LORA_ALPHA=${LORA_ALPHA:-8}
elif [[ "$NUM_GPUS_ACTOR_ROLLOUT_REWARD" -le 2 ]]; then
  ENABLE_RM=${ENABLE_RM:-0}
  REWARD_GPU_MEM_UTIL=${REWARD_GPU_MEM_UTIL:-0.12}
  ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.30}
  TRAIN_BSZ=${TRAIN_BSZ:-2}
  MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-1024}
  # 15 mirrors the canonical BAGEL FlowGRPO recipes (rollout 15, val 50).
  GEN_STEPS=${GEN_STEPS:-15}
  GEN_HW=${GEN_HW:-512}
  CFG_TEXT_SCALE=${CFG_TEXT_SCALE:-1.0}
  CFG_IMG_SCALE=${CFG_IMG_SCALE:-1.5}
  LORA_RANK=${LORA_RANK:-8}
  LORA_ALPHA=${LORA_ALPHA:-16}
else
  # >=4 GPUs: RM on by default — see the branch comment above. GEN cannot train without it.
  ENABLE_RM=${ENABLE_RM:-1}
  REWARD_GPU_MEM_UTIL=${REWARD_GPU_MEM_UTIL:-0.15}
  # This bounds only the GEN lane's KV/workspace, not its checkpoint footprint; keep 0.25.
  ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.25}
  # 4 prompt groups x N=4 sibling episodes = 16 episodes per optimizer step, and PPO_MINI=4
  # keeps them in ONE mini-batch (4 episodes per rank at PPO_MICRO=1). The reference run at
  # 2 x N=4 (8 episodes/step) measured a training reward with no trend under this recipe's
  # noise; every extra sibling episode is another sample of the same task distribution,
  # whereas a smaller batch only re-rolls the same dice. Memory is not the constraint at
  # this batch size: ppo_micro_batch_size_per_gpu=1 plus actor param/optimizer offload means
  # the activations are per-episode, so doubling TRAIN_BSZ doubles the *step time*, not the
  # peak (modulo the bounds the rollout/RM engines already carry via their mem-util knobs).
  TRAIN_BSZ=${TRAIN_BSZ:-4}
  GEN_N_DEFAULT=4
  MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-2048}
  # pr634 UniGRPO SDE kernel: noise 0.8, SDE on the first ~20% of the schedule (window fixed at 0).
  GEN_STEPS=${GEN_STEPS:-10}
  NOISE_LEVEL=${NOISE_LEVEL:-0.8}
  SDE_WINDOW_SIZE=${SDE_WINDOW_SIZE:-2}
  SDE_WINDOW_RANGE=${SDE_WINDOW_RANGE:-[0,2]}
  GEN_HW=${GEN_HW:-512}
  # Canonical BAGEL text CFG (4.0), the fix for washed GEN images; costs one extra forward
  # per enabled scale. Set 1.0 to restore cheaper CFG-free rollouts.
  CFG_TEXT_SCALE=${CFG_TEXT_SCALE:-4.0}
  # MUST be pinned: rollout defaults 1.5 vs training's model.pipeline.cfg_img_scale 1.0; the
  # trainer copies this dict onto model.pipeline, so setting it here fixes both sides.
  CFG_IMG_SCALE=${CFG_IMG_SCALE:-1.5}
  LORA_RANK=${LORA_RANK:-8}
  LORA_ALPHA=${LORA_ALPHA:-16}
fi
if [[ "$ENABLE_RM" != "1" ]]; then
  ENABLE_RM=0
fi

# Reward-pool boot knobs, kept off the colocated actor: pin max_model_len to the episode
# window (not the RM's 262144) and kv_cache_memory_bytes so vLLM skips its memory-profiling
# snapshot comparison (a deterministic boot, decoupled from REWARD_GPU_MEM_UTIL).
REWARD_MAX_MODEL_LEN=${REWARD_MAX_MODEL_LEN:-8192}
REWARD_KV_CACHE_BYTES=${REWARD_KV_CACHE_BYTES:-2147483648}  # pinned KV per RM GPU
# Sidecar is an opt-in fallback when the RM lane is off; the router shadows it when ENABLE_RM=1.
JUDGE_SERVER=${JUDGE_SERVER:-0}  # external judge is opt-in; the RM pool scores by default
JUDGE_URL=${JUDGE_URL:-http://127.0.0.1:8093}
JUDGE_VLLM_URL=""
[[ "$JUDGE_SERVER" == "1" ]] && JUDGE_VLLM_URL="$JUDGE_URL"
JUDGE_GOOD_ENOUGH_THRESHOLD=${JUDGE_GOOD_ENOUGH_THRESHOLD:-0.80}
# GEN-lane reward backend (agentic_image_gen.score_backend); default unified_reward_rubric.
PICKSCORE_GOOD_ENOUGH_THRESHOLD=${PICKSCORE_GOOD_ENOUGH_THRESHOLD:-1.0}
# The judge shares the last visible device (the emptiest); override JUDGE_GPU to move it.
if [[ -z "${JUDGE_GPU:-}" ]]; then
  JUDGE_GPU="${CUDA_VISIBLE_DEVICES##*,}"
fi
# JUDGE_IMAGE_MODEL normally comes from JUDGE_ENV_FILE; read just that var in a subshell
# (sourcing the file would clobber this recipe's state).
if [[ -z "${JUDGE_IMAGE_MODEL:-}" && -n "${JUDGE_ENV_FILE:-}" ]]; then
  JUDGE_IMAGE_MODEL="$(bash -c "source '${JUDGE_ENV_FILE}' >/dev/null 2>&1; printf '%s' \"\${JUDGE_IMAGE_MODEL:-}\"")"
fi
JUDGE_IMAGE_MODEL="${JUDGE_IMAGE_MODEL:-${_HF_HUB}/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"

_judge_ready() { curl -fsS -m 2 "${JUDGE_URL%/}/v1/models" >/dev/null 2>&1; }
JUDGE_SERVER_PID=""
if [[ "$JUDGE_SERVER" == "1" ]]; then
  if _judge_ready; then
    echo "[INFO] image judge already serving at ${JUDGE_URL}; reusing it"
  else
    if [[ ! -e "${JUDGE_IMAGE_MODEL}" ]]; then
      echo "[ERROR] JUDGE_IMAGE_MODEL=${JUDGE_IMAGE_MODEL} does not exist." >&2
      exit 1
    fi
    # NOT under $RUN_DIR: Hydra refuses to write into a run dir that already exists.
    JUDGE_LOG="/tmp/bagel_judge_${RUN_TS}.log"
    echo "[INFO] starting image judge on GPU ${JUDGE_GPU} (mem-util ${AGENTIC_REFLECT_GPU_MEM_UTIL:-0.32}); log ${JUDGE_LOG}"
    # Keep ONLY assignments on these continuation lines: a # comment terminates the command
    # and drops the rest. PYTHONPATH/PATH/LD_LIBRARY_PATH are needed by the sidecar exec.
    CUDA_VISIBLE_DEVICES="$JUDGE_GPU" \
      JUDGE_IMAGE_MODEL="$JUDGE_IMAGE_MODEL" \
      AGENTIC_REFLECT_GPU_MEM_UTIL="${AGENTIC_REFLECT_GPU_MEM_UTIL:-0.32}" \
      AGENTIC_REFLECT_MAX_NUM_SEQS="${AGENTIC_REFLECT_MAX_NUM_SEQS:-2}" \
      AGENTIC_REFLECT_MAX_MODEL_LEN="${AGENTIC_REFLECT_MAX_MODEL_LEN:-4096}" \
      PYTHONPATH="${_SCRIPT_DIR}/../agent_llm${PYTHONPATH:+:${PYTHONPATH}}" \
      PATH="${_REPO_ROOT}/.venv/bin:${PATH}" \
      LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
      PYTORCH_NVML_BASED_CUDA_CHECK="${PYTORCH_NVML_BASED_CUDA_CHECK:-1}" \
      bash "${_SCRIPT_DIR}/../agent_llm/run_judge_image_tool_server.sh" >"$JUDGE_LOG" 2>&1 &
    JUDGE_SERVER_PID=$!
    for _ in $(seq 1 240); do
      _judge_ready && break
      if ! kill -0 "$JUDGE_SERVER_PID" 2>/dev/null; then
        echo "[ERROR] image judge exited during startup; last lines of ${JUDGE_LOG}:" >&2
        tail -n 40 "$JUDGE_LOG" >&2 || true
        exit 1
      fi
      sleep 2
    done
    if ! _judge_ready; then
      echo "[ERROR] image judge not ready at ${JUDGE_URL} after 480s; see ${JUDGE_LOG}" >&2
      exit 1
    fi
    echo "[INFO] image judge ready at ${JUDGE_URL} (pid ${JUDGE_SERVER_PID})"
    trap 'kill "${JUDGE_SERVER_PID}" 2>/dev/null || true' EXIT INT TERM
  fi
elif [[ "$ENABLE_RM" == "1" ]]; then
  # Internal path: a stale JUDGE_URL is harmless (the router wins) and its absence is expected.
  if _judge_ready; then
    echo "[INFO] image judge detected at ${JUDGE_URL} but unused: the reward pool's router scores"
  else
    echo "[INFO] ENABLE_RM=1: image rewards and Yes/No signals come from the reward loop manager's"
    echo "[INFO]   own router (reward_router_address -> ${REWARD_MODEL:-<reward.reward_model.model_path>})"
  fi
fi

# data.val_batch_size only sets the batch; the V1 val path still iterates the whole val set
# (val_max_samples defaults to -1), so cap it — raise or set -1 to test validation itself.
# 32, not 8: a binary dim is 1/val_max_samples wide, so 8 samples quantize ``val-core`` in
# steps of 0.125 and a 0.03 jump is invisible, while 32 resolves it to 0.031. The val pass
# is 50-step ODE (deterministic), so the extra prompts cost wall time, not signal.
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-32}

# Fixed validation holdout: samples 9001/9002 (cafe poster, reflect + plan) and 9003/9004
# (CN poster, reflect + plan) -- the same four the Mode (2a) RPCO recipe dumps. One
# deterministic ODE render per validation step under ``rollout_images/val/step_XXXXXX/
# sample_900N.01/``, because ``val-core`` is a scalar mean over the whole val parquet: a
# lane can collapse into a degenerate but still-scoring mode and barely move it. These
# rows are artifacts only -- the holdout writes no TransferQueue row (see
# ``BagelCorlAgentLoopWorkerTQ._run_holdout_prompt``), so it cannot perturb the val batch.
export AGENTIC_VAL_VIZ="${AGENTIC_VAL_VIZ:-1}"

# Advantage normalization. ``False`` is Dr.GRPO (arXiv:2503.20783): the advantage is the raw
# centred reward, so the update scales with the actual signal and a near-tie is nearly a
# no-op. Stock GRPO divides by the group std, which is *gap-independent* — at N=2 every gap
# becomes +/-1/sqrt(2), i.e. a full-amplitude push on a coin flip. Read on both lanes: UND
# token GRPO (``compute_advantage_for_multi_trajectories``) and GEN FlowGRPO
# (``apply_gen_flowgrpo_advantage``).
NORM_ADV_BY_STD=${NORM_ADV_BY_STD:-False}

# Actor (UND) LR. The schedule below (5% warmup, cosine to 10% of peak) was already the
# mitigation for the old constant-1e-4 run, but the *peak* stayed 1e-4 -- and every run that
# reached the 60-85 step range still hit the divergence tripwire:
#
#   bagel_corl_prfix_t07_20261016_102501_a{1,2,3}: rollout_corr/kl 0.03 -> 14.8/16.9/13.7
#   at steps 84/70/77, with actor/entropy collapsing 0.17 -> 0.085. The recipe has no KL
#   anchor and no entropy bonus, so nothing pulls the actor back.
#
# A LoRA rank-8 actor on a 7B model does not need 1e-4; the comment that introduced the
# cosine schedule already said the late steps should "move ~10x less". Make that the peak:
# 1e-5 for UND, 1e-5 for the GEN lane (was 3e-5). ``ACTOR_LR``/``GEN_LR`` are knobs so a
# sweep can move one lane without editing the file.
ACTOR_LR=${ACTOR_LR:-1e-5}
GEN_LR=${GEN_LR:-1e-5}
LR_WARMUP_RATIO=${LR_WARMUP_RATIO:-0.05}
LR_SCHEDULER=${LR_SCHEDULER:-cosine}
MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}

# GEN (FlowGRPO) importance-ratio trust band. ``DiffusionLossConfig.clip_ratio`` defaults to
# 1e-4, and this recipe used to inherit it. Measured on
# ``bagel_corl_prfix_t07_20261016_102501_a1``: ``actor/ratio_std`` is 2.7e-3 at every step,
# i.e. ~27x the band, so ``actor/pg_clipfrac`` sat at 0.90-0.93 from step 1 (before any real
# policy movement) and never changed -- nearly every GEN sample was clipped on numerical
# noise rather than on policy change. 0.02 is ~7 sigma of that measured noise: wide enough
# that the clip band means "the policy moved", not "two forwards disagreed".
#
# Verify on the next run: ``actor/pg_clipfrac`` must drop well below 0.9 at step 1 and start
# tracking ``rollout_corr/*``. If it still pins at ~0.9, the ratio noise is a real engine
# mismatch and the fix is the forward, not the band.
GEN_CLIP_RATIO=${GEN_CLIP_RATIO:-0.02}

# Checkpointing. ``trainer.save_freq`` defaults to -1 (save nothing) and this recipe never
# overrode it, so every tripwire abort forfeited the whole run: the 20261016 session spent
# 6.5h x 4 launches re-learning the same steps 1-80, which is why no reward trend survived
# across attempts. Save every 25 steps (25 x ~5 min = ~2h of work) and keep two, so an abort
# restarts from the last good policy instead of the base checkpoint.
# ``resume_mode=auto`` reads ``checkpoints/<project>/<experiment>/latest_checkpointed_iteration.txt``;
# the experiment name is stable across attempts of one supervision session.
SAVE_FREQ=${SAVE_FREQ:-25}
RESUME_MODE=${RESUME_MODE:-auto}
MAX_CKPT_KEEP=${MAX_CKPT_KEEP:-2}

# Smoke defaults: N = sibling episodes (>=2 required for GEN training), S = seeds per call
# (default 1). Do not let a stale J/K from the shell leak in. ``GEN_N_DEFAULT`` carries the
# GPU-budget-appropriate N set above (2 for a smoke budget, 4 on >=4 GPUs).
if [[ -z "${N:-}" ]]; then
  if [[ -n "${J:-}" ]]; then N=$J; else N=${GEN_N_DEFAULT:-2}; fi
fi
if [[ -z "${S:-}" ]]; then
  if [[ -n "${K:-}" && "$K" -ge 2 ]]; then
    S=$K
  else
    S=1
  fi
fi
if [[ "$S" -lt 1 ]]; then
  echo "gen_samples_per_call S must be >= 1 (one image per call; got S=$S)." >&2
  exit 1
fi
if [[ "$N" -lt 1 ]]; then
  echo "rollout.n N must be >= 1 (got N=$N)." >&2
  exit 1
fi
if [[ "$S" -eq 1 && "$N" -lt 2 ]]; then
  # With one seed per call the FlowGRPO group for a call is the N siblings, so N=1 leaves
  # every advantage identically zero (the group mean equals the lone member). Fail loud
  # instead of training a zero GEN gradient for the whole run.
  echo "S=1 groups GEN by task, so N must be >= 2 or every GEN advantage is zero (got N=$N)." >&2
  exit 1
fi
echo "bagel_corl smoke: N=$N siblings, S=$S seeds/call (in-episode J/K are runtime)"

# Launch divisibility is checked on the episode count (ppo_mini_batch_size x N); the authoritative check is validate_bagel_corl_config.
PPO_MINI=${PPO_MINI:-$TRAIN_BSZ}
PPO_MICRO=${PPO_MICRO:-1}
if (( TRAIN_BSZ % PPO_MINI != 0 )); then
  echo "data.train_batch_size=${TRAIN_BSZ} must be divisible by ppo_mini_batch_size=${PPO_MINI}." >&2
  exit 1
fi
EPISODES_PER_UPDATE=$(( PPO_MINI * N ))
if (( EPISODES_PER_UPDATE % NUM_GPUS_ACTOR_ROLLOUT_REWARD != 0 )); then
  echo "(ppo_mini_batch_size=${PPO_MINI} x N=${N})=${EPISODES_PER_UPDATE} must be divisible by" >&2
  echo "n_gpus_per_node=${NUM_GPUS_ACTOR_ROLLOUT_REWARD}." >&2
  exit 1
fi
PER_RANK_EPISODES=$(( EPISODES_PER_UPDATE / NUM_GPUS_ACTOR_ROLLOUT_REWARD ))
if (( PPO_MICRO > 0 && PER_RANK_EPISODES % PPO_MICRO != 0 )); then
  echo "(ppo_mini_batch_size x N / n_gpus_per_node)=${PER_RANK_EPISODES} must be divisible by" >&2
  echo "ppo_micro_batch_size_per_gpu=${PPO_MICRO}." >&2
  exit 1
fi

# R2 conditioning cache: enable_prompt_embed_cache and *_routing_affinity are a pair (the
# pin keeps an S-group on one replica). All three are passed as +key=value because the
# composed RolloutConfig node lacks these DiffusionRolloutConfig fields at compose time.
AGENT_WORKERS=${AGENT_WORKERS:-8}
ENABLE_PEC=${ENABLE_PEC:-1}
PEC_SIZE=${PEC_SIZE:-$(( S * N * AGENT_WORKERS ))}
if (( PEC_SIZE < 32 )); then PEC_SIZE=32; fi
if [[ "${ENABLE_PEC}" == "1" ]]; then
  PEC_FLAG=True
  PEC_AFFINITY=True
else
  # Affinity without a cache is incoherent and is rejected in validate_bagel_corl_config.
  PEC_FLAG=False
  PEC_AFFINITY=False
fi
echo "bagel_corl R2: prompt_embed_cache=${PEC_FLAG} affinity=${PEC_AFFINITY} size=${PEC_SIZE} (S*N*workers=${S}x${N}x${AGENT_WORKERS})"

# UniGRPO-aligned launch knobs (RFC 453 §2/§4.4): per-expert LRs, lane weights, GRPO-Guard ablation.
# Keep comment lines OUTSIDE the backslash-continued launch command below.
#
# * data.max_* and rollout.prompt/response_length are pinned from MAX_PROMPT_LEN and must match.
# * val_kwargs is explicit so validation does not inherit the stochastic training rollout
#   (the '+' prefix is required; the schema has no pipeline/algo subtree there).
# * SYNC_REFILL_FAILED_GROUPS=1 refills all-failed groups instead of raising (0 fails loud).
# * agentic_image_gen.run_dir is pinned to $RUN_DIR so runs cannot overwrite each other's dumps.
# * rollout.repetition_penalty takes the '+' prefix. This tree requires the *installed* verl
#   (the one exporting ``build_rollout_dist_profiler`` -- the sibling-checkout probe at the top
#   rejects the older ``../verl``), and that verl's ``trainer/config/rollout/rollout.yaml`` no
#   longer declares the key: it kept ``temperature`` / ``top_p`` / ``top_k`` and moved the rest
#   into the ``RolloutConfig`` dataclass, which still carries ``repetition_penalty``. A plain
#   override therefore dies during composition with
#   ``Key 'repetition_penalty' is not in struct`` before any GPU work starts, and dropping it
#   is not an option: the dataclass default of 1.0 is what
#   ``composite_agent_loop`` stamps into every UND request, where the AR strategy's
#   ``setdefault`` lets it win over ``bagel_corl_deploy_ar.yaml``'s 1.05 -- i.e. silently back
#   to the untruncated, anti-repetition-free sampling that degenerates the checkpoint into
#   ``assistant\nassistant\n...`` (see the ROLLOUT_REPETITION_PENALTY comment above).
# * rollout.enable_prefix_caching is pinned False to match what both deploy yamls already ask
#   for. ``verl``'s generic default is True, and ``vllm_async_server`` forwards it verbatim as
#   an engine argument, which *overrides* the stage-level ``enable_prefix_caching: false`` in
#   ``bagel_corl_deploy_tp2.yaml`` (stage 0) and ``bagel_corl_deploy_ar.yaml`` (stages 0/1). The
#   branch's vllm-omni pin (0.30.0) validates that against the diffusion KV mode:
#   ``enable_prefix_caching=True requires diffusion_kv_mode='paged_scheduler'``, while the
#   Bagel deploy configs leave the mode at its ``dense_legacy`` default -- so engine init died
#   in ``AsyncOmni(**engine_args)`` before the first rollout. Setting it here rather than
#   enabling ``paged_scheduler`` keeps the diffusion KV implementation the deploy configs were
#   validated against; no stage in either yaml wants prefix caching on.
# * BAGEL_CORL_GEN_DECODE=sde records the latents/log-probs FlowGRPO needs; ode disables the GEN loss.
export BAGEL_CORL_GEN_DECODE=${GEN_DECODE:-sde}

# Fail the run the step the actor leaves its trust region, instead of only warning
# (``OmniBagelCoRLTrainerSync._divergence_tripwire``, which reads these two env vars).
#
# Default is 1 here, not the class default of 0. This recipe sets no KL anchor
# (``use_kl_loss`` stays false) and no entropy bonus (``entropy_coeff`` stays 0), so a step
# that has already left the region cannot come back on its own: every remaining step spends
# rollout + reward budget on a policy that will not be used. That is not a hypothetical --
# ``bagel_corl_prfix_t07_20261016_023749_a1`` crossed ``rollout_corr/kl`` 41.8 against the
# limit of 20 at step 84, kept going for another 115 steps, and produced nothing.
#
# Aborting is affordable because this recipe now checkpoints every ``SAVE_FREQ`` steps
# (``trainer.save_freq`` + ``resume_mode=auto``): an abort costs at most the steps since the
# last save, and the supervisor relaunch resumes from that checkpoint instead of step 0.
# Without checkpointing the abort forfeits the remaining budget -- that was the 20261016
# behaviour (4 launches, ~80 steps each, nothing accumulated), so do not set
# ``SAVE_FREQ<=0`` together with ``BAGEL_CORL_DIVERGENCE_ABORT=1``.
export BAGEL_CORL_DIVERGENCE_ABORT="${BAGEL_CORL_DIVERGENCE_ABORT:-1}"
"${_PY}" -m verl_omni.trainer.main_omni \
    hydra.run.dir=$RUN_DIR \
    agentic_image_gen.run_dir=$RUN_DIR \
    agentic_image_gen.vllm_url=$JUDGE_VLLM_URL \
    agentic_image_gen.good_enough_threshold=$JUDGE_GOOD_ENOUGH_THRESHOLD \
    agentic_image_gen.score_backend=${SCORE_BACKEND:-unified_reward_rubric} \
    agentic_image_gen.unified_good_enough_threshold=${UNIFIED_GOOD_ENOUGH_THRESHOLD:-0.62} \
    agentic_image_gen.pickscore_good_enough_threshold=$PICKSCORE_GOOD_ENOUGH_THRESHOLD \
    trainer.v1.trainer_mode=bagel_corl_sync \
    algorithm.norm_adv_by_std_in_grpo=$NORM_ADV_BY_STD \
    trainer.v1.sampler.sync_refill_failed_groups=${SYNC_REFILL_FAILED_GROUPS:-True} \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=$TRAIN_BSZ \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_PROMPT_LEN \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path=$model_name \
    actor_rollout_ref.model.tokenizer_path=$model_name \
    +actor_rollout_ref.model.algorithm=flow_grpo \
    actor_rollout_ref.model.model_type=diffusion_model \
    +actor_rollout_ref.model.composite_mode=bagel_corl \
    +actor_rollout_ref.model.architecture=OmniBagelForConditionalGeneration \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.model.lora_dtype=bfloat16 \
    actor_rollout_ref.model.lora.merge=True \
    actor_rollout_ref.model.target_modules="['q_proj','k_proj','v_proj','o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj','q_proj_moe_gen','k_proj_moe_gen','v_proj_moe_gen','o_proj_moe_gen','mlp_moe_gen.gate_proj','mlp_moe_gen.up_proj','mlp_moe_gen.down_proj']" \
    actor_rollout_ref.model.fsdp_layer_prefixes="['layers.']" \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=$LR_WARMUP_RATIO \
    actor_rollout_ref.actor.optim.lr_scheduler_type=$LR_SCHEDULER \
    actor_rollout_ref.actor.optim.min_lr_ratio=$MIN_LR_RATIO \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    +actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo \
    +actor_rollout_ref.actor.diffusion_loss.clip_ratio=$GEN_CLIP_RATIO \
    +actor_rollout_ref.model.lr_gen=$GEN_LR \
    +actor_rollout_ref.actor.diffusion_loss.loss_weight_und=1.0 \
    +actor_rollout_ref.actor.diffusion_loss.loss_weight_gen=1.0 \
    actor_rollout_ref.rollout.name=vllm_omni \
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEM_UTIL \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.max_model_len=$MAX_PROMPT_LEN \
    actor_rollout_ref.rollout.prompt_length=$MAX_PROMPT_LEN \
    actor_rollout_ref.rollout.response_length=$MAX_PROMPT_LEN \
    +actor_rollout_ref.rollout.pipeline.height=$GEN_HW \
    +actor_rollout_ref.rollout.pipeline.width=$GEN_HW \
    +actor_rollout_ref.rollout.pipeline.num_inference_steps=$GEN_STEPS \
    +actor_rollout_ref.rollout.pipeline.max_sequence_length=$MAX_PROMPT_LEN \
    +actor_rollout_ref.rollout.pipeline.cfg_text_scale=$CFG_TEXT_SCALE \
    +actor_rollout_ref.rollout.pipeline.cfg_img_scale=$CFG_IMG_SCALE \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$ROLLOUT_TOP_P \
    actor_rollout_ref.rollout.top_k=$ROLLOUT_TOP_K \
    +actor_rollout_ref.rollout.repetition_penalty=$ROLLOUT_REPETITION_PENALTY \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    +actor_rollout_ref.rollout.enable_prompt_embed_cache=$PEC_FLAG \
    +actor_rollout_ref.rollout.enable_prompt_embed_cache_routing_affinity=$PEC_AFFINITY \
    +actor_rollout_ref.rollout.prompt_embed_cache_size=$PEC_SIZE \
    actor_rollout_ref.rollout.agent.num_workers=$AGENT_WORKERS \
    +actor_rollout_ref.rollout.algo.noise_level=${NOISE_LEVEL:-0.7} \
    +actor_rollout_ref.rollout.algo.sde_window_size=${SDE_WINDOW_SIZE:-2} \
    +actor_rollout_ref.rollout.algo.sde_window_range=${SDE_WINDOW_RANGE:-[0,7]} \
    +actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=${VAL_GEN_STEPS:-50} \
    +actor_rollout_ref.rollout.val_kwargs.algo.noise_level=${VAL_NOISE_LEVEL:-0.0} \
    actor_rollout_ref.rollout.agent.default_agent_loop=bagel_multiturn_agent \
    +actor_rollout_ref.rollout.agent.gen_samples_per_call=$S \
    +actor_rollout_ref.rollout.agent.max_generate_passes=${MAX_GENERATE_PASSES:-3} \
    +actor_rollout_ref.rollout.agent.max_und_turns=${MAX_UND_TURNS:-8} \
    +actor_rollout_ref.rollout.agent.und_ar_serving_ready=$BAGEL_UND_AR_SERVING_READY \
    +actor_rollout_ref.rollout.agent.und_deploy_config=$BAGEL_UND_DEPLOY_CONFIG \
    +actor_rollout_ref.rollout.agent.und_n_gpus=$UND_N_GPUS \
    +actor_rollout_ref.rollout.agent.und_gpu_memory_utilization=$UND_GPU_MEM_UTIL \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.deploy_config=$BAGEL_DEPLOY_CONFIG \
    reward.num_workers=$(( ENABLE_RM == 1 ? NUM_GPUS_ACTOR_ROLLOUT_REWARD / REWARD_TP : 0 )) \
    reward.reward_model.enable=$ENABLE_RM \
    reward.reward_model.model_path=$reward_model_name \
    reward.reward_model.rollout.name=$REWARD_ENGINE \
    reward.custom_reward_function.path=pkg://verl_omni.utils.reward_score.bagel_rm_image_scorer \
    reward.custom_reward_function.name=compute_score \
    reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
    reward.reward_model.rollout.gpu_memory_utilization=$REWARD_GPU_MEM_UTIL \
    reward.reward_model.rollout.enforce_eager=True \
    reward.reward_model.rollout.free_cache_engine=False \
    reward.reward_model.rollout.skip_tokenizer_init=False \
    reward.reward_model.rollout.max_model_len=$REWARD_MAX_MODEL_LEN \
    +reward.reward_model.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=$REWARD_KV_CACHE_BYTES \
    trainer.val_before_train=True \
    trainer.test_freq=${TEST_FREQ:-30} \
    data.val_batch_size=${VAL_BSZ:-$TRAIN_BSZ} \
    data.val_max_samples=$VAL_MAX_SAMPLES \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$TOTAL_STEPS" \
    trainer.save_freq=$SAVE_FREQ \
    trainer.resume_mode=$RESUME_MODE \
    trainer.max_actor_ckpt_to_keep=$MAX_CKPT_KEEP \
    trainer.logger="['console','wandb']" \
    trainer.project_name=${PROJECT_NAME:-bagel_corl} \
    trainer.experiment_name=${EXPERIMENT_NAME:-bagel_corl_pr1} \
    trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
    trainer.nnodes=1 \
    ${DEFAULT_LOCAL_DIR:+trainer.default_local_dir="$DEFAULT_LOCAL_DIR"} \
    "$@"
