#!/bin/bash
# Python-stack-first, aligned old_log_prob + update_actor profiling for the
# full-weight optimized Qwen-Image OCR recipe.
#
# Scope:
#   - the Ray TaskRunner ("controller"): Python sampling and NVTX;
#   - ActorRollout rank 0 by default: Python sampling and NVTX;
#   - all three training steps in one continuous capture window.
#
# CUDA API/kernel software tracing, Python GIL events, OS-runtime events, native
# CPU sampling, and context-switch events stay off. This keeps the diagnostic
# focused on Python control flow around old_log_prob and update_actor GPU-idle
# intervals without multiplying millions of software-instrumented events across
# all actor ranks. Analyze step 2 for steady-state behavior: step 1 contains
# profiler startup, while step 3 closes the capture.
#
# Each target process writes its own .nsys-rep. Open all reports together with
# Nsight Systems "New multi-report view". TSC is the preferred alignment source
# for reports from one host, but Nsight Systems may fall back to UTC.
#
# Run from the verl-omni repository root:
#   bash examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_fsdp2_benchmark_nsys.sh
#
# Optional environment controls:
#   RUN_DIR=<path>           output root; also supplies the default profile run ID
#   PROFILE_RUN_ID=<id>      report filename ID; supplies RUN_DIR when it is unset
#   TRAINING_STEPS=3          total run length; every step is captured continuously
#   PYTHON_SAMPLE_HZ=20       Python stack samples per second
#   PROFILE_RANKS="[0]"       "all" or a Hydra list such as "[0,7]"
#   PROFILE_FINALIZE_TIMEOUT_S=120
set -euo pipefail
set -x

WORKSPACE=${WORKSPACE:-$HOME}
if [[ -n "${RUN_DIR:-}" ]]; then
    default_profile_run_id=${RUN_DIR%/}
    default_profile_run_id=${default_profile_run_id##*/}
elif [[ -n "${PROFILE_RUN_ID:-}" ]]; then
    default_profile_run_id=$PROFILE_RUN_ID
    RUN_DIR=$WORKSPACE/logs/$default_profile_run_id
else
    default_profile_run_id="$(date +"%Y%m%d_%H%M%S")_$$"
    RUN_DIR=$WORKSPACE/logs/$default_profile_run_id
fi
TRAINING_STEPS=${TRAINING_STEPS:-3}
PYTHON_SAMPLE_HZ=${PYTHON_SAMPLE_HZ:-20}
PROFILE_RANKS=${PROFILE_RANKS:-"[0]"}
PROFILE_FINALIZE_TIMEOUT_S=${PROFILE_FINALIZE_TIMEOUT_S:-120}
NUM_GPUS_PROFILE=${NUM_GPUS:-8}
NUM_NODES_PROFILE=${NUM_NODES:-1}
profile_run_id=${PROFILE_RUN_ID:-$default_profile_run_id}

if ! [[ "$TRAINING_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: TRAINING_STEPS must be a positive integer." >&2
    exit 2
fi
if (( NUM_NODES_PROFILE != 1 )); then
    echo "ERROR: this recipe requires one node to favor TSC alignment across reports." >&2
    echo "Multi-node profiling also requires a collector for every node-local Ray session." >&2
    exit 2
fi
if ! [[ "$profile_run_id" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "ERROR: PROFILE_RUN_ID may contain only letters, digits, underscore, and dash." >&2
    exit 2
fi
if ! command -v nsys >/dev/null 2>&1; then
    echo "ERROR: nsys is not in PATH. Install Nsight Systems and expose its CLI in PATH." >&2
    exit 1
fi

profile_steps="["
for ((profile_step = 1; profile_step <= TRAINING_STEPS; profile_step++)); do
    if (( profile_step > 1 )); then
        profile_steps+=","
    fi
    profile_steps+="$profile_step"
done
profile_steps+="]"

if [[ "$PROFILE_RANKS" == "all" ]]; then
    expected_worker_reports=$NUM_GPUS_PROFILE
    actor_rank_overrides=(
        actor_rollout_ref.actor.profiler.all_ranks=True
        'actor_rollout_ref.actor.profiler.ranks=[]'
    )
else
    compact_ranks=${PROFILE_RANKS//[[:space:]]/}
    if ! [[ "$compact_ranks" =~ ^\[[0-9]+(,[0-9]+)*\]$ ]]; then
        echo 'ERROR: PROFILE_RANKS must be "all" or a list such as "[0,7]".' >&2
        exit 2
    fi
    rank_csv=${compact_ranks:1:${#compact_ranks}-2}
    IFS=, read -r -a selected_ranks <<< "$rank_csv"
    declare -A seen_ranks=()
    for rank in "${selected_ranks[@]}"; do
        if (( rank >= NUM_GPUS_PROFILE )); then
            echo "ERROR: rank $rank is outside NUM_GPUS=$NUM_GPUS_PROFILE." >&2
            exit 2
        fi
        if [[ -n "${seen_ranks[$rank]:-}" ]]; then
            echo "ERROR: duplicate rank $rank in PROFILE_RANKS." >&2
            exit 2
        fi
        seen_ranks[$rank]=1
    done
    expected_worker_reports=${#selected_ranks[@]}
    PROFILE_RANKS=$compact_ranks
    actor_rank_overrides=(
        actor_rollout_ref.actor.profiler.all_ranks=False
        "actor_rollout_ref.actor.profiler.ranks=$PROFILE_RANKS"
    )
fi
expected_reports=$((expected_worker_reports + 1))

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
base_recipe="$script_dir/run_qwen_image_ocr_fsdp2_benchmark.sh"
profile_dst="$RUN_DIR/nsight_update_actor"
mkdir -p "$profile_dst"
chmod 700 "$profile_dst"

# Keep native CPU sampling and software-instrumented CUDA/GIL/OSRT/scheduling
# events off. At 20 Hz, aligned Python stacks explain controller and worker
# control flow around the profiled phases.
training_status=0
bash "$base_recipe" \
    trainer.total_training_steps="$TRAINING_STEPS" \
    global_profiler.tool=nsys \
    "global_profiler.steps=$profile_steps" \
    global_profiler.profile_continuous_steps=True \
    global_profiler.global_tool_config.nsys.discrete=False \
    actor_rollout_ref.actor.profiler.tool=nsys \
    actor_rollout_ref.actor.profiler.enable=True \
    "${actor_rank_overrides[@]}" \
    actor_rollout_ref.ref.profiler.enable=False \
    actor_rollout_ref.rollout.profiler.enable=False \
    global_profiler.global_tool_config.nsys.controller_nsight_options.trace='"nvtx"' \
    global_profiler.global_tool_config.nsys.controller_nsight_options.cuda-memory-usage='"false"' \
    +global_profiler.global_tool_config.nsys.controller_nsight_options.capture-range=cudaProfilerApi \
    +global_profiler.global_tool_config.nsys.controller_nsight_options.capture-range-end='"repeat:1:async"' \
    +global_profiler.global_tool_config.nsys.controller_nsight_options.kill=none \
    +global_profiler.global_tool_config.nsys.controller_nsight_options.sample=none \
    +global_profiler.global_tool_config.nsys.controller_nsight_options.cpuctxsw=none \
    +global_profiler.global_tool_config.nsys.controller_nsight_options.python-sampling='"true"' \
    +global_profiler.global_tool_config.nsys.controller_nsight_options.python-sampling-frequency="$PYTHON_SAMPLE_HZ" \
    +global_profiler.global_tool_config.nsys.controller_nsight_options.wait=primary \
    +global_profiler.global_tool_config.nsys.controller_nsight_options.o="\"controller_update_actor_${profile_run_id}_%h_pid%p\"" \
    global_profiler.global_tool_config.nsys.worker_nsight_options.trace='"nvtx"' \
    global_profiler.global_tool_config.nsys.worker_nsight_options.cuda-memory-usage='"false"' \
    global_profiler.global_tool_config.nsys.worker_nsight_options.capture-range=cudaProfilerApi \
    global_profiler.global_tool_config.nsys.worker_nsight_options.capture-range-end='"repeat:1:async"' \
    global_profiler.global_tool_config.nsys.worker_nsight_options.kill=none \
    +global_profiler.global_tool_config.nsys.worker_nsight_options.sample=none \
    +global_profiler.global_tool_config.nsys.worker_nsight_options.cpuctxsw=none \
    +global_profiler.global_tool_config.nsys.worker_nsight_options.python-sampling='"true"' \
    +global_profiler.global_tool_config.nsys.worker_nsight_options.python-sampling-frequency="$PYTHON_SAMPLE_HZ" \
    +global_profiler.global_tool_config.nsys.worker_nsight_options.wait=primary \
    +global_profiler.global_tool_config.nsys.worker_nsight_options.o="\"actor_update_${profile_run_id}_rank_%q{RANK}_%h_pid%p\"" \
    "$@" || training_status=$?

# Ray fixes nsys output below its current session. Copy only files created by
# this recipe; leave any unrelated profiler reports in the Ray session alone.
# Keep .qdstrm too: it is the recoverable raw stream if a node lacks the host
# importer needed to finalize .nsys-rep locally.
nsight_src="${TMPDIR:-/tmp}/ray/session_latest/logs/nsight"
controller_prefix="controller_update_actor_${profile_run_id}_"
actor_prefix="actor_update_${profile_run_id}_rank_"

# Controller and actor capture start once before step 1 and stop once after the
# final step. Result generation then starts asynchronously. Wait until every
# matching report is present and stable before copying it.
finalize_deadline=$((SECONDS + PROFILE_FINALIZE_TIMEOUT_S))
last_artifact_size=-1
stable_artifact_checks=0
no_finalizer_checks=0
reports_ready=0
while (( SECONDS < finalize_deadline )); do
    shopt -s nullglob
    pending_reports=(
        "$nsight_src"/"$controller_prefix"*.nsys-rep
        "$nsight_src"/"$actor_prefix"*.nsys-rep
    )
    pending_raw=(
        "$nsight_src"/"$controller_prefix"*.qdstrm
        "$nsight_src"/"$actor_prefix"*.qdstrm
    )
    observed_report_count=0
    observed_raw_count=0
    artifact_size=0
    for artifact in "${pending_reports[@]}"; do
        if artifact_bytes=$(stat -c %s "$artifact" 2>/dev/null); then
            if (( artifact_bytes > 0 )); then
                observed_report_count=$((observed_report_count + 1))
            fi
            artifact_size=$((artifact_size + artifact_bytes))
        fi
    done
    for artifact in "${pending_raw[@]}"; do
        if artifact_bytes=$(stat -c %s "$artifact" 2>/dev/null); then
            observed_raw_count=$((observed_raw_count + 1))
            artifact_size=$((artifact_size + artifact_bytes))
        fi
    done
    if (( observed_report_count == expected_reports && observed_raw_count == 0 )) \
        && (( artifact_size == last_artifact_size )); then
        stable_artifact_checks=$((stable_artifact_checks + 1))
    else
        stable_artifact_checks=0
    fi
    if (( stable_artifact_checks >= 2 )); then
        reports_ready=1
        break
    fi

    if pgrep -u "$(id -u)" -f "(nsys|QdstrmImporter).*${profile_run_id}" >/dev/null 2>&1; then
        no_finalizer_checks=0
    else
        no_finalizer_checks=$((no_finalizer_checks + 1))
    fi
    if (( no_finalizer_checks >= 2 )); then
        if (( observed_report_count == expected_reports && observed_raw_count == 0 )); then
            reports_ready=1
        fi
        break
    fi

    last_artifact_size=$artifact_size
    sleep 1
done

report_count=0
raw_count=0
shopt -s nullglob
artifacts=(
    "$nsight_src"/"$controller_prefix"*
    "$nsight_src"/"$actor_prefix"*
)
for artifact in "${artifacts[@]}"; do
    artifact_name=${artifact##*/}
    case "$artifact" in
        *.nsys-rep)
            if [[ -s "$artifact" ]] && cp -v "$artifact" "$profile_dst/"; then
                chmod 600 "$profile_dst/$artifact_name"
                report_count=$((report_count + 1))
            else
                echo "WARNING: report is empty or vanished while copying: $artifact" >&2
            fi
            ;;
        *.qdstrm)
            if cp -v "$artifact" "$profile_dst/"; then
                chmod 600 "$profile_dst/$artifact_name"
                raw_count=$((raw_count + 1))
            else
                echo "WARNING: raw stream vanished while copying: $artifact" >&2
            fi
            ;;
    esac
done

report_error=0
if (( reports_ready == 0 )); then
    echo "ERROR: reports did not become complete and stable within ${PROFILE_FINALIZE_TIMEOUT_S}s." >&2
    report_error=1
fi
if (( report_count != expected_reports )); then
    echo "ERROR: expected $expected_reports finalized reports (controller + selected actor ranks)," \
        "found $report_count (raw streams: $raw_count)." >&2
    report_error=1
fi

echo "Copied $report_count report(s) and $raw_count raw stream(s) to $profile_dst"
echo "Profile run ID: $profile_run_id"
echo "Open all *.nsys-rep in one Nsight Systems multi-report view."
echo "Verify Analysis Summary -> Report alignment source = TSC before comparing ranks."
echo "Correlate controller and actor Python backtraces with the profiling NVTX ranges."

if (( training_status != 0 )); then
    echo "ERROR: training exited with status $training_status; targeted artifacts above were preserved." >&2
    exit "$training_status"
fi
if (( report_error != 0 )); then
    exit "$report_error"
fi
