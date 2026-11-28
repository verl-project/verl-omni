#!/bin/bash
# Aligned end-to-end diffusion-step profiling for a V0 benchmark recipe.
#
# Scope:
#   - the Ray TaskRunner ("controller"): Python and native CPU sampling,
#     context switches, OS runtime, NVTX, and CUDA;
#   - ActorRollout rank 0 by default: Python sampling, NVTX, CUDA, and OS runtime;
#   - training steps 2 through the configured final step in one continuous
#     capture window.
#
# Rank-selected actor workers additionally capture OS-runtime events, native
# CPU samples, and context switches. CUDA tracing uses the software activity
# backend so independent role-local sessions can share a GPU without competing
# for the exclusive hardware tracing resource. CUDA API backtraces and memory
# usage tracking stay disabled to limit steady-state overhead. Together with
# Python sampling, the remaining signals distinguish
# Python GC from CUDA synchronization, allocator/IPC cleanup, collectives, and
# OS waits during update_weights. The same CPU-side signals are collected for
# the controller and selected workers.
# Step 1 is an unprofiled model/engine warmup. Step 2 pays profiler startup cost,
# every intermediate step is a steady-state analysis interval, and the final
# step pays profiler stop cost. Excluding warmup also reduces report size.
# Four steps provide a steady-state step between profiler start and stop. A
# two-step run is also supported for pipeline smoke testing.
#
# Each target process writes its own .nsys-rep. Open all reports together with
# Nsight Systems "New multi-report view". TSC is the preferred alignment source
# for reports from one host, but Nsight Systems may fall back to UTC.
#
# Usage:
#   bash examples/run_diffusion_benchmark_nsys_v0.sh <benchmark-recipe> [Hydra overrides...]
#
# Optional environment controls:
#   RUN_DIR=<path>           output root; also supplies the default profile run ID
#   PROFILE_RUN_ID=<id>      report filename ID; supplies RUN_DIR when it is unset
#   TRAINING_STEPS=4          total run length; steps 2 through N are captured
#   PROFILE_FINALIZE_TIMEOUT_S=120
set -euo pipefail
set -x

if (( $# == 0 )); then
    echo "Usage: $0 <benchmark-recipe> [Hydra overrides...]" >&2
    exit 2
fi
base_recipe=$1
shift
if [[ ! -f "$base_recipe" ]]; then
    echo "ERROR: profiling base recipe does not exist: $base_recipe" >&2
    exit 2
fi

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
TRAINING_STEPS=${TRAINING_STEPS:-4}
PROFILE_FINALIZE_TIMEOUT_S=${PROFILE_FINALIZE_TIMEOUT_S:-120}
NUM_NODES_PROFILE=${NUM_NODES:-1}
profile_run_id=${PROFILE_RUN_ID:-$default_profile_run_id}

if ! [[ "$TRAINING_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: TRAINING_STEPS must be a positive integer." >&2
    exit 2
fi
if (( TRAINING_STEPS < 2 )); then
    echo "ERROR: TRAINING_STEPS must be at least 2 for an unprofiled warmup and a captured step." >&2
    exit 2
fi
if (( TRAINING_STEPS < 4 )); then
    echo "WARNING: TRAINING_STEPS=$TRAINING_STEPS is suitable for a pipeline smoke test," >&2
    echo "WARNING: but it has no steady-state step between profiler start and stop." >&2
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
if ! [[ "$PROFILE_FINALIZE_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: PROFILE_FINALIZE_TIMEOUT_S must be a positive integer." >&2
    exit 2
fi
if ! command -v nsys >/dev/null 2>&1; then
    echo "ERROR: nsys is not in PATH. Install Nsight Systems and expose its CLI in PATH." >&2
    exit 1
fi

# Continuous profiling starts immediately before step 2 and stops immediately
# after the configured final step. Steps 3 through N-1 are steady-state
# intervals, so TRAINING_STEPS=4 provides one such interval.
profile_steps="["
for ((profile_step = 2; profile_step <= TRAINING_STEPS; profile_step++)); do
    if (( profile_step > 2 )); then
        profile_steps+=","
    fi
    profile_steps+="$profile_step"
done
profile_steps+="]"

profile_dst="$RUN_DIR/nsight"
mkdir -p "$profile_dst"
chmod 700 "$profile_dst"

# Entrypoints and callers can reference this experiment's run ID in overrides.
set -- "${@//@RUN_ID@/$profile_run_id}"

query_nsight_sessions() {
    local started_ns status=0
    started_ns=$(date +%s%N)
    nsys sessions list --output-format=json || status=$?
    printf 'NSIGHT_QUERY phase=%s exit_code=%s elapsed_ms=%s\n' \
        "$1" "$status" "$((($(date +%s%N) - started_ns) / 1000000))" >&2
    return "$status"
}

# Let Nsight assign unique names, including for actors inheriting runtime_env.
# This single-node recipe assumes no concurrent Nsight run by the same user.
if ! sessions_before=$(query_nsight_sessions before_training); then
    echo "ERROR: unable to query Nsight sessions before training." >&2
    exit 1
fi
if ! session_names_before=$(python -c '
import json, sys
print(json.dumps([s["name"] for s in json.load(sys.stdin)]))
' <<< "$sessions_before"); then
    echo "ERROR: invalid Nsight session status output before training." >&2
    exit 1
fi

training_status=0
bash "$base_recipe" \
    trainer.total_training_steps="$TRAINING_STEPS" \
    global_profiler.tool=nsys \
    "global_profiler.steps=$profile_steps" \
    global_profiler.profile_continuous_steps=True \
    global_profiler.global_tool_config.nsys.discrete=False \
    actor_rollout_ref.actor.profiler.tool=nsys \
    actor_rollout_ref.actor.profiler.enable=True \
    actor_rollout_ref.actor.profiler.all_ranks=False \
    'actor_rollout_ref.actor.profiler.ranks=[0]' \
    actor_rollout_ref.ref.profiler.enable=False \
    global_profiler.global_tool_config.nsys.controller_nsight_options.trace='"nvtx,cuda-sw,osrt"' \
    global_profiler.global_tool_config.nsys.controller_nsight_options.cuda-memory-usage='"false"' \
    +global_profiler.global_tool_config.nsys.controller_nsight_options.capture-range=cudaProfilerApi \
    +global_profiler.global_tool_config.nsys.controller_nsight_options.capture-range-end='"repeat:1:async"' \
    +global_profiler.global_tool_config.nsys.controller_nsight_options.kill=none \
    +global_profiler.global_tool_config.nsys.controller_nsight_options.sample=process-tree \
    +global_profiler.global_tool_config.nsys.controller_nsight_options.cpuctxsw=process-tree \
    +global_profiler.global_tool_config.nsys.controller_nsight_options.backtrace=dwarf \
    +global_profiler.global_tool_config.nsys.controller_nsight_options.samples-per-backtrace=8 \
    +global_profiler.global_tool_config.nsys.controller_nsight_options.osrt-threshold=10000 \
    +global_profiler.global_tool_config.nsys.controller_nsight_options.osrt-backtrace-threshold=80000 \
    +global_profiler.global_tool_config.nsys.controller_nsight_options.cudabacktrace=none \
    +global_profiler.global_tool_config.nsys.controller_nsight_options.python-sampling='"true"' \
    +global_profiler.global_tool_config.nsys.controller_nsight_options.python-sampling-frequency=20 \
    +global_profiler.global_tool_config.nsys.controller_nsight_options.wait=primary \
    +global_profiler.global_tool_config.nsys.controller_nsight_options.o="\"controller.run-${profile_run_id}.%h.pid%p\"" \
    global_profiler.global_tool_config.nsys.worker_nsight_options.trace='"nvtx,cuda-sw,osrt"' \
    global_profiler.global_tool_config.nsys.worker_nsight_options.cuda-memory-usage='"false"' \
    global_profiler.global_tool_config.nsys.worker_nsight_options.capture-range=cudaProfilerApi \
    global_profiler.global_tool_config.nsys.worker_nsight_options.capture-range-end='"repeat:1:async"' \
    global_profiler.global_tool_config.nsys.worker_nsight_options.kill=none \
    +global_profiler.global_tool_config.nsys.worker_nsight_options.sample=process-tree \
    +global_profiler.global_tool_config.nsys.worker_nsight_options.cpuctxsw=process-tree \
    +global_profiler.global_tool_config.nsys.worker_nsight_options.backtrace=dwarf \
    +global_profiler.global_tool_config.nsys.worker_nsight_options.samples-per-backtrace=8 \
    +global_profiler.global_tool_config.nsys.worker_nsight_options.osrt-threshold=10000 \
    +global_profiler.global_tool_config.nsys.worker_nsight_options.osrt-backtrace-threshold=80000 \
    +global_profiler.global_tool_config.nsys.worker_nsight_options.cudabacktrace=none \
    +global_profiler.global_tool_config.nsys.worker_nsight_options.python-sampling='"true"' \
    +global_profiler.global_tool_config.nsys.worker_nsight_options.python-sampling-frequency=20 \
    +global_profiler.global_tool_config.nsys.worker_nsight_options.wait=primary \
    +global_profiler.global_tool_config.nsys.worker_nsight_options.o="\"worker.run-${profile_run_id}.rank%q{RANK}.%h.pid%p\"" \
    "$@" || training_status=$?

# Ray fixes nsys output below its current session. Copy only files created by
# this recipe; leave any unrelated profiler reports in the Ray session alone.
# Keep .qdstrm too: it is the recoverable raw stream if a node lacks the host
# importer needed to finalize .nsys-rep locally.
# Assume session_latest still points to this run's cluster. Starting another
# cluster can redirect it; reusing a cluster and run ID can include old reports.
nsight_src="${TMPDIR:-/tmp}/ray/session_latest/logs/nsight"
# Independent actors may prepend role/rank fields. Match the complete run field.
# Custom output templates must preserve .run-<id>. for this collector.
artifact_pattern="*.run-${profile_run_id}.*"

# Training has returned, so no further captures should start. Nsight keeps
# asynchronous report generation visible in its session state, even before a
# report appears in the output directory. Idle sessions may outlive training.
# Do not infer completeness from report counts, file sizes, or importer PIDs.
finalize_started_s=$SECONDS
report_error=0
while true; do
    if ! sessions=$(query_nsight_sessions after_training); then
        echo "ERROR: unable to query Nsight sessions." >&2
        report_error=1
        break
    fi
    # Unknown states remain pending rather than being mistaken for completion.
    if ! pending_sessions=$(python -c '
import json, sys
idle = {"Launched", "ConfiguredLaunched", "StartRange"}
existing = set(json.loads(sys.argv[1]))
print("\n".join(s["name"] + " " + s["state"] for s in json.load(sys.stdin)
                if s["name"] not in existing and s["state"] not in idle))
' "$session_names_before" <<< "$sessions"); then
        echo "ERROR: invalid Nsight session status output." >&2
        report_error=1
        break
    fi
    if [[ -z "$pending_sessions" ]]; then
        break
    fi
    # Check the budget between queries; an in-flight query is not interrupted.
    if (( SECONDS - finalize_started_s >= PROFILE_FINALIZE_TIMEOUT_S )); then
        echo "ERROR: Nsight sessions still pending after ${PROFILE_FINALIZE_TIMEOUT_S}s waiting budget:" >&2
        echo "$pending_sessions" >&2
        report_error=1
        break
    fi
    sleep 1
done
echo "NSIGHT_FINALIZE elapsed_s=$((SECONDS - finalize_started_s)) exit_code=$report_error"

# On failure, still preserve available reports and recoverable raw streams.
report_count=0
raw_count=0
shopt -s nullglob
artifacts=("$nsight_src"/$artifact_pattern)
for artifact in "${artifacts[@]}"; do
    artifact_name=${artifact##*/}
    case "$artifact" in
        *.nsys-rep|*.qdstrm)
            if [[ -s "$artifact" ]] && cp -v "$artifact" "$profile_dst/"; then
                chmod 600 "$profile_dst/$artifact_name"
                if [[ "$artifact" == *.nsys-rep ]]; then
                    report_count=$((report_count + 1))
                else
                    raw_count=$((raw_count + 1))
                fi
            else
                echo "ERROR: unable to copy artifact: $artifact" >&2
                report_error=1
            fi
            ;;
    esac
done

if (( report_count == 0 )); then
    echo "ERROR: no .nsys-rep reports were copied." >&2
    report_error=1
fi

echo "Copied $report_count report(s) and $raw_count raw stream(s) to $profile_dst"
echo "Profile run ID: $profile_run_id"
echo "Open all *.nsys-rep in one Nsight Systems multi-report view."
echo "Verify Analysis Summary -> Report alignment source = TSC before comparing ranks."
echo "Correlate the captured process backtraces across the profiling window."

if (( training_status != 0 )); then
    echo "ERROR: training exited with status $training_status; targeted artifacts above were preserved." >&2
    exit "$training_status"
fi
if (( report_error != 0 )); then
    exit "$report_error"
fi
