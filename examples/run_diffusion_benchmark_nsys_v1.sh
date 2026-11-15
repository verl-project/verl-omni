#!/bin/bash
# Aligned end-to-end diffusion-step profiling for a V1 benchmark recipe.
#
# This entrypoint shares launch and capture settings with the V0 recipe, but
# adds rollout, reward, and GPU metrics profiling. Collection waits for actual
# Nsight sessions rather than predicting the number of reports.
set -euo pipefail

if (( $# == 0 )); then
    echo "Usage: $0 <benchmark-recipe> [Hydra overrides...]" >&2
    exit 2
fi
base_recipe=$1
shift

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
extra_overrides=(
    actor_rollout_ref.rollout.profiler.tool=nsys
    actor_rollout_ref.rollout.profiler.enable=True
    actor_rollout_ref.rollout.profiler.all_ranks=False
    'actor_rollout_ref.rollout.profiler.ranks=[0]'
    reward.profiler.tool=nsys
    reward.profiler.enable=True
    reward.profiler.all_ranks=False
    'reward.profiler.ranks=[0]'
    ++global_profiler.global_tool_config.nsys.gpu_metrics_nsight_options.trace='"cuda-sw"'
    ++global_profiler.global_tool_config.nsys.gpu_metrics_nsight_options.cuda-memory-usage='"false"'
    ++global_profiler.global_tool_config.nsys.gpu_metrics_nsight_options.capture-range=cudaProfilerApi
    ++global_profiler.global_tool_config.nsys.gpu_metrics_nsight_options.capture-range-end='"repeat:1:async"'
    ++global_profiler.global_tool_config.nsys.gpu_metrics_nsight_options.kill=none
    ++global_profiler.global_tool_config.nsys.gpu_metrics_nsight_options.sample=none
    ++global_profiler.global_tool_config.nsys.gpu_metrics_nsight_options.cpuctxsw=none
    ++global_profiler.global_tool_config.nsys.gpu_metrics_nsight_options.cudabacktrace=none
    ++global_profiler.global_tool_config.nsys.gpu_metrics_nsight_options.gpu-metrics-devices='"all"'
    ++global_profiler.global_tool_config.nsys.gpu_metrics_nsight_options.gpu-metrics-frequency=20
    ++global_profiler.global_tool_config.nsys.gpu_metrics_nsight_options.wait=primary
    '++global_profiler.global_tool_config.nsys.gpu_metrics_nsight_options.o="gpu_metrics_@RUN_ID@_%h_pid%p"'
)
exec bash "$script_dir/run_diffusion_benchmark_nsys_v0.sh" \
    "$base_recipe" "${extra_overrides[@]}" "$@"
