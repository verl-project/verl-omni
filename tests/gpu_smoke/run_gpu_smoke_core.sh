#!/usr/bin/env bash
# Core GPU smoke tests (2-GPU): rollout, engines, agent loop, and reward loop.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib_gpu_smoke.sh"
gpu_smoke_init "core" 2 "$@"

run_test 0 "vllm-omni rollout" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" \
    pytest -s tests/workers/rollout/rollout_vllm/test_vllm_omni_generate.py

run_test 1 "diffusion agent loop" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" \
    pytest -s tests/agent_loop/test_diffusion_agent_loop.py

run_test 2 "diffusers FSDP engine" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" \
    pytest -s tests/workers/test_diffusers_fsdp_engine.py

# Skips itself if the optional `veomni` backend is not installed (importorskip).
run_test 3 "diffusers VeOmni engine" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" \
    pytest -s tests/workers/test_diffusers_veomni_engine.py

run_test 4 "diffusion rollout seed multi-worker" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" \
    pytest -s tests/agent_loop/test_diffusion_rollout_seed_gpu.py

run_test 5 "visual reward manager" \
    env CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_LIST}" \
    pytest -s tests/reward_loop/test_visual_reward_manager.py

gpu_smoke_summary
