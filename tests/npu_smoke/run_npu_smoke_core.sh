#!/usr/bin/env bash

set -euo pipefail

export NPU_SMOKE_GROUP=core
source "$(dirname "${BASH_SOURCE[0]}")/lib_npu_smoke.sh" "$@"

print_header
run_test 0 "vllm-omni rollout + sleep/wake_up" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    python3 -m pytest -s tests/workers/rollout/rollout_vllm/test_vllm_omni_generate_npu.py
run_test 3 "diffusers FSDP engine inference + train" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    python3 -m pytest -s tests/workers/test_diffusers_fsdp_engine_npu.py
run_test 4 "diffusion agent loop request + step execution" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    python3 -m pytest -s tests/agent_loop/test_diffusion_agent_loop_npu.py::test_diffusion_single_turn_npu
run_test 5 "unseeded multi-worker rollout diversity" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    python3 -m pytest -s tests/agent_loop/test_diffusion_agent_loop_npu.py::test_rollout_without_seed_npu
run_test 6 "seeded multi-worker rollout reproducibility" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    python3 -m pytest -s tests/agent_loop/test_diffusion_agent_loop_npu.py::test_rollout_seed_reproducibility_npu
run_test 7 "standalone vLLM reward engine startup + OCR scoring" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    python3 -m pytest -s tests/reward_loop/test_reward_engine_npu.py::test_standalone_reward_engine_npu
run_test 11 "reward restart determinism and JPEG rule scoring" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    python3 -m pytest -s \
        tests/reward_loop/test_reward_engine_npu.py::test_deterministic_reward_reproducibility_single_replica_npu \
        tests/reward_loop/test_reward_engine_npu.py::test_deterministic_reward_reproducibility_multi_replica_npu \
        tests/reward_loop/test_reward_engine_npu.py::test_rule_reward_npu
run_test 12 "rollout request batching + step-execution contracts" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    python3 -m pytest -s tests/workers/rollout/rollout_vllm/test_vllm_omni_generate_contracts_npu.py
run_test 13 "diffusers FSDP/FSDP2 engine with sequence parallel 2" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    python3 -m pytest -s tests/workers/test_diffusers_fsdp_engine_sp_npu.py
run_test 14 "diffusers Ulysses sequence-parallel equivalence" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
        MODEL_PATH="${QWEN_IMAGE_MODEL_PATH:-${HOME}/.cache/modelscope/hub/models/tiny-random/Qwen-Image}" \
    python3 -m torch.distributed.run --nproc_per_node="${NUM_NPUS}" --local-ranks-filter=0 tests/workers/test_diffusers_ulysses.py
run_test 15 "trainable FSDP2 snapshot restore (two-rank)" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" REQUIRE_TRAINABLE_SNAPSHOT_GPU=1 \
    python3 -m pytest -s tests/workers/test_trainable_snapshot_gpu.py
run_test 17 "OmniAgentLoopWorker image_gen tool agent loop" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    python3 -m pytest -s tests/agent_loop/test_omni_agent_loop_worker_npu.py
# FSDP1 collection loses nested LoRA keys on the pinned stack; keep it visible
# as an uncovered scenario while exercising both FSDP2 adapter contracts.
run_test 28 "dual-adapter LoRA FSDP2 switch and EMA" \
    env ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
        NUM_GPUS="${NUM_NPUS}" DEVICE_NAME=npu \
        MODEL_PATH="${QWEN_IMAGE_MODEL_PATH:-${HOME}/.cache/modelscope/hub/models/tiny-random/Qwen-Image}" \
    python3 -m pytest -s tests/workers/test_diffusers_fsdp_lora_adapter.py \
        --deselect "tests/workers/test_diffusers_fsdp_lora_adapter.py::test_diffusers_fsdp_lora_adapter_switch[fsdp]" \
        --deselect "tests/workers/test_diffusers_fsdp_lora_adapter.py::test_diffusers_fsdp_lora_adapter_copy_ema[fsdp]"
run_test 30 "named image previews and H3 offload placeholders" \
    env \
        ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" NUM_NPUS="${NUM_NPUS}" \
    python3 -m pytest -q tests/trainer/diffusion/test_artifact_preview_dump_npu.py \
        tests/pipelines/test_minimax_h3_meta_output_npu.py
cleanup_runtime
print_summary
