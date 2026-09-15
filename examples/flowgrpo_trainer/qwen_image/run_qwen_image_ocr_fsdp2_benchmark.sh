#!/bin/bash
# Qwen-Image full-weight RL throughput benchmark for 8 x 80 GB GPUs.
#
# This is an incremental override of run_qwen_image_ocr.sh. FSDP2 is required
# because FSDP1 runs out of memory on this configuration. Rollout step execution
# and reward-model CUDA graphs/batching are enabled to improve throughput.
# FSDP2 forward prefetch is also enabled. The benchmark enables
# regional torch.compile while retaining Hub FA3 for actor training and rollout.
# Graph breaks are allowed so third-party attention preprocessing can stay eager
# without requiring coordinated compiler, Diffusers, and FA3 patches. Append
# actor_rollout_ref.model.use_regional_compile=False to compare against eager
# execution. This benchmark disables checkpoint saving and periodic validation.
#
# Keep recompiles shared because regional compilation invokes torch.compile
# once per repeated transformer block. The 60 structurally identical Qwen Image
# blocks can then reuse compiled entries instead of compiling every graph-break
# region independently for every block.
# Compile dynamic shapes up front because Qwen Image derives its rotary-embedding
# length from each batch's text mask. Static compilation specializes every
# repeated block for each prompt length and exhausts Dynamo's accumulated
# recompile limit before the first training step completes.
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
NUM_GPUS=${NUM_GPUS:-8}
NUM_NODES=${NUM_NODES:-1}
REWARD_TP=1
export TORCH_LOGS="${TORCH_LOGS:-graph_breaks,recompiles}"
echo "Using TORCH_LOGS=$TORCH_LOGS for torch.compile diagnostics."

NUM_GPUS=$NUM_GPUS NUM_NODES=$NUM_NODES bash "$SCRIPT_DIR/run_qwen_image_ocr.sh" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.model.use_regional_compile=True \
    'actor_rollout_ref.model.regional_compile_options={backend:inductor,mode:default,fullgraph:false,dynamic:true}' \
    actor_rollout_ref.rollout.step_execution=True \
    reward.num_workers=$((NUM_GPUS / REWARD_TP)) \
    reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
    reward.reward_model.rollout.enforce_eager=False \
    reward.reward_model.rollout.max_num_seqs=128 \
    reward.reward_model.rollout.max_model_len=8192 \
    trainer.logger='["console", "tensorboard"]' \
    trainer.experiment_name=qwen_image_ocr_8x80g_fsdp2_fa3_compile_graph_breaks_benchmark \
    trainer.resume_mode=disable \
    trainer.save_freq=0 \
    trainer.test_freq=0 \
    "$@"
