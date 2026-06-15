(multi_node_training)=
# Multi-Node Training

Last updated: 06/15/2026

Scale FlowGRPO (or any diffusion RL) training across multiple nodes. This guide
uses the Qwen-Image OCR LoRA example to explain every change needed when moving
from one node to a multi-node cluster.

## Introduction

The single-node quickstart runs all components — actor training, rollout
generation, reward scoring, and reference log-prob computation — on 4 GPUs
inside one machine. When you need more throughput (larger global batch, more
rollout samples, or faster iteration), you can scale horizontally by adding
more nodes.

Multi-node training in VeRL-Omni distributes the same components across
multiple machines connected by a high-speed interconnect (InfiniBand or RoCE).
The reference multi-node script
(`examples/flowgrpo_trainer/run_qwen_image_ocr_lora_multi_node.sh`) runs on
`NNODES × GPUS_PER_NODE` GPUs (default: 2 × 4 = 8) and achieves roughly linear
throughput scaling by increasing the global batch size while keeping per-GPU
work constant.

### What changes and what stays the same

| What changes | What stays the same |
|---|---|
| Global batch size (scaled by GPU count) | Per-GPU micro-batch size (constant) |
| Number of rollout replicas and reward workers | Rollout TP/DP (1 replica per GPU) |
| FSDP shard boundary (per-node) | Optimizer settings (lr, weight decay) |
| Trainer topology (`nnodes`, `n_gpus_per_node`) | Algorithm (FlowGRPO), model, LoRA config |
| Attention backend | Rollout sampling config (noise, SDE, CFG) |

### Architecture: how components are placed across nodes

```text
Node 0 (rank 0..3)                          Node 1 (rank 4..7)
┌─────────────────────────────┐          ┌─────────────────────────────┐
│  GPU 0: replica + reward    │          │  GPU 4: replica + reward    │
│  GPU 1: replica + reward    │          │  GPU 5: replica + reward    │
│  GPU 2: replica + reward    │          │  GPU 6: replica + reward    │
│  GPU 3: replica + reward    │          │  GPU 7: replica + reward    │
│  (reward model TP=4)        │          │  (reward model TP=4)        │
│  FSDP shard group (GPUs 0-3)│          │  FSDP shard group (GPUs 4-7)│
└─────────────────────────────┘          └─────────────────────────────┘
         ▲                                        ▲
         │         NCCL / InfiniBand              │
         └────────────────┬───────────────────────┘
                          │
              ┌───────────┴───────────┐
              │  Shared filesystem    │
              │  (models, data, ckpt) │
              └───────────────────────┘
```

With the default `ROLLOUT_TP=1` and `ROLLOUT_DP=1`, each GPU hosts one
independent vLLM-Omni rollout replica. Each replica runs entirely within one
GPU, requiring no cross-node coordination for generation. The reward model
(Tensor Parallelism = `REWARD_TP=4`) is **colocated** with the rollout
replicas on **every node** — one reward replica spans all 4 GPUs within a
node via Ray's fractional GPU scheduling, sharing each GPU with its rollout
replica. With 2 nodes this gives 2 independent reward replicas
(`reward.num_workers = TOTAL_GPUS / REWARD_TP = 2`), one per node. FSDP
shards the actor parameters within each node via
`fsdp_size=$GPUS_PER_NODE`, avoiding expensive cross-node all-gathers during
training.

## Prerequisites

- Complete the {doc}`single-node quickstart <flowgrpo_quickstart>` on one node
  first. Multi-node training uses the same dataset, models, and base
  configuration.
- At least two nodes, each with the same number of GPUs, connected by
  InfiniBand or RoCE. All nodes must have identical software environments
  (same Python, CUDA, and pip packages) and shared access to model weights,
  data, and a writable checkpoint directory (e.g., via NFS or HDFS).

## Conversion recipe: single-node → multi-node

The multi-node script at
`examples/flowgrpo_trainer/run_qwen_image_ocr_lora_multi_node.sh` is a
mechanical transformation of the single-node baseline at
`examples/flowgrpo_trainer/run_qwen_image_ocr_lora.sh`. The seven changes below
cover every line that differs.

### 1. Topology variables

Replace the hardcoded GPU count with cluster-wide variables:

```bash
# ── Single-node ─────────────────────
NUM_GPUS_ACTOR_ROLLOUT_REWARD=4

# ── Multi-node ──────────────────────
NNODES=${NNODES:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
TOTAL_GPUS=$((NNODES * GPUS_PER_NODE))     
```

The environment variables accept overrides so the same script works on any
cluster size:

```bash
NNODES=4 GPUS_PER_NODE=8 bash run_qwen_image_ocr_lora_multi_node.sh
```

### 2. Batch-size scaling

Scale `train_batch_size` and `ppo_mini_batch_size` by the GPU ratio so each
GPU processes the same amount of work as in the single-node run:

```bash
# ── Single-node ─────────────────────
data.train_batch_size=32
actor_rollout_ref.actor.ppo_mini_batch_size=16

# ── Multi-node ──────────────────────
TRAIN_BATCH_SIZE=$((32 * TOTAL_GPUS / 4))       
PPO_MINI_BATCH_SIZE=$((16 * TOTAL_GPUS / 4))    
PPO_MICRO_BATCH_PER_GPU=16                     

data.train_batch_size=$TRAIN_BATCH_SIZE
actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_PER_GPU
```

The divisor `4` is the single-node GPU count. If your baseline uses 8 GPUs,
divide by 8 instead. See {doc}`../algo/flowgrpo` for the detailed relationship
between `train_batch_size`, `ppo_mini_batch_size`, and
`ppo_micro_batch_size_per_gpu`.

### 3. Rollout and reward workers

Scale the number of agent-loop workers and reward workers to cover every GPU:

```bash
# ── Single-node ─────────────────────
actor_rollout_ref.rollout.agent.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / ROLLOUT_TP))
reward.num_workers=$((NUM_GPUS_ACTOR_ROLLOUT_REWARD / REWARD_TP))

# ── Multi-node ──────────────────────
ROLLOUT_NUM_WORKERS=$((TOTAL_GPUS / ROLLOUT_TP)) 
actor_rollout_ref.rollout.agent.num_workers=$ROLLOUT_NUM_WORKERS

reward.num_workers=$((TOTAL_GPUS / REWARD_TP))
```

Each rollout replica needs one AgentLoopWorker client; each reward replica
needs one reward worker. With `ROLLOUT_TP=1`, `ROLLOUT_DP=1`, and `REWARD_TP=4`
on 8 GPUs, you get 8 rollout replicas + 2 reward workers.

The rollout parallelism triplet (`ROLLOUT_TP`, `ROLLOUT_DP`,
`data_parallel_size`) controls how many GPUs each vLLM-Omni replica spans:

```bash
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP    # default: 1
actor_rollout_ref.rollout.data_parallel_size=1                       # default: 1
```

Replicas = `TOTAL_GPUS / (ROLLOUT_TP × data_parallel_size)`. The script
keeps `ROLLOUT_TP=1` and `data_parallel_size=1` so each GPU runs one
self-contained replica — the simplest and most robust layout for multi-node.
Increasing `ROLLOUT_TP` shards the rollout model across GPUs (useful when a
single GPU runs out of memory) but requires that `GPUS_PER_NODE` is divisible
by `ROLLOUT_TP × data_parallel_size`.

### 4. FSDP configuration

FSDP must shard within each node, not across the network:

```bash
# ── Single-node ─────────────────────
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \

# ── Multi-node ──────────────────────
actor_rollout_ref.actor.fsdp_config.fsdp_size=$GPUS_PER_NODE \
actor_rollout_ref.actor.fsdp_config.reshard_after_forward=False \
actor_rollout_ref.actor.fsdp_config.offload_policy=True \
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
```

| Parameter | Value | Why |
|---|---|---|
| `fsdp_size` | `$GPUS_PER_NODE` | Limits FSDP's communication group to GPUs on the same physical node, avoiding cross-node all-gather latency |
| `reshard_after_forward` | `False` | Keeps full parameters in memory after the forward pass instead of re-gathering them from shards. Trades memory for speed — safe with `offload_policy=True` |
| `offload_policy` | `True` | Moves policy parameters to CPU when the actor is idle, freeing GPU memory for rollout and reward |

### 5. Attention backend

Multi-node runs benefit from FlashAttention 3's varlen hub backend, which
optimizes variable-length sequence batching across distributed GPUs:

```bash
# ── Multi-node only ─────────────────
actor_rollout_ref.model.attn_backend="_flash_3_varlen_hub"
```

This replaces the default attention implementation and is strongly recommended
for any multi-node diffusion training run.

### 6. Trainer topology

Tell the trainer how many nodes and GPUs-per-node it has:

```bash
# ── Single-node ─────────────────────
trainer.n_gpus_per_node=$NUM_GPUS_ACTOR_ROLLOUT_REWARD \
trainer.nnodes=1 \

# ── Multi-node ──────────────────────
trainer.n_gpus_per_node=$GPUS_PER_NODE \
trainer.nnodes=$NNODES \
```

### 7. Experiment name

Tag the run with the cluster topology for bookkeeping:

```bash
# ── Single-node ─────────────────────
trainer.experiment_name=qwen_image_ocr_lora \

# ── Multi-node ──────────────────────
trainer.experiment_name=qwen_image_ocr_lora_multinode_${NNODES}x${GPUS_PER_NODE} \
```

## Full reference script

```{literalinclude} ../../examples/flowgrpo_trainer/run_qwen_image_ocr_lora_multi_node.sh
:language: bash
:caption: examples/flowgrpo_trainer/run_qwen_image_ocr_lora_multi_node.sh
```

## How to run

### Environment variables

The script reads these variables at runtime:

| Variable | Default | Description |
|---|---|---|
| `NNODES` | `2` | Total number of nodes |
| `GPUS_PER_NODE` | `4` | GPUs on each node |
| `WORKSPACE` | `$HOME` | Shared directory for data and checkpoints |

All other parameters (model paths, learning rate, LoRA config, etc.) are
hardcoded in the script. Override them by appending key-value pairs to the
command line, just like the single-node script.

### Step 1: Start Ray on the master node

```bash
ray start --head --num-gpus=$GPUS_PER_NODE \
  --node-ip-address=$MASTER_IP \
  --dashboard-host=0.0.0.0
```

Replace `$MASTER_IP` with the master node's IP address (e.g., `192.168.1.10`)
and `$GPUS_PER_NODE` with the number of GPUs on that node.

### Step 2: Start Ray on each slave node

```bash
ray start --address=$MASTER_IP:6379 --num-gpus=$GPUS_PER_NODE
```

Run this on **every** worker node to join the Ray cluster. All nodes must
share the same `$GPUS_PER_NODE`.

### Step 3: Launch training on the master node

```bash
bash examples/flowgrpo_trainer/run_qwen_image_ocr_lora_multi_node.sh
```

The script reads `NNODES` and `GPUS_PER_NODE` from the environment (defaults:
`NNODES=2`, `GPUS_PER_NODE=4`). Override as needed:

```bash
NNODES=2 GPUS_PER_NODE=4 bash examples/flowgrpo_trainer/run_qwen_image_ocr_lora_multi_node.sh
```
