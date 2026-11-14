# Experimental FSDP2 reward model design

Last updated: 11/13/2026.

This design document is for developers maintaining reward execution, placement and lifecycle management.
FSDP2 reward-model support is experimental; its current scope and lifecycle guarantees are described below.
It explains how a frozen reward model spans two GPUs within the existing named native reward interface.
For scorer configuration and the inference/scoring contract, see [Named Reward Models](../algo/named_reward_models.md).

## Supported deployment

The initial adapter supports pretrained PickScore with float32 inference, one node and one dedicated two-GPU
reward pool. It requires synchronous V1 diffusion training, one scoring facade and required reward terms.
Streaming, shared-GPU phase scheduling, additional adapters and trainable reward parameters are separate integrations.

`placement.replica_groups: [[0, 1]]` selects one logical replica with two ordered ranks in the parent reward pool.
Both indices must identify distinct GPU bundles. Flat `placement.devices` continues to select complete model replicas.
An omitted or null `fsdp` uses default deadlines for replica groups and leaves flat placement unchanged.

```yaml
trainer:
  use_v1: true
  v1:
    trainer_mode: sync
reward:
  num_workers: 1
  reward_model:
    enable: false
    enable_resource_pool: true
    nnodes: 1
    n_gpus_per_node: 2
  reward_manager:
    name: MultiVisualRewardManager
  models:
    pickscore:
      backend: native
      offload: true
      model_path: /path/to/PickScore_v1
      placement:
        replica_groups: [[0, 1]]
      executor:
        model: verl_omni.utils.reward_score.pickscore_reward:PickScoreNativeModel
        kwargs:
          processor_path: /path/to/clip_processor
      fsdp:
        timeout: 120
        collective_timeout: 30
  reward_functions:
    pickscore:
      path: pkg://verl_omni.utils.reward_score.pickscore_reward
      name: compute_score_pickscore_native
      required: true
```

The checkpoint must remain immutable across load/unload cycles. Both deadlines must be finite and positive.
The diffusion entrypoint rejects unsupported placement, executor and trainer settings before selecting the V1 or
legacy trainer and before allocating actors.

## Execution and ownership

```mermaid
flowchart LR
    dispatcher[Named reward dispatcher] --> owner[Deployment owner]
    owner --> facade[CPU scoring facade]
    facade --> rank0[FSDP2 rank 0]
    facade --> rank1[FSDP2 rank 1]
```

The facade presents one logical scoring worker to the existing dispatcher. It serializes complete batches across
the rank pair, so both ranks enter the same collective sequence. Each rank executes the same preprocessed inputs;
only rank zero publishes raw features. Existing reward functions retain score calculation and weighting.

The rank adapter uses the existing FSDP2 loading utilities and DTensor parameters. CLIP feature methods run inside
one managed forward boundary; root parameters are explicitly reshared after inference. Wake reloads the immutable
checkpoint. Sleep waits for accepted scoring to finish, then releases weights on both ranks.

The deployment owner retains rank handles and records the facade before its readiness RPC. This lets construction
rollback terminate actors even when the facade constructor or readiness check fails.

## Failure and shutdown contract

A rank error, timeout, partial submission or facade failure invalidates the whole replica. Cleanup attempts the
facade and both ranks even if one termination fails. A definite actor-death response is required for success;
temporary unavailability and timeout remain cleanup failures. Primary and cleanup errors are preserved together.
Rank termination probes use Ray's built-in readiness RPC, which remains valid after process-group teardown.
The failed replica is not retried or reused.

Concurrent close callers wait for the same owned termination to settle and observe its retained error. Cancelling
a caller does not release accepted work early. Waiting for the termination lock runs outside the event loop.
Trainer finalization closes reward resources and the transfer queue on both success and failure.

## Validation

The lightweight CPU contracts are in `tests/reward_loop/test_native_fsdp_integration_on_cpu.py`; they do not start
Ray or load model weights. Manual accelerator validators cover pretrained parity, constructor/readiness failures,
rank/facade faults and repeated load/unload. Cleanup checks retain actor IDs and require terminal states before
Ray shutdown. End-to-end SD3.5 training additionally checks finite gradients, changed LoRA tensors, exact Adam
counters and checkpoint continuation. These checks establish correctness for the exercised configuration.

Run from the repository root with local immutable checkpoints and two GPUs with sufficient free memory:

```bash
CUDA_VISIBLE_DEVICES='' "$PYTHON" -m pytest \
  tests/reward_loop/test_native_fsdp_integration_on_cpu.py -q --asyncio-mode=auto
CUDA_VISIBLE_DEVICES="$GPU_PAIR" "$PYTHON" -m tests.reward_loop.validate_native_fsdp \
  --model-path "$PICKSCORE_MODEL" --processor-path "$PICKSCORE_PROCESSOR" \
  --media-jsonl "$MEDIA_JSONL" --output "$PARITY_OUTPUT"
```

The manual SD3.5 functional study uses 128 training and 32 held-out prompts from a pinned upstream PickScore
split. Preparation selects the first unique captions with 5–20 whitespace-separated words and at most 140
characters, preserves source order and excludes every training caption from validation. It records source
indices, selected captions and file checksums. No reward-based selection is used.

The reference runs 64 updates; a separate process restores checkpoint 32 and continues to 64 with the same
configuration. Verification compares saved images, scores, logged gradient norms, model tensors and optimizer state,
checks frozen weights and LoRA updates, and requires exact Adam counters. Plotting needs matplotlib.
The override template enables the existing actor and rollout full-determinism options and uses serial rollout
requests (`max_num_seqs=1`) to control the continuation comparison; it is not a throughput benchmark.
Exact continuation also requires the row-order correction tracked in [#742](https://github.com/verl-project/verl-omni/pull/742)
and the separately scoped [strict actor determinism fix](https://github.com/WEIXILIN0307/verl-omni/tree/fix/diffusers-fsdp-strict-determinism).
The recorded study includes both prerequisites; the reward patch alone does not establish exact continuation.
From the repository root, with three available GPUs and immutable local model directories:

```bash
export PYTHONPATH="$PWD"
export STUDY_OUTPUT=/path/to/sd35-study
"$PYTHON" -m tests.special_e2e.reproduce_native_fsdp_sd35 --phase prepare \
  --output "$STUDY_OUTPUT" --sd35-model "$SD35_MODEL" \
  --pickscore-model "$PICKSCORE_MODEL" --processor "$PICKSCORE_PROCESSOR"
CUDA_VISIBLE_DEVICES="$GPU_TRIPLE" "$PYTHON" -m tests.special_e2e.reproduce_native_fsdp_sd35 \
  --phase reference --output "$STUDY_OUTPUT"
CUDA_VISIBLE_DEVICES="$GPU_TRIPLE" "$PYTHON" -m tests.special_e2e.reproduce_native_fsdp_sd35 \
  --phase resume --output "$STUDY_OUTPUT"
CUDA_VISIBLE_DEVICES='' "$PYTHON" -m tests.special_e2e.reproduce_native_fsdp_sd35 \
  --phase verify --output "$STUDY_OUTPUT"
"$PYTHON" -m tests.special_e2e.reproduce_native_fsdp_sd35 --phase plot --output "$STUDY_OUTPUT"
```

The override template is `tests/special_e2e/native_fsdp_sd35_overrides.json`. The complete unsmoothed training
and validation means are written to `curves.csv`; `comparison.json` records continuation checks. Validation
shading is one standard error across the fixed prompts, not an interval over independent training runs.
The trainer saves training media on a best-effort basis. Verification accepts a missing training dump only
when the log records that the previous dump was still running; all logged training means and every validation
image set are still required. The report lists inspected samples and skipped training dumps separately.
This single-seed study establishes the exercised integration behavior, not convergence or quality gains.
