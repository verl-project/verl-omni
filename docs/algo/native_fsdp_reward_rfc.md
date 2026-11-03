# RFC: Single-node, two-GPU native FSDP2 reward replicas

Last updated: 11/02/2026.

Status: proposed. This RFC follows the [native FSDP discussion in #432](https://github.com/verl-project/verl-omni/issues/432#issuecomment-6471111363).
The configuration names, execution contract and initial support scope are proposed for maintainer review.
The associated PR provides a reference implementation and validation entry points.

## Problem and goals

Each bundle in the existing `placement.devices` configuration represents a complete model replica.
Adding devices increases the replica count but cannot shard one model across GPUs.
Wrapping two existing workers in FSDP would violate the dispatch contract: they can receive different samples
or execute different numbers of forwards, causing collective ordering to diverge.

The initial proposal adds one explicit logical replica: **a single node, a dedicated reward GPU pool,
two FSDP2 ranks, frozen FP32 PickScore, and synchronous V1 diffusion training**.
The goals are model sharding, score equivalence and correct lifecycle management.
Performance and convergence require separate evaluation.
`offload=true` unloads and reloads weights between phases; `offload=false` keeps them resident in the dedicated pool.
Both modes have been exercised in real training.

```text
existing scorer / dispatcher -> one logical replica -> rank 0 + rank 1
                                                      same batch and collective order
                            <- one raw output       <- wait for both ranks
```

## Configuration and compatibility

Apply this fragment to an existing synchronous V1 training configuration.
Also set `reward.reward_model.enable_resource_pool=true`, `nnodes=1` and `n_gpus_per_node=2`,
and select `MultiVisualRewardManager` for named rewards. Model-free rewards such as JPEG can be combined as usual.

```yaml
reward:
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
          processor_path: /path/to/PickScore_processor
      fsdp:
        timeout: 120
        collective_timeout: 30
  reward_functions:
    pickscore:
      path: pkg://verl_omni.utils.reward_score.pickscore_reward
      name: compute_score_pickscore_native
      weight: 0.7
      required: true
```

`replica_groups` explicitly enables sharding and is mutually exclusive with `devices`.
Existing flat `devices` configurations retain complete-replica semantics.
Group entries are logical bundle indices in the parent resource pool; declaration order determines rank order.
They are not physical GPU indices.
The initial implementation permits one named deployment with one two-rank group, requires positive finite timeouts,
and rejects duplicate, overlapping or out-of-range placement.
The model path must be explicit. The executor accepts only `processor_path`, so unsupported model options cannot be silently ignored.

Every term bound to this deployment must set the boolean `required: true`, making group failure fatal to training.
This reuses the manager's existing required-term semantics and preserves optional behavior for other scorers.

## Execution contract

| Boundary | Initial contract |
| --- | --- |
| Dispatch | Register one CPU scoring facade with the dispatcher. Broadcast the same preprocessed inference batch to both ranks, wait for both, and publish only rank 0's output. |
| Model adaptation | Run PickScore feature extraction inside the FSDP-managed `forward`. Represent scalar `logit_scale` as a one-element parameter inside the adapter and restore the scalar output. Preserve the checkpoint and scoring formula. |
| Loading | Load the checkpoint on CPU on rank 0 and construct the other rank on meta. Reuse pinned verl's `apply_fsdp2` and `fsdp2_load_full_state_dict`, including buffer initialization. |
| Ordering | Use a group-level synchronous lock for wake, infer, sleep and close, with monotonically increasing inference sequence numbers. Do not introduce a separate per-rank microbatcher. |
| Lifecycle | Reuse the native lifecycle owner. Sleep drains accepted scoring work before destroying both ranks' weights; wake reloads the original checkpoint. The trainer retains rank termination handles while the facade orders normal collectives. Close drains work, destroys the process group and confirms rank/facade death. |
| Cancellation | The executor retains accepted background operations. Repeated cancellation cannot let scoring outlive sleep or close. |
| Failure | Partial submission, rank exit, forward failure, facade exit or heartbeat timeout permanently invalidates the group. The trainer independently terminates the facade and all ranks and confirms their death, without GC, recovery barriers or automatic collective retries. Cleanup failure preserves the original error and fails the job. |
| Training exit | Attempt to close both the FSDP deployment and TransferQueue on every successful or failed init/fit exit. Failure to close one must not prevent attempting the other. |
| Resume | Reconstruct frozen reward weights from the original checkpoint. Actor weights, optimizer state and training steps follow the existing checkpoint path. |

The core implementation is [native_fsdp.py](../../verl_omni/reward_loop/native_fsdp.py),
integrated through the existing named deployment, dispatcher, native scorer and manager.
[#723](https://github.com/verl-project/verl-omni/issues/723) /
[#760](https://github.com/verl-project/verl-omni/pull/760) own the scorer protocol and batching.
This proposal executes an already-submitted inference batch as one complete collective unit.

The dispatcher delegates this group's RPC ownership to the existing lifecycle owner; other groups retain their current path.
`timeout` bounds rank RPC waits and facade heartbeat responses, with heartbeat intervals of at most one second.
It is not a deadline for an entire training batch. `collective_timeout` is the process-group communication deadline.
Each failure-cleanup stage waits at most `min(timeout, 10)` seconds for death confirmation and continues attempting
other actors if one stage fails. Cold loading on a shared host may require explicit larger loading/communication budgets.

## Independently reviewable changes

1. **Sharding core and contract tests:** two-GPU loading, raw-output/score parity, ordering, reload, cancellation, failures and group cleanup.
2. **Explicit placement and deployment integration:** strict configuration, parent-pool bundle mapping, one logical worker, construction rollback and existing-configuration regression coverage.
3. **Training lifecycle and end-to-end acceptance:** real rollout, actor updates, validation, checkpoints, resume and fit-failure cleanup.

The reference implementation covers these boundaries. It reuses upstream's existing engine-argument probing with a real model path.

## Acceptance and follow-up scope

The implementation was integrated on upstream `main` at `aaaea177e80dd55b6d90a3d83ef08984bafc3a46`.
The scoped CPU regression suite passes 1571 tests (32 warnings). GPU checks passed 10 rank/NCCL failure cases,
4 facade failure cases, 20 pretrained-model load/score/unload cycles and two real SD3.5 Medium FlowGRPO updates.
The maximum observed PickScore absolute error on that integration was `1.79e-7`.
The production implementation and GPU validation entry points are unchanged by the subsequent documentation and PR-scope cleanup.

Longer experiments belong to the separately identified original development baseline.
Real PickScore has 986,109,441 parameters, with 493,055,873 / 493,053,568 held by the two ranks.
Parity against the unsharded native model covered batches 1, 3, 16 and 17, reordered samples and long prompts,
with a maximum observed score absolute error of `2.38e-7`.
Existing manager training/validation inputs and PickScore/JPEG weighting at 0.7/0.3 passed.

On that original baseline, real SD3.5 Medium FlowGRPO completed 10 continuous updates at 256×256 with 8 denoising steps,
finite nonzero gradients, training/validation images and checkpoints each step, increasing optimizer counters,
and 484 changing LoRA tensors across the final two checkpoints.
Resume completed step 11 and verified optimizer progress from 10 to 11.
Resident mode and an independent Python venv each completed two training updates and confirmed resource death before shutdown.
The independent venv's related CPU suite passed 1107 tests.
See the [validation report](native_fsdp_reward_validation.md) for commands, results and the distinction between baselines.

Multiple logical replicas, multiple nodes, streaming, shared-GPU scheduling, CPU/NPU FSDP, other model adapters,
reward-weight training, throughput/peak-memory benchmarks and long-term convergence require independent implementation and acceptance.
No support or benefit is claimed for those areas here.
