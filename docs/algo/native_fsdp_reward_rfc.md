# RFC: Native FSDP2 reward replicas

Last updated: 11/13/2026.

## Summary

This proposal adds explicit two-rank FSDP2 replicas to the existing named native reward deployment interface,
following the [native FSDP discussion in #432](https://github.com/verl-project/verl-omni/issues/432#issuecomment-6471111363).
A deployment can contain multiple independent replicas on one node. Each replica shards one frozen FP32 model;
the existing dispatcher distributes samples across replicas, while both ranks inside a replica execute the same batch.

The core design covers synchronous V1 diffusion training, PickScore and CLIPScore adapters, dedicated CUDA/NCCL
placement and CPU/Gloo reward inference. [#855](https://github.com/verl-project/verl-omni/pull/855) introduces the
single-replica PickScore foundation. Multiple replicas, CLIPScore and CPU/Gloo extend it in separately accepted changes.
Streaming, shared-GPU phases and multi-node placement also have separate integration and acceptance boundaries below.
NPU execution and reward-parameter training require their own device and optimization contracts.

## Problem and goals

Every entry in `placement.devices` represents a complete model replica. Adding devices increases the number of
replicas but does not reduce the model parameters held by each worker. Treating these independent workers as FSDP
ranks would break collective ordering: the dispatcher can assign different samples or different numbers of forwards
to them. A rank waiting for one all-gather cannot safely execute another replica's request.

The missing abstraction is a logical replica containing cooperating shard processes. It needs one ordered command
stream and one output boundary. The deployment owner must independently retain enough information to terminate
all its processes when a scoring facade is dead or unresponsive.

The goals are explicit sharding, equivalence with existing native scores, preserved sample order, bounded failure
detection and deterministic cleanup. Throughput, peak-memory savings and convergence require workload-specific evaluation.

## Scope and compatibility

| Area | Core contract |
| --- | --- |
| Deployment | One named FSDP deployment; one or more disjoint two-rank replicas on a single node. Model-free terms may be combined with it. |
| Models | Frozen FP32 PickScore v1 and CLIPScore using CLIP checkpoints supported by their existing inferencers. |
| Accelerator placement | A dedicated single-node reward pool; two distinct logical GPU bundles per replica. |
| CPU placement | Gloo ranks on the controller's node, ordinary Ray CPU reservations, no accelerator resource pool. |
| Training integration | Synchronous V1 diffusion reward phases. CPU support concerns reward inference, not diffusion training. |
| Residency | `offload=true` destroys weights between phases and reconstructs them on wake; `offload=false` keeps them resident. |
| Compatibility | Flat accelerator `devices` still denote complete replicas; CPU `num_replicas` still counts complete replicas. |

The core rejects streaming rewards, shared accelerator pools, multiple reward nodes and multiple named deployments
when FSDP is selected. No existing setting silently enables sharding. Reduced precision, other model families,
groups larger than two ranks, elastic recovery and reward-parameter training are outside the core scope.

## Configuration

The multi-replica example requires the replica-group increment. CLIPScore and CPU examples additionally require
their corresponding increments. The foundation accepts one PickScore group on dedicated GPUs.

### Two independent GPU replicas

Apply this fragment to an existing synchronous V1 diffusion configuration:

```yaml
trainer:
  use_v1: true
  v1:
    trainer_mode: sync
reward:
  reward_model:
    enable: false
    enable_resource_pool: true
    nnodes: 1
    n_gpus_per_node: 4
  reward_manager:
    name: MultiVisualRewardManager
  models:
    pickscore:
      backend: native
      model_path: /path/to/PickScore_v1
      offload: true
      placement:
        replica_groups: [[0, 1], [2, 3]]
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
    jpeg:
      path: pkg://verl_omni.utils.reward_score.jpeg_compressibility
      name: compute_score
      weight: 0.3
```

`replica_groups` is mutually exclusive with `devices`. Each inner list contains exactly two distinct non-negative
integers. Entries must be unique across the deployment and within the parent pool's bounds. Declaration order gives
rank order: `[[3, 1], [0, 2]]` assigns the first replica's rank 0 to logical bundle 3 and rank 1 to bundle 1.
These are parent-pool bundle indices, not physical GPU indices or host-wide CUDA ordinals. The parent pool retains
ownership of placement groups; the reward implementation owns only the actors it creates in those bundles.

The number of scoring facades equals the number of replica groups. `reward.num_workers` does not create extra
FSDP ranks. Optional `dispatch_batch_size` retains completion-driven dispatch; omitting it retains static splitting.
Neither mode changes the collective batch inside a selected replica.

`model_path` must explicitly identify the frozen checkpoint. Its contents and the processor revision must remain
stable across reload and resume. FSDP adapters accept only `executor.kwargs.processor_path`; unsupported executor
options are rejected. Every term bound to the deployment must use the boolean `required: true`, so a collective
failure cannot become a zero reward and allow training to proceed.

### CLIPScore adapter

Replace the PickScore model and term under `reward` with:

```yaml
models:
  clipscore:
    backend: native
    model_path: /path/to/clip-vit-base-patch32
    offload: true
    placement:
      replica_groups: [[0, 1], [2, 3]]
    executor:
      model: verl_omni.utils.reward_score.clipscore_reward:CLIPScoreNativeModel
      kwargs:
        processor_path: /path/to/clip-vit-base-patch32
reward_functions:
  clipscore:
    path: pkg://verl_omni.utils.reward_score.clipscore_reward
    name: compute_score_clipscore_native
    required: true
```

The adapter returns raw image/text features. Score interpretation stays in `pairwise_clipscore`, preserving its
default `2.5 * max(cosine, 0)` and FP32 arithmetic. PickScore continues using its existing normalization and learned
logit scale. Managers retain named-media validation, weighting, metric names and training/validation projections.

### CPU/Gloo replicas

Keep the model, executor and scorer from the CLIPScore example. Set
`reward.reward_model.enable_resource_pool=false` and `reward.reward_model.nnodes=1`, and replace only
`reward.models.clipscore.placement` with:

```yaml
resource: cpu
replica_groups: [[0, 1], [2, 3]]
cpus_per_worker: 1
```

CPU integers are logical rank labels, not accelerator bundles or pinned CPU-core numbers. Group count determines
replica count; `num_replicas` must be omitted. Each rank reserves `cpus_per_worker` Ray CPUs and zero GPUs.
Hard node affinity keeps the ranks on the controller's node. CPU actors clear inherited Ray accelerator-visibility
overrides so a zero-GPU worker does not attempt manual accelerator setup. Existing complete CPU replicas continue
to use `resource: cpu` with `num_replicas` and no `replica_groups`.

## Architecture and execution

Flat placement treats an omitted or null `fsdp` as disabled. Replica-group placement supplies the same default
deadlines for omitted or null `fsdp` through mapping and typed dataclass construction. Explicit deadlines remain
subject to finite, positive-value validation.

```text
named reward terms -> existing dispatcher -> facade A -> rank A0 + rank A1
                                         -> facade B -> rank B0 + rank B1
                      ordered results <- rank-zero raw outputs from each replica

deployment owner retains every facade/rank handle and owns forced termination
```

### Dispatch and scoring boundary

Each group has its own verl `RayWorkerGroup`, process group, scoring facade, operation lock and inference sequence.
Groups do not participate in each other's collectives. The existing dispatcher splits work across facades and
restores sample order, including odd batches, static padding and completion-driven chunks.

Inside a replica, the facade preprocesses the complete inference request before collective admission. Both ranks
receive identical image and text tensors and execute the same FSDP-managed forward. Requests carry a monotonically
increasing sequence checked by each rank. The controller waits for both replies, requires one rank-zero result and
one empty rank-one reply, then exposes per-sample raw outputs through the native executor interface.
Scorers remain responsible for converting those outputs into rewards.

[#723](https://github.com/verl-project/verl-omni/issues/723) and
[#760](https://github.com/verl-project/verl-omni/pull/760) own scorer protocol and executor batching work.
This execution layer consumes an already-submitted inference batch. It adds no competing scorer registry or
rank-local microbatch queue. Preprocessing failures before admission do not submit a partial collective.

### Model adaptation and loading

`PickScoreForward` places CLIP feature methods behind a regular `forward` so FSDP hooks surround the computation.
Its frozen scalar `logit_scale` becomes a length-one parameter because FSDP2 rejects scalar parameters; the returned
value is restored to scalar shape. `CLIPScoreForward` reuses this boundary and returns only image/text embeddings.
Real OpenAI CLIP checkpoints expose transposed projection parameters that FSDP2 rejects as non-contiguous.
The CLIPScore adapter materializes only those frozen non-contiguous parameters, preserving values, names and shapes.

Rank 0 loads the checkpoint on CPU; rank 1 constructs the architecture on meta. Both reuse pinned verl's `apply_fsdp2`
and CLIP encoder-layer wrapping. CUDA loading uses `fsdp2_load_full_state_dict`. The CPU path uses PyTorch's public
`set_model_state_dict` with rank-zero broadcast because the pinned verl loader materializes on an accelerator.
CPU buffer broadcast also initializes non-persistent buffers on the meta-created rank.

Parameters are frozen and forwards run under `torch.no_grad()`. After forward, the root is explicitly resharded,
and accelerator completion is synchronized before acknowledgement. No reward optimizer or backward pass is created.

### Memory and design tradeoffs

FSDP shards persistent parameters; it still gathers weights for forward, retains replicated buffers and allocates
activations and communication workspace. Both ranks execute the same batch. This is parameter sharding, not tensor
parallel computation or a guarantee that any checkpoint will fit. Rank 0 also needs host memory for the full checkpoint
and loading state. Acceptance must measure the actual loading, gathered-forward and post-unload memory peaks.

Reloading on every phase avoids maintaining mutable reward state but adds checkpoint I/O and construction cost.
Dedicated residency avoids that repeated loading cost while retaining its GPU footprint. Neither choice establishes
an end-to-end speedup. Additional logical replicas duplicate the frozen model across groups and trade resources
for independently dispatched work.

Reusing verl's worker groups and FSDP2 utilities preserves its placement, wrapping and checkpoint contracts.
Wrapping independently dispatched existing workers would violate collective order; implementing custom tensor
partitioning would introduce another checkpoint/loading contract. A separate inference service or scorer registry
is unnecessary for these concrete adapters. Deployment-wide invalidation preserves required-term batch semantics
instead of continuing with a changing set of healthy replicas after failure.

### Lifecycle

| State / operation | Required behavior |
| --- | --- |
| Construct | Validate placement/executor before allocation; retain every rank/facade handle immediately, before readiness. Confirm termination of all created actors on failure. |
| Asleep → wake | Load and shard both ranks before acknowledging readiness. Repeated resident wake is idempotent. |
| Awake → infer | Hold the replica lock through preprocessing, submission and both replies. Reject inference while asleep or closed. |
| Awake → sleep | Wait behind accepted inference, discard weights on every rank and acknowledge the unloaded state. Retain process groups for wake. |
| Close | Drain ordered work, unload weights, destroy process groups, terminate owned actors and confirm death. Close is idempotent after successful termination. |
| Invalidated | Stop new deployment RPCs and terminate every group. Never reuse a failed process group. Retain cleanup failures for later close calls. |

The deployment owner records each facade before submitting its readiness RPC. Readiness acknowledges construction;
it does not transfer ownership. Constructor errors, readiness submission errors and readiness deadlines use the same
bounded owner termination path as runtime failures. Facade cleanup failure still permits every rank cleanup attempt;
construction and cleanup errors remain observable together, including on a later close.

`offload=true` reconstructs from the original checkpoint on the next wake; it does not retain a mutable CPU copy
of reward weights. `offload=false` suppresses phase unload in a dedicated pool; explicit final close still destroys
the deployment. Inference sequence numbers persist across sleep/wake within the same replica.

The executor retains accepted thread/Ray work through caller cancellation using the existing ownership helper.
Repeated cancellation must not allow a forward to outlive sleep or close. Cancellation does not roll back a
partially executed collective or release the lifecycle lock early.

### Failure ownership and deadlines

The facade orders normal collectives. The deployment owner independently holds all facade and rank handles;
it can stop submissions and kill ranks even if the facade cannot execute cleanup. A rank exit, forward failure,
partial submission, sequence mismatch, collective timeout, facade exit or unresponsive facade invalidates the
deployment. Losing one replica terminates the others rather than returning a partial reward batch.

`fsdp.timeout` defaults to 120 seconds and bounds each rank RPC wait and facade health-response wait.
Facade health probes run at intervals of at most one second while its RPC is pending. It is not a deadline for
the entire training batch. `fsdp.collective_timeout` defaults to 30 seconds and sets the process-group communication
deadline. Both values must be finite and positive; cold loading may require explicit larger budgets.

Forced cleanup fences submissions by terminating facades before ranks. Each termination stage uses a shared
death-confirmation deadline of at most `min(timeout, 10)` seconds and attempts every owned actor even after errors.
Actor death must be confirmed before successful shutdown; garbage collection and `ray.shutdown()` do not establish
successful deployment cleanup. Primary and cleanup errors are preserved together. Failed death confirmation
requires job termination; there is no collective retry, recovery barrier or automatic replacement replica.

Only a definite actor-death response confirms successful RPC termination. Temporary actor unavailability and
probe timeouts remain cleanup failures. Closing admission does not establish completed termination: concurrent
close callers wait for that operation and receive its retained error. Cancelling a close caller does not release
ownership before cleanup settles; waiting on the termination lock runs outside the event loop.

### Synchronous trainer and resume

The V1 trainer uses its existing reward phase to wake, score and sleep the named deployment. Both successful and
exceptional initialization/fit exits attempt deployment close and TransferQueue close. Failure in one cleanup
must not prevent attempting the other. Required reward failure must abort before actor-update entry.

The existing checkpoint manager owns actor weights, optimizer state and training-step continuation. Frozen reward
weights are reconstructed from the immutable checkpoint after resume; no reward optimizer state is added to actor
checkpoints. Changing reward weights or processor revision across resume changes the training objective and requires
explicit experiment configuration.

## Independent integrations

### Resident streaming

Integrate into the existing streaming lifecycle in [#730](https://github.com/verl-project/verl-omni/pull/730).
Require a dedicated reward pool and `offload=false`; do not add per-sample wake/sleep. Reuse the established async
reward client for admission and pending-work ownership. Admission closes before accepted work drains, and accepted
collectives settle before replica termination. Independent supervision must detect rank/facade failure even when
direct agent-to-facade requests bypass the controller's normal scoring RPC.

`max_inflight` bounds each `AsyncRewardClient`, not a deployment-wide semaphore. With several agent workers,
acceptance must account for all clients and measure aggregate pending work. One client's peak of two requests
establishes only that client's bound. Request/inference overlap alone establishes neither concurrent GPU kernels
nor a throughput improvement.

### Shared-GPU phases

Use explicit synchronous ownership transitions:

```text
rollout sleep ACK -> reward wake -> scoring completes -> all reward ranks unload ACK
                  -> actor update -> rollout weight synchronization
```

Actor parameters and optimizer state must be offloaded before reward wake; reference execution requires its own
offload contract. Streaming and resident reward weights are incompatible with this phase schedule. Every required
reward must settle and unload before actor-update entry. Scoring or unload failure must abort the update, preserve
the latest completed checkpoint and terminate every reward group. Ray reservations do not establish that the
model/optimizer memory fits; real training must complete on the target hardware.

### Multi-node, NPU and trainable rewards

The first multi-node increment places independent two-rank replicas within each equal-capacity node. It must define
global-bundle to node-local placement, per-group rendezvous and failure ownership across hosts. It does not shard
one replica across nodes. Acceptance requires two physical hosts, actual GPU placement and node/network failures;
four GPUs on one host and connectivity probes cannot substitute for it.

NPU support requires actual device, HCCL, FSDP and checkpoint-loading validation. CPU/Gloo results do not establish
NPU compatibility. Trainable reward parameters require a defined objective, data and loss before implementation,
then optimizer ownership, score-version consistency, sleep/wake persistence and checkpoint/resume acceptance.
Frozen inference provides none of those training guarantees.

## Acceptance criteria and evidence

The [validation report](native_fsdp_reward_validation.md) records reproduction commands and separates core results
from independent integrations. Each claimed configuration requires:

1. Real pretrained raw-output and score parity against the unsharded inferencer, including batches 1/3/16/17,
   odd/reordered/repeated samples, truncation, training/validation projections and configured score weighting.
2. Real DTensor sharding: every parameter is sharded, each pair's local element counts sum to the full model, and
   physical device identities match declared logical bundles. Multiple groups retain independent process groups.
3. Repeated wake/infer/sleep cycles, stable post-unload allocations, resident behavior and drain through cancellation,
   sleep and close. Unsupported configuration fails before resource allocation.
4. Actual rank/facade exits, unresponsive processes, constructor/forward failures, missing/corrupt weights and real
   collective timeout. Failed second-group construction rolls back an already-created first group. No partial result,
   actor-update entry or owned actor survives a fatal reward failure.
5. Real rollout/actor optimization, validation media, checkpoints and resume. Inspect finite model/optimizer tensors
   per rank, changed nonempty trainable shards, unchanged frozen tensors, exact expected optimizer-step progression
   and finite nonzero gradients. A successful process exit alone is insufficient.

Completed single-node checks include real CPU/Gloo and CUDA/NCCL PickScore/CLIPScore inference, two independent
replica groups, 20-cycle reloads and actual rank/facade failures. Four-GPU CLIPScore reached maximum raw-feature
absolute error `3.814697265625e-6` and score error `2.980232238769531e-7`; four-GPU PickScore reached
`5.900859832763672e-6` and `1.1920928955078125e-7` respectively. These are observed errors, not new scorer tolerances.

Construction rollback additionally passed actual constructor/readiness errors and hangs with one and two groups;
the two-group cases confirmed all six owned actors DEAD. The corrected single-replica candidate completed dedicated
SD3.5 Medium training and checkpoint 2→3 continuation at 256×256/eight denoising steps, with finite optimizer/model
tensors, changed LoRA tensors, unchanged frozen tensors and exact Adam step progression.

Separate shared-phase and streaming integrations completed short real SD3.5 Medium FlowGRPO training/resume runs
at 256×256 with eight denoising steps. GPU streaming evidence covers one two-rank group and one agent worker.
Shared four-GPU normal training covers two groups, FSDP2 actors and KL disabled. These runs establish functional
updates and continuation for those configurations; they do not establish general release readiness.

Remaining gates include GPU streaming with two groups and multiple agent
workers; fatal two-group shared-training failures; reference/KL or FSDP1 actor acceptance if claimed; physical
multi-node and NPU execution; and longer runs at the intended production resolution/denoising workload.
Final execution evidence must identify the complete candidate, helper scripts, runtime, configuration, model and
data revisions. Historical results remain associated with the sources actually executed.

## Reviewable increments

| Increment | Acceptance boundary |
| --- | --- |
| Two-rank core | Explicit placement, identical-rank execution, scorer parity, lifecycle and controller-owned termination. |
| Single-node replica groups | Reuse dispatch; preserve ordering; roll back partial construction; terminate all groups on failure. |
| CLIPScore | Reuse inferencer/scorer; verify real non-contiguous projections and independent feature parity. |
| CPU/Gloo | Reuse rank execution with CPU placement/loading; prove zero GPU reservations, sharding and cleanup. |
| Streaming | Integrate with #730; accept residency, per-client admission, draining and independent supervision. |
| Shared phases | Accept acknowledged memory ownership transitions and no update after scoring/unload failure. |
| Multi-node | Accept node-local replica placement and failures on two physical hosts. |
| NPU / trainable rewards | Separate device-validation and optimization/checkpoint projects. |

Each increment has its own implementation diff and acceptance decision. The single-replica foundation is sufficient
to review #855; multiple replicas, CLIPScore and CPU/Gloo are follow-up changes with explicit dependencies.
Streaming and shared phases integrate independently. Multi-node, NPU and reward-parameter training remain separate projects.
