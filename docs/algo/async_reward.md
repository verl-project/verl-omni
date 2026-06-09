(async_reward)=
# Async Reward for Diffusion Training

Last updated: 06/08/2026

Async reward lets VeRL-Omni score completed rollout samples through reward-loop
workers while other samples are still being generated. It is useful when reward
computation is expensive, for example when a VLM judge, OCR model, preference
model, or external HTTP scorer takes a significant fraction of the training step.

> **Status:** Supported for online diffusion training. The current
> implementation overlaps reward computation with rollout collection inside a
> training step; the policy update still waits for the full batch before
> advantage computation and actor optimization.

## Why

In a standard online FlowGRPO step, training data flows through three major
stages:

1. The rollout engine generates images or videos for each prompt.
2. The reward function scores each generated sample.
3. The trainer computes advantages and updates the actor.

If the reward model is colocated with the actor or rollout workers, reward
scoring often sits on the critical path. This is especially visible for
multimodal reward models: rollout GPUs may finish some samples early, but the
trainer cannot use those completed samples until reward computation finishes for
the whole batch.

Async reward moves reward scoring into reward-loop workers. When a rollout
sample finishes, the agent loop immediately sends that sample to a reward worker.
Other rollout samples continue running at the same time. With
`reward.reward_model.enable_resource_pool=True`, those reward workers can also
use a dedicated GPU pool, so expensive reward inference does not time-share the
same GPUs as actor training and rollout generation.

This reduces the end-to-end step time when reward latency is large enough to
hide behind the remaining rollout work.

## What async reward means

Async reward in VeRL-Omni is **sample-level streaming reward computation** within
an otherwise on-policy training step.

<img width="1380" height="1020" alt="demo" src="https://github.com/user-attachments/assets/eaa577d0-e608-4a21-b044-961a89bcc590" />



The upper panel shows the colocated/synchronous case: early rollout samples sit
idle until the slowest sample finishes, then the reward batch runs, then actor
training starts. The lower panel shows async reward: each completed sample is
streamed to a reward worker immediately. Training still starts only after the
full scored batch is ready, but the reward stage is partly hidden behind the
remaining rollout work.


The important boundary is the policy update. Async reward does **not** make the
actor update proceed on partial or stale batches. The trainer still assembles the
full rollout batch, extracts rewards, computes advantages, and then performs the
actor update. This keeps the usual on-policy FlowGRPO semantics while reducing
idle time inside the rollout/reward phase.

For fully asynchronous RL across training iterations, where rollout generation
and policy optimization run continuously with controlled policy staleness, see
systems such as Relax. VeRL-Omni's async reward feature is narrower: it overlaps
reward computation with rollout collection for the current batch.

## Quickstart

Run the async reward example:

```bash
bash examples/flowgrpo_trainer/run_qwen_image_ocr_lora_async_reward.sh
```

The example uses four GPUs for actor/rollout and one GPU for reward inference:

```bash
NUM_GPUS_ACTOR_ROLLOUT=4
NUM_GPUS_REWARD=1
ROLLOUT_TP=1
REWARD_TP=1
```

The key overrides are:

```bash
reward.num_workers=$((NUM_GPUS_REWARD / REWARD_TP)) \
reward.reward_model.enable=True \
reward.reward_model.model_path=$reward_model_name \
reward.reward_model.rollout.name=$REWARD_ENGINE \
reward.reward_model.enable_resource_pool=True \
reward.reward_model.nnodes=1 \
reward.reward_model.n_gpus_per_node=$NUM_GPUS_REWARD \
reward.reward_model.rollout.tensor_model_parallel_size=$REWARD_TP \
reward.custom_reward_function.path=$reward_function_path \
reward.custom_reward_function.name=compute_score_ocr
```

## Config reference

The most important settings live under `reward`:

| Config | Meaning |
| --- | --- |
| `reward.reward_model.enable=True` | Enables model-backed reward computation. |
| `reward.reward_model.enable_resource_pool=True` | Allocates a separate Ray resource pool for reward-model workers. This is the setting that enables reward computation to run on dedicated GPUs. |
| `reward.reward_model.n_gpus_per_node` / `reward.reward_model.nnodes` | Size of the reward-model resource pool. |
| `reward.num_workers` | Number of reward-loop workers. Usually set to `NUM_GPUS_REWARD / REWARD_TP`. |
| `reward.reward_model.rollout.tensor_model_parallel_size` | Tensor-parallel size for reward-model inference. Increase this when the reward model does not fit on one GPU. |
| `reward.custom_reward_function.path` / `name` | Reward function used by the reward manager. It may be a normal function or an `async def` coroutine. |
| `reward.reward_manager.name` / `module.path` | Optional reward manager override, for example `MultiVisualRewardManager` when combining multiple rewards. |

The base reward config documents these fields in
`verl_omni/trainer/config/reward/reward.yaml`.

## How it plugs in

Async reward is enabled by passing reward-loop worker handles into the rollout
agent loop. This happens when either there is no reward model, or when the reward
model has its own resource pool:

```python
enable_agent_reward_loop = (
    not self.use_rm or self.config.reward.reward_model.enable_resource_pool
)
reward_loop_worker_handles = (
    self.reward_loop_manager.reward_loop_workers
    if enable_agent_reward_loop
    else None
)
```

The diffusion agent loop runs one async task per rollout sample. After a sample
finishes generation, `_compute_score` builds a one-sample `DataProto` containing
the prompt, visual response, and reward metadata, then sends it to a reward-loop
worker:

```python
selected_reward_loop_worker_handle = random.choice(
    self.reward_loop_worker_handles
)
result = await selected_reward_loop_worker_handle.compute_score.remote(data)
output.reward_score = result["reward_score"]
output.extra_fields["reward_extra_info"] = result["reward_extra_info"]
```

When the rollout manager returns to the trainer, samples that were scored through
the reward loop already contain `rm_scores`. The trainer therefore skips the
colocated reward path:

```python
if self.use_rm and "rm_scores" not in batch.batch.keys():
    batch_reward = self._compute_reward_colocate(batch)
    batch = batch.union(batch_reward)
```

This is why async reward can reduce the measured `reward` section in the trainer
timer: the reward work has already been streamed during generation.

## Reward function behavior

The visual reward manager supports both coroutine and synchronous reward
functions:

- If the reward function is `async def`, it is awaited directly.
- If the reward function is synchronous, it is run through
  `loop.run_in_executor(...)` so it does not block the reward worker's event
  loop.

This makes async reward compatible with both model-backed rewards and ordinary
Python reward code.

For multiple reward functions, `MultiVisualRewardManager` loads each configured
sub-reward, detects whether it is async, and combines the weighted scores. Sub
rewards currently run sequentially inside one sample, but each individual
function can still be async and reward workers can process samples concurrently.

## External HTTP scorers

Async reward also pairs well with external HTTP scorers. The HTTP reward client
(`verl_omni.utils.reward_score.http_scorer_client`) is an `async` reward
function that sends generated images to a separate scorer service. Because reward
workers batch samples with `asyncio.gather`, requests in the batch can hit the
HTTP service concurrently rather than serially.

See {doc}`../start/http_scorer` for the service protocol and an end-to-end OCR
reward-server example.

## Performance notes

The Qwen-Image OCR LoRA benchmark in `examples/flowgrpo_trainer/README.md`
compares sync reward with async reward:

| Script | GPUs | Async reward GPUs | Time per step |
| --- | --- | --- | --- |
| `run_qwen_image_ocr_lora.sh` | 4 | 0 | 420 s |
| `run_qwen_image_ocr_lora_async_reward.sh` | 5 | 1 | 360 s |

The async setup is faster in wall-clock time because reward inference runs on a
dedicated fifth GPU and overlaps with rollout collection. Its images/GPU/s number
can be lower because the denominator includes the extra reward GPU; use both
step time and total GPU budget when comparing setups.

## When to use it

Async reward is most useful when:

- The reward model is expensive relative to rollout generation.
- The reward model causes OOM when colocated with actor/rollout workers.
- You have spare GPUs that can host a dedicated reward pool.
- You call an external scorer service and want many reward requests in flight.

It may not help when:

- The reward is a cheap rule-based function.
- Rollout generation dominates the full step and reward latency is small.
- You do not have additional GPU budget and colocated scheduling is already
  efficient.
- The reward service has low concurrency or becomes the bottleneck.

## Troubleshooting

**Reward model OOM.** Increase `REWARD_TP`, reduce `reward.num_workers`, or
allocate more reward GPUs through `reward.reward_model.n_gpus_per_node`.

**Reward workers are idle.** Check that
`reward.reward_model.enable_resource_pool=True` is set and that
`reward.num_workers > 0`.

**Rollout still waits a long time.** Async reward only overlaps reward with
unfinished rollout work. If every rollout finishes before reward starts to
dominate, the trainer still waits for reward completion before training.

**HTTP scorer is slow.** Scale the scorer service, increase its worker count, or
check network latency. The HTTP scorer client reuses `aiohttp.ClientSession` and
offloads image serialization to a thread pool, but the remote service still needs
enough capacity.

## References

- [Flow-GRPO: Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470)
  describes the online RL algorithm used by the FlowGRPO examples.
- [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256)
  describes the verl systems model behind flexible role placement and resource
  pools.
- [Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale](https://arxiv.org/abs/2604.11554)
  is related systems work on fully asynchronous RL training with explicit
  staleness control.
