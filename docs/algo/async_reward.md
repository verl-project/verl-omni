(async_reward)=
# Async Reward for Diffusion Training

Last updated: 06/09/2026

Async reward lets VeRL-Omni score completed rollout samples through reward-loop
workers while other samples are still being generated. It is useful when reward
computation is expensive, for example, when a VLM judge, OCR model, preference
model, or external HTTP scorer takes a significant fraction of the training step.

## Motivation

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

<img width="1367" height="1020" alt="image" src="https://github.com/user-attachments/assets/eeeafc07-a11f-47d1-9ba6-f03ae032a9c5" />

The upper panel shows the synchronous reward case: rollout workers can continue
generating later samples, but reward scoring starts only after the full rollout
batch is ready. The lower panel shows async reward: each completed sample is
streamed to a reward worker immediately, while rollout workers continue on later
samples. Training still starts only after the full scored batch is ready, but
the reward stage is partly hidden behind the remaining rollout work.


The important boundary is the policy update. Async reward does **not** make the
actor update proceed on partial or stale batches. The trainer still assembles the
full rollout batch, extracts rewards, computes advantages, and then performs the
actor update. This keeps the usual on-policy FlowGRPO semantics while reducing
idle time inside the rollout/reward phase.

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

## External HTTP scorers

Async reward also pairs well with external HTTP scorers. The HTTP reward client
(`verl_omni.utils.reward_score.http_scorer_client`) is an `async` reward
function that sends generated images to a separate scorer service. Because reward
workers batch samples with `asyncio.gather`, requests in the batch can hit the
HTTP service concurrently rather than serially.

See {doc}`../start/http_scorer` for the service protocol and an end-to-end OCR
reward-server example.

## References

- [Flow-GRPO: Training Flow Matching Models via Online RL](https://arxiv.org/abs/2505.05470)
  describes the online RL algorithm used by the FlowGRPO examples.
- [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256)
  describes the verl systems model behind flexible role placement and resource
  pools.
