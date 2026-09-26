# Named Reward Models

Last updated: 09/26/2026

This guide describes how to configure and extend named model-backed rewards
under `reward.models` in `verl-omni`. For the general Reward Loop interface and
custom reward functions, refer to the upstream `verl` documentation.

`reward.models` lets one training job use one or more independently managed
model-backed rewards. It supports an engine backend, a native backend, or both
in the same job.

The framework deliberately separates inference from scoring:

- a named model owns resources, inference access, and lifecycle;
- a reward function converts one training sample into model inputs and converts
  the model output into a score;
- `MultiVisualRewardManager` combines scores with a weighted sum.

PickScore is an example of this contract, not a special case in the framework.

Named models currently use the visual sample contract. Select the manager
explicitly; the framework does not rewrite a user-provided manager:

```yaml
reward:
  reward_manager:
    name: MultiVisualRewardManager
```

Audio and other modality-specific multi-reward managers are follow-up work.

## Native replica scheduling

Named models use static padded splitting by default: each worker receives one
equal-sized chunk. For native models whose sample costs or replica speeds vary,
set `reward.models.<name>.dispatch_batch_size` to a positive integer to enable
completion-driven dispatch:

```yaml
reward:
  models:
    pickscore:
      backend: native
      placement:
        devices: [0, 1]
      executor:
        model: verl_omni.utils.reward_score.pickscore_reward:PickScoreNativeModel
      dispatch_batch_size: 8
```

Keep the model's executor arguments and reward function configured as described
below. Each placement entry still owns one complete model replica. Each replica
receives at most one scoring RPC at a time and takes the next contiguous batch
when it finishes. The final batch may be smaller; no duplicate padding is added.
Results are restored to input order before existing per-model score aggregation.
Omit this field or set it to `null` to keep static splitting. Engine models reject
this option; their internal scheduling and parallelism remain engine-owned.

Choose a batch size large enough for efficient model batching but small enough
to leave work available for faster replicas. This controls reward-loop sample
dispatch, not the model's own inference batch size. Opt in only when samples can
be scored independently: batch-sensitive or replica-local random scorers can
change scores when batch boundaries or replica assignments change.

On a dispatched scoring failure or caller cancellation, no new batches are
submitted once the dispatcher observes it. Already submitted RPCs are drained
before the error is raised and models sleep, including across mixed engine/native
groups. There is no automatic retry, actor recovery, speculative execution,
autoscaling or streaming-trainer support. A stuck RPC can therefore delay drain;
this option does not introduce a timeout or health-check policy.

## Backend selection

| Backend | Use it when | Reward-function arguments |
| --- | --- | --- |
| `engine` | The model is supported by the existing `verl.RewardModelManager` and should be served by vLLM | `reward_router_address` and `model_name` |
| `native` | The model can be loaded and called directly in a reward worker, including ordinary Transformers models | `reward_model`, an inference handle |

The engine path documented here uses vLLM. vLLM-Omni reward serving is not
implemented; vLLM-Omni can still be used independently for actor rollout.

Existing jobs without `reward.models` continue to use the existing single-model
reward path. Do not set `reward.reward_model.enable=true` together with
`reward.models`.

## Migrate an existing engine reward

An existing single-model engine configuration has one global reward model and
one custom reward function:

```yaml
reward:
  reward_model:
    enable: true
    enable_resource_pool: true
    n_gpus_per_node: 2
    nnodes: 1
    model_path: /models/ocr
    rollout:
      name: vllm
      tensor_model_parallel_size: 2

  custom_reward_function:
    path: pkg://my_package.ocr_reward
    name: compute_score
```

To migrate it, disable the existing single model, create a named `engine` model,
and move the scoring function into `reward.reward_functions`. The model name and
reward term name can be the same:

```yaml
reward:
  reward_model:
    enable: false
    enable_resource_pool: true
    n_gpus_per_node: 2
    nnodes: 1

  models:
    ocr:
      backend: engine
      model_path: /models/ocr
      n_gpus_per_node: 2
      nnodes: 1
      rollout:
        name: vllm
        tensor_model_parallel_size: 2

  reward_functions:
    ocr:
      path: pkg://my_package.ocr_reward
      name: compute_score
      weight: 1.0
      required: true
```

Remove the old `custom_reward_function.path` override when switching to
`reward_functions`. The score function can keep the existing router-based
contract if it already accepts `reward_router_address` and `model_name`:

```python
async def compute_score(
    data_source,
    solution_image,
    ground_truth,
    extra_info,
    reward_router_address,
    model_name,
):
    response = await call_openai_compatible_server(
        address=reward_router_address,
        model=model_name,
        image=solution_image,
        prompt=ground_truth,
    )
    return {"score": parse_score(response)}
```

`reward.reward_model.rollout` remains the common engine default. Values under a
named model's `rollout` override those defaults. A named model's `model_path`
also overrides the common `reward_model.model_path` fallback.

## Model-to-reward binding

A reward term automatically uses a model with the same name:

```yaml
reward:
  models:
    quality:
      backend: native
      model_path: /models/quality
      placement:
        devices: [0]
      executor:
        model: my_package.reward_model:QualityModel

  reward_functions:
    quality:
      path: pkg://my_package.reward_score
      name: compute_quality_score
```

Set `model` explicitly when the names differ or several reward terms share one
model:

```yaml
reward_functions:
  semantic_quality:
    model: quality
    path: pkg://my_package.reward_score
    name: compute_semantic_quality
```

Every term uses the existing aggregation contract:

```text
final_reward = sum(term.weight * term.score)
```

`required=true` makes a scoring failure fatal. An optional term records the
error and contributes zero. Model setup and lifecycle failures are always
fatal.

## Use engine only

This example serves one model with two-way tensor parallelism:

```yaml
reward:
  reward_model:
    enable: false
    enable_resource_pool: true
    n_gpus_per_node: 2
    nnodes: 1

  models:
    ocr:
      backend: engine
      offload: true
      model_path: Qwen/Qwen3-VL-8B-Instruct
      n_gpus_per_node: 2
      nnodes: 1
      rollout:
        name: vllm
        tensor_model_parallel_size: 2
        data_parallel_size: 1
        pipeline_model_parallel_size: 1

  reward_functions:
    ocr:
      path: pkg://verl_omni.utils.reward_score.genrm_ocr
      name: compute_score_ocr
      weight: 1.0
```

The engine owns serving, request batching, and TP/DP/PP. The reward function
owns the request format and score calculation.

## Use native only

This example starts one complete Transformers model replica on each of two
native reward workers:

```yaml
reward:
  reward_model:
    enable: false
    enable_resource_pool: true
    n_gpus_per_node: 2
    nnodes: 1

  models:
    quality:
      backend: native
      offload: true
      model_path: /models/quality
      placement:
        devices: [0, 1]
      executor:
        model: my_package.reward_model:TransformersRewardModel
        kwargs:
          torch_dtype: bfloat16

  reward_functions:
    quality:
      path: pkg://my_package.reward_score
      name: compute_quality_score
      weight: 1.0
```

`executor.model` accepts an importable `module:Class`, a
`pkg://module:Class`, or a supported Python file path plus class name. The
framework supplies `model_path` and the worker-local `device` unless those
arguments are already present in `executor.kwargs`.

Each native model entry is one deployment. Different checkpoints or lifecycle
policies require different named deployments; multiple reward functions may
share one deployment through their `model` field. In this PR, every
`placement.devices` entry creates one complete replica of that deployment.
Future FSDP support will need an explicit replica-group schema because a flat
device list cannot distinguish full replicas from ranks within one sharded
replica.

### Wrap a Transformers model for native mode

A Transformers checkpoint does not need an inference server. Add a small model
adapter that loads the processor and model and exposes `infer()`. The adapter
owns inference only; it must not decide the final reward semantics.

```python
import torch
from transformers import AutoModel, AutoProcessor


class TransformersRewardModel:
    def __init__(
        self,
        model_path: str,
        device: torch.device,
        torch_dtype: str = "bfloat16",
    ):
        self.device = device
        dtype = getattr(torch, torch_dtype)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).eval().to(device)

    @torch.inference_mode()
    def infer(self, texts, images):
        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)
        return {"logits": outputs.logits_per_image.detach().cpu()}

    def close(self):
        del self.model
        del self.processor
```

Then add a separate score adapter. Its `reward_model` argument is the native
inference handle exposed by the framework:

```python
async def compute_quality_score(
    data_source,
    solution_image,
    ground_truth,
    extra_info,
    reward_model,
):
    del data_source, extra_info
    output = await reward_model.infer(
        texts=[ground_truth or ""],
        images=[solution_image],
    )
    return {"score": float(output["logits"][0, 0])}
```

The names and shapes passed to `infer()` are an internal contract between these
two adapters; the framework does not prescribe them. Both synchronous and
asynchronous `infer()` and `close()` implementations are accepted. Synchronous
methods run outside the reward worker's event loop. `close()` is optional; the
executor also runs garbage collection and clears the accelerator cache when a
model sleeps.

## Mix engine and native models

Engine and native models can be scored in the same job. The following dedicated
four-device parent pool is split into a two-device engine allocation and an
independent two-device native subpool:

```yaml
reward:
  reward_model:
    enable: false
    enable_resource_pool: true
    n_gpus_per_node: 4
    nnodes: 1

  models:
    ocr:
      backend: engine
      offload: true
      model_path: /models/ocr
      n_gpus_per_node: 2
      nnodes: 1
      rollout:
        name: vllm
        tensor_model_parallel_size: 2

    quality:
      backend: native
      offload: true
      model_path: /models/quality
      placement:
        devices: [2, 3]
      executor:
        model: my_package.reward_model:TransformersRewardModel

  reward_functions:
    ocr:
      path: pkg://my_package.ocr_reward
      name: compute_score
      weight: 0.4
    quality:
      path: pkg://my_package.reward_score
      name: compute_quality_score
      weight: 0.6
```

All engine allocations are carved out first in configuration order. Each
native model then receives an independent subpool at the parent-pool bundle
indices listed in `placement.devices`. These are global indices within the
trainer-selected parent resource pool, not physical CUDA/NPU IDs and not
tensor-parallel ranks. Indices may be non-contiguous, but cannot overlap another
native model or the engine allocation. Each index creates one complete native
replica in this PR; it does not identify a rank in a sharded model group.

An engine model's world size is
`replicas * TP * DP * PP`. If `n_gpus_per_node` and `nnodes` are set on that
model, their product must be at least the world size and a multiple of it. All
named allocations together must fit in the selected parent pool.

Set `reward.reward_model.enable_resource_pool=false` to split the trainer's
global parent pool instead. Set it to `true` to create the dedicated parent pool
whose size is controlled by `reward.reward_model.n_gpus_per_node` and `nnodes`.

## Lifecycle and execution

`offload` has the same meaning for both backends:

- `true` (default): wake before scoring and sleep afterward;
- `false`: keep the model resident across training steps.

Independent named models are woken, scored, and slept concurrently. Native
batches are padded and split evenly across the workers assigned to that model
by default. Native deployments can opt into completion-driven microbatch dispatch
with `dispatch_batch_size`; see [Native replica scheduling](#native-replica-scheduling).

The reward loop exposes `async_compute_rm_score()` for asynchronous callers and
keeps `compute_rm_score()` as the synchronous compatibility entrypoint used by
current trainers. Cleanup is attempted even when inference or scoring fails.

## PickScore validation recipe

The standard Qwen-Image-Edit launcher uses native PickScore. A mixed vLLM and
native parity recipe is available at
`tests/special_e2e/run_qwen_image_edit_lora_v1_npu_engine_native.sh`. It runs the
same reward through both backends for validation and is not a production
example.

Engine PickScore uses vLLM's pooling runner and `/v1/embeddings`. Its reward
function computes:

```text
PickScore = logit_scale * cosine(text_embedding, image_embedding) / 26
```

The configured `logit_scale` is already exponentiated and must not be passed
through `exp()` again.

### Replica scheduling benchmark

The opt-in two-GPU benchmark compares static splitting with dynamic microbatches
of 4 and 16 on the same pretrained PickScore replicas:

```bash
python tests/reward_loop/benchmark_replica_dispatch.py \
  --model-path /path/to/PickScore_v1 \
  --processor-path /path/to/clip_processor \
  --output /path/to/benchmark-results.json
```

It warms each configuration and records six paired rounds of 128 samples,
alternating execution order and the slow replica. Conditions include balanced
service and explicit synthetic per-sample delays; the latter demonstrate
sensitivity to imbalance, not naturally occurring model latency. Every arm
checks score parity, ordering, exact-once dispatch and bounded concurrency.

Timings cover dispatch and scoring with resident models and actor-cached images.
They exclude model loading, offloading and full image-payload transfer, and are
not end-to-end training measurements. Small microbatches can reduce batching
efficiency; retain the static default unless a representative workload benefits.

## Current limitations

- Named-model aggregation currently uses the visual reward manager contract.
- Native models are replicated; FSDP and tensor parallelism are not supported.
- CPU-native placement is not supported.
- Native routing defaults to a static even split; optional microbatch dispatch
  balances available work, without preempting or stealing an active batch.
- Named models do not participate in streaming reward computation.
- vLLM-Omni reward serving is not implemented.

Automatic migration of every existing reward implementation and a unified
streaming/FSDP design remain follow-up work. The configuration migration and
extension contracts supported by this change are documented above.
