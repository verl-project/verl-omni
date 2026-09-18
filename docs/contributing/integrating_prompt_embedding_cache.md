# Integrating the Diffusion Prompt-Embedding Cache

Last updated: 08/13/2026.

This guide defines the contract for making a diffusion rollout pipeline
compatible with vLLM-Omni's prompt-embedding cache. The cache wraps
`pipeline.encode_prompt()` and reuses its inference outputs across requests
with the same cache key. It is disabled by default.

The vLLM-Omni implementation and the authoritative cache-key rules are in
[vllm-project/vllm-omni#2962](https://github.com/vllm-project/vllm-omni/pull/2962).
Read that implementation before introducing a new `encode_prompt()` signature
or forwarding a new argument through an existing one.

## Primary Correctness Requirement

Prompt caching must not change the semantic output of `encode_prompt()`.
Whether the wrapper is disabled, takes a miss, or takes a hit, the pipeline must
receive the same inference result that the pre-cache implementation produced.
For prompt embeddings and masks this includes:

- numerical values;
- device;
- dtype;
- batch shape;
- maximum-sequence-length truncation behavior;
- `num_images_per_prompt` or `num_videos_per_prompt` expansion behavior.

The cache stores detached outputs, so downstream rollout code must treat prompt
embeddings and masks as read-only. Do not add in-place mutations after
`encode_prompt()`. A miss executes the underlying method and stores its output;
a hit returns that stored value. The observable inference values and shapes must
remain identical in both cases.

Explicit Tensor inputs remain supported for compatibility, but they bypass the
cache. Precomputed prompt embeddings also retain their existing path and must
not be converted back to token IDs just to make a cache key.

## Cache-Key Contract

vLLM-Omni binds the actual positional and keyword arguments to the
`encode_prompt()` signature, applies defaults, and builds a key from every
bound argument. A call is cacheable only when every bound argument is safe to
serialize as a key and no precomputed embedding argument is supplied.

Supported key values are recursively composed from:

- `None`, `str`, `int`, `float`, `bool`, and `bytes`;
- `torch.device` and `torch.dtype`;
- NumPy scalar values;
- `list` and `tuple` containing only supported values;
- `dict` whose values contain only supported values.

`torch.Tensor`, PIL images, NumPy arrays, custom objects, and a list, tuple, or
dictionary containing any of them are not cacheable. If any bound
`encode_prompt()` argument is not cacheable, vLLM-Omni bypasses the cache for
the entire call rather than building a partial key. Therefore, a list is not
the only supported input type, but token IDs and masks should normally be kept
as lists at the `encode_prompt()` boundary instead of tensors.

Calls with a non-`None` `prompt_embeds`, `negative_prompt_embeds`,
`pooled_prompt_embeds`, `negative_pooled_prompt_embeds`, `prompt_embeds_mask`,
or `negative_prompt_embeds_mask` also bypass the cache. They are already
precomputed-output paths and caching them would retain large tensor values
without avoiding text-encoder work.

All arguments that can affect text encoding must be included in the
`encode_prompt()` call and be considered when auditing key eligibility. This
includes, where the pipeline uses them:

- positive prompt IDs and masks;
- negative prompt IDs and masks;
- maximum sequence length;
- output multiplicity such as `num_images_per_prompt` or
  `num_videos_per_prompt`;
- every other non-default argument that changes the text-encoding result.

Do not drop a semantic argument merely to make a call cacheable. Instead,
represent it with a supported value type or allow that call to bypass the cache.

## Required Pipeline Changes

Do not update only one visible `encode_prompt()` call. Trace every path that
supplies IDs, masks, or text to the text encoder.

1. **Request batching:** For token-ID adapters, use the list-preserving path in
   [`request_batch.py`](../../verl_omni/pipelines/request_batch.py) before
   cacheable calls only. When the cache is enabled,
   `collate_prompt_rows(..., preserve_lists=True)` and
   `collate_prompt_mask(..., preserve_lists=True)` batch and pad list values
   without converting them to tensors before the cache wrapper. When it is
   disabled or not installed, preserve the original tensor collation path.
2. **Adapter boundary:** Make `forward()` and, if implemented,
   `prepare_encode()` accept the forms the cacheable path can produce: flat
   lists, batched lists, and existing Tensor inputs. Compute batch size without
   forcing a list through early tensorization.
3. **`encode_prompt()` implementation:** Accept the supported list forms and
   convert IDs and masks to tensors on `self.device` inside `encode_prompt()`,
   after the wrapper has built or looked up the key. The conversion must
   canonicalize list and Tensor inputs to the same tensors used by the
   pre-cache implementation.
4. **Raw-text fallback:** When a step-execution path tokenizes raw text, return
   token lists while `self._prompt_embed_cache.enabled` is true. Preserve its
   existing Tensor tokenizer/device path when the cache is disabled.
5. **Special cases:** Preserve precomputed embedding bypasses. For an empty
   negative prompt, construct negative IDs and masks in the same compatible
   representation as the positive prompt rather than introducing an early
   Tensor only on the cacheable path.

The cache configuration must be exposed through
`DiffusionRolloutConfig` and forwarded by
[`vllm_omni_async_server.py`](../../verl_omni/workers/rollout/vllm_rollout/vllm_omni_async_server.py)
to `AsyncOmni`:

```yaml
actor_rollout_ref:
  rollout:
    enable_prompt_embed_cache: true
    prompt_embed_cache_size: 32
```

`prompt_embed_cache_size` is per diffusion worker, must be positive, and uses
least-recently-used eviction.

## Current Reference Paths

The current integration audits every token-ID rollout path that reaches
`encode_prompt()`:

| Pipeline | Required cache-compatible paths |
| --- | --- |
| Qwen-Image FlowGRPO | [`qwen_image_flow_grpo/vllm_omni_rollout_adapter.py`](../../verl_omni/pipelines/qwen_image_flow_grpo/vllm_omni_rollout_adapter.py): request-batch collation, normal `forward()`, raw-text `_tokenize_text_prompt()`, and step-execution `prepare_encode()`. |
| Qwen-Image MixGRPO | Reuses the Qwen-Image FlowGRPO rollout adapter and inherits the same changes. |
| Qwen-Image DiffusionNFT | [`qwen_image_diffusion_nft/vllm_omni_rollout_adapter.py`](../../verl_omni/pipelines/qwen_image_diffusion_nft/vllm_omni_rollout_adapter.py): step-execution `prepare_encode()` and `_tokenize_step_prompt()`. List-to-Tensor conversion is owned by the shared [`QwenImageTokenIdPromptMixin`](../../verl_omni/pipelines/qwen_image_flow_grpo/common.py). |
| Qwen-Image online DPO | [`qwen_image_dpo/vllm_omni_rollout_adapter.py`](../../verl_omni/pipelines/qwen_image_dpo/vllm_omni_rollout_adapter.py): normal `forward()` retains compatible IDs and masks through each `encode_prompt()` call. |

Use this table as an audit checklist for a new model. A model can omit paths it
does not implement, but every text-encoding path it does implement must obey
the same input-key and output-parity contracts.

## Routing Affinity for Repeated Rollouts

Cache entries are private to each rollout replica. The trainer expands each
input prompt into `n` samples with `DataProto.repeat(..., interleave=True)`, so
the `n` samples keep the same source `uid`.

With affinity disabled, `DiffusionSingleTurnAgentLoop` uses a fresh
`uuid4().hex` request ID for every sample. This is exactly the pre-affinity
behavior. Because every ID is new, the global load balancer has no sticky entry
and assigns each sample independently to the replica with the fewest in-flight
requests. The `n` samples of one prompt can therefore be spread across
replicas, each with a separate prompt cache.

Enable affinity only for a measured repeated-rollout workload:

```yaml
actor_rollout_ref:
  rollout:
    enable_prompt_embed_cache: true
    prompt_embed_cache_size: 128
    enable_prompt_embed_cache_routing_affinity: true
```

When both options are enabled, the diffusion agent loop instead uses the
source prompt's stable `uid` as the request ID. The first sample of each prompt
group is still assigned to the currently least-loaded replica. The load
balancer records that `uid -> replica` mapping, so the remaining `n - 1`
samples for that prompt group return to the same replica and reuse its local
entry. With `n=16` and an otherwise empty cache, the ideal bound is one miss
and fifteen hits per prompt on that replica.

This changes scheduling granularity, not the global load-balancing objective:
with `x` prompts and `n` samples per prompt, different prompt groups are still
assigned approximately evenly across replicas, while all `n` samples of a
single prompt remain together. It is not a hash-based fixed assignment and it
does not send every prompt to one predetermined replica.

Affinity changes only request routing; it must not affect token IDs, prompt
embeddings, sampling parameters, or model weights. It is disabled by default
and has no effect unless prompt caching is enabled.

Do not enable affinity solely to increase hit rate. Its possible cost is
group-level rather than sample-level balancing: a small number of prompts,
large `n`, or materially different per-prompt generation costs can leave one
replica with a heavier prompt group while others finish sooner. The same risk
applies if the same global load balancer is used to route AR reward-model
requests: a prompt group's AR reward work is kept together instead of being
sample-balanced. This option changes only requests through
`DiffusionSingleTurnAgentLoop`; it does not by itself alter independently
managed reward workers. Measure per-replica prompt-group count, queue time, and
throughput in the deployment that will use affinity, and leave it disabled when
the group-level tail cost exceeds the cache benefit.

## Step Execution

Prompt caching and step execution are compatible. For a new request,
step-execution `prepare_encode()` must call the wrapped `encode_prompt()` with
cacheable inputs. Later denoising steps reuse request-local prompt embeddings
and do not re-encode the prompt. The cache applies across requests handled by
the same worker.

```yaml
actor_rollout_ref:
  rollout:
    step_execution: true
    enable_prompt_embed_cache: true
    prompt_embed_cache_size: 32
```

## Validation Checklist

Add CPU coverage for all supported pipeline paths. At minimum, verify:

- Tensor, flat-list, and batched-list token IDs and masks where supported;
- normal `forward()`, step execution, and raw-text fallback paths;
- positive and negative prompts, empty negative prompts, and precomputed
  embedding bypasses;
- cache-disabled wrapper execution, cache miss, and cache hit;
- exact post-`encode_prompt()` output parity against the pre-cache behavior and
  exact miss-versus-hit parity for values, device, dtype, masks, and shapes.

Use `rtol=0, atol=0` when outputs are expected to be identical. For performance
work, inspect per-worker cache `hits`, `misses`, and `bypassed` counts; do not
infer cache use solely from end-to-end generation timing.

## Performance Scope

The cache helps only when the same cacheable text-encoding call reaches the
same worker again. Unique prompts, short prompts, large replica counts without
affinity, and work dominated by denoising or reward evaluation can make the
end-to-end gain small. Treat cache support as a compatibility feature until a
target workload demonstrates a material benefit.
