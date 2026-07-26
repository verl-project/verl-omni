(rollout_batching)=
(request_level_batching)=
# Diffusion Rollout Batching

Last updated: 07/22/2026.

vLLM-Omni can batch diffusion rollouts in two different ways. They look similar
in config (`max_num_seqs`) but they are **different engines** and are
**mutually exclusive**.

| Mode | What gets batched | Config switch |
|---|---|---|
| **Step-wise continuous batching** | Denoising *steps* across in-flight requests | `step_execution=true` |
| **Request-level batching** | Whole requests in one transformer forward | `step_execution=false` (default) + `supports_request_batch=True` |
| **Serial** | Nothing | `max_num_seqs=1` (or leave engine default when unsupported) |

```{note}
Setting `step_execution=true` disables request-level packing inside the engine.
Do not enable both modes at once.
```

---

## Which mode should I use?

1. Prefer the **faster** mode for your model (benchmark gen / step time).
2. If tied or unknown and the adapter supports step-wise → use **step-wise**.
3. Else if the adapter supports request-level → use **request-level**.
4. Else → serial.

**How to tell support**

| Mode | Adapter signal |
|---|---|
| Step-wise | Implements `prepare_encode`, `denoise_step`, `step_scheduler`, `post_decode` |
| Request-level | Sets `supports_request_batch = True` and handles `DiffusionRequestBatch` in `forward` |

**Example-script defaults today**

| Recipe | Default mode |
|---|---|
| Qwen-Image FlowGRPO (`run_qwen_image_ocr*.sh`) | Step-wise (`step_execution=true`, `max_num_seqs=256`) |
| SD3.5 FlowGRPO (`run_sd35_medium_ocr_lora.sh`) | Request-level (`step_execution=false`, `max_num_seqs=256`, `wait_ms=10`) |

On Qwen-Image FlowGRPO e2e LoRA (32×16, 512²) the two modes were essentially
tied (~106–108s gen). SD3.5 currently has request-level support only.

---

## Step-wise continuous batching

### Idea

Each request advances one denoising step at a time. The scheduler can mix steps
from many requests in the same worker, so long-running generations overlap.

```text
req A: encode → step0 → step1 → … → decode
req B:        encode → step0 → step1 → …
              ↑ steps from A and B can interleave
```

### Enable

```bash
actor_rollout_ref.rollout.step_execution=true
++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=256
```

| Knob | Meaning |
|---|---|
| `step_execution` | Turns on the step lifecycle (`prepare_encode` → steps → `post_decode`). |
| `max_num_seqs` | Max concurrent requests in the step scheduler. |

### Implement for a new model

Add the step lifecycle methods to the existing rollout adapter (same
`algorithm=` registration — no separate `*_stepwise` package). See
[`integrating_a_stepwise_continuous_batching_model.md`](../contributing/integrating_a_stepwise_continuous_batching_model.md).

---

## Request-level batching

### Idea

Multiple complete requests are packed into one `DiffusionRequestBatch` and run
through a single `forward()` (one packed transformer batch per denoising step
of that fused group).

```text
req A ─┐
req B ─┼─→ DiffusionRequestBatch → forward() → split per request
req C ─┘
```

### Enable

```bash
actor_rollout_ref.rollout.step_execution=false
++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.max_num_seqs=256
++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.request_batch_max_wait_ms=10
```

| Knob | Meaning |
|---|---|
| `step_execution=false` | Required (yaml default). |
| `max_num_seqs` | Max requests packed per forward. Engine default is effectively serial (`1`) unless raised. |
| `request_batch_max_wait_ms` | Optional admission wait before the first schedule of a wave. Default `0`; `10` is enough for typical training bursts. |

Override with `MAX_NUM_SEQS` / `REQUEST_BATCH_MAX_WAIT_MS` in the example
scripts, or Hydra as above.

```{warning}
Request-level packing uses more activation memory than step-wise for the same
`max_num_seqs`. For Qwen-Image e2e LoRA + true CFG at 512², keep
`max_num_seqs≤32` on the request-level path; larger values can OOM. Step-wise
Qwen examples can use `256`.
```

### Expected performance (request-level vs serial)

Gen time excludes step 1 (warmup).

| Workload | serial (`max_num_seqs=1`) | request-level | Δ gen |
|---|---:|---:|---:|
| Qwen-Image LoRA, 32×16, 512², `max_num_seqs=32` | 226.4s | 107.9s | **−52%** |
| SD3.5 LoRA, 8×8, 384² FA3, `max_num_seqs=256` | 25.4s | 22.3s | **−12%** |

`wait_ms` has little effect on these recipes (`0` vs `50` differs by ~2% on
Qwen-Image). The gain comes from `max_num_seqs > 1`.

### Implement for a new model

1. Set `supports_request_batch = True` on the pipeline class.
2. Accept `OmniDiffusionRequest | DiffusionRequestBatch` in `forward`.
3. Collate / split with helpers in
   [`verl_omni/pipelines/request_batch.py`](../../verl_omni/pipelines/request_batch.py).

Details:
[`integrating_a_diffusion_model.md`](../contributing/integrating_a_diffusion_model.md#request-level-batching).

---

## Quick comparison

| | Step-wise | Request-level |
|---|---|---|
| Unit of batching | Denoising step | Whole request |
| Adapter API | Step lifecycle methods | Batched `forward()` |
| `step_execution` | `true` | `false` |
| Memory shape | Usually lighter concurrency | Heavier packed forwards |
| Best when | Step methods exist and match or beat request-level | Step methods missing, or request packing is faster |
