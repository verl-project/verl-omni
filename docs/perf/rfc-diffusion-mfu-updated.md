Last updated: 06/01/2026

## Summary

This issue tracks an RFC proposing a **diffusion-aware MFU counter** for verl-omni so that `actor/mfu`, `actor_infer/mfu`, and `ref/mfu` (where applicable) are reported for diffusion RL runs the same way they are for LLM RL runs upstream in [verl](https://github.com/verl-project/verl).

The current `verl-omni` `TrainingWorker` silently sets `self.flops_counter = None` for any `model_type` of `diffusion_model` / `diffusion_dp_model` because the upstream `verl.utils.flops_counter.FlopsCounter` only understands HuggingFace LLM `PretrainedConfig` schemas, not `diffusers.ConfigMixin`.

Looking for feedback on:

1. Is the architecture-agnostic split (class-based registry per pipeline (`DiffusionModelFlops`) + `get_latent_seqlens` / `resolve_cfg_passes` helpers) the right factoring for adding Wan2.2 / Z-Image / SD3 / Flux later?
2. Are the FLOPs-formula conventions (factor 6 for fwd+bwd dense, factor 12 for non-causal attention, `/3` divisor for `forward_only`) the right match to upstream's LLM counter?
3. Reviewers familiar with FlowGRPO / DPO / MixGRPO: does the `num_timesteps = data["all_timesteps"].shape[1]` (with `1` fallback for DPO) cover the algorithms in scope?

A prototype implementation lives on `feat/diffusion-mfu` (4× L20 smoke test results are in §6); a PR will follow once the design is settled.

> Note: this RFC was drafted with AI assistance. A human submitter has reviewed every line and run the test commands listed in §6.

---

## 1. Motivation

Upstream [`verl`](https://github.com/verl-project/verl) reports Model FLOPs Utilization (MFU) for the actor under three keys:

| key                       | what is measured                              |
| ------------------------- | --------------------------------------------- |
| `perf/mfu/actor`          | actor training step (forward + backward)      |
| `perf/mfu/actor_infer`    | actor log-prob recompute (forward only)       |
| `perf/mfu/ref`            | reference policy log-prob (forward only)      |

MFU is the only metric that lets users compare hardware/compiler efficiency across runs independent of model size, batch size, or sequence length.

In `verl-omni` the diffusion training stack (Qwen-Image FlowGRPO / MixGRPO / DanceGRPO / SD3-DPO / DiffusionNFT) currently **does not** emit any MFU metric. The relevant block in `verl_omni/workers/engine_workers.py` reads:

```python
if hasattr(self.model_config, "hf_config"):
    self.flops_counter = FlopsCounter(self.model_config.hf_config)
else:
    # for Diffusion models, FlopsCounter is not supported yet.
    self.flops_counter = None
```

i.e. the moment `model_type` is `diffusion_model` / `diffusion_dp_model` MFU is silently dropped because `verl.utils.flops_counter.FlopsCounter` only understands HuggingFace LLM `PretrainedConfig` schemas (`hidden_size`, `num_hidden_layers`, `intermediate_size`, …).

This RFC proposes a **diffusion-aware MFU counter** that hooks into the same `_postprocess_output` path the LLM trainer already uses, so users get the familiar `actor/mfu`, `actor_infer/mfu`, `ref/mfu` keys for diffusion RL runs as well.

## 2. Why upstream `FlopsCounter` cannot be reused as-is

| Aspect              | LLM (`FlopsCounter`)                                  | Diffusion DiTs (Qwen-Image, SD3, Flux, Wan2.2, Z-Image, …)                  |
| ------------------- | ----------------------------------------------------- | --------------------------------------------------------------------------- |
| Config schema       | `transformers.PretrainedConfig`                       | `diffusers.ConfigMixin` (FrozenDict). Field names diverge across families: `num_layers` vs `n_layers`, `attention_head_dim` vs `dim_head`, `ffn_dim` vs implicit `4·dim`. |
| Token notion        | one causal stream (`batch_seqlens: list[int]`)         | image-latent / video-latent stream + (optional) text-encoder stream; some models concatenate them, some keep them separate. |
| Attention topology  | causal — `seqlen²` halved (factor 6).                  | (a) joint full attention over `img_s + txt_s` (Qwen-Image, SD3, Flux), (b) self-attn on image only + cross-attn to text (Wan2.2, PixArt), (c) single-stream concat (Z-Image), all non-causal (factor 12). |
| Forwards per step   | 1.                                                    | 1 (unconditional / class-cond / guidance-distilled like Flux) or 2 (True-CFG, standard CFG). |
| Calls per `train_batch` | 1 (one optimizer step).                           | `num_inference_steps` denoising steps (typ. 5–50).                         |
| Output              | logits over vocab.                                    | latent noise prediction (`patch² * out_channels` for image, `prod(patch) * out_channels` for video). |
| Latent layout       | n/a.                                                  | image: `(B, C, H, W)` 4-D or `(B, L, C)` 3-D; video: `(B, C, T, H, W)` 5-D; FlowGRPO-style adds an extra `T_steps` axis. |
| Sparsity            | MoE handled by per-arch estimator.                    | MoE variants (Wan2.2-MoE, future DiT-MoE) handled by per-arch estimator using `num_experts` / `num_experts_per_tok`. |

Forcing the LLM counter onto diffusion configs would either crash on missing fields or silently underestimate FLOPs by ~3–5× (it would miss every joint- attention layer, the per-step iteration over the denoising schedule, and the CFG-driven extra forward pass).

## 3. Design

### 3.1 Mirror the upstream pattern

Add a single new utility:

```
verl_omni/utils/diffusion_flops_counter.py
```

with the same shape as `verl.utils.flops_counter`:

- A registry of architecture → `DiffusionModelFlops` subclass, keyed by the value stored in `DiffusionModelConfig.architecture` (auto-detected from `model_index.json`'s `_class_name`, e.g. `"QwenImagePipeline"`).
- A `DiffusionFlopsCounter` class with `estimate_flops(latent_seqlens, prompt_seqlens, delta_time, num_timesteps, cfg_passes) -> (estimated_tflops, promised_tflops)`.
- Reuses the existing device → peak-FLOPS table from `verl.utils.flops_counter.get_device_flops`, with a new `VERL_OMNI_DEVICE_FLOPS_TFLOPS` env var to override mis-detected device peaks (e.g. H200s reporting as L20X).

The counter is **stateless** apart from the architecture name and the diffusers transformer config dict. It runs pure-CPU after the metadata all-gather, on every rank — matching the upstream LLM counter (the computation is cheap and the inputs are already replicated, so no rank-0 gating is needed).

### 3.2 FLOPs formula (Qwen-Image reference implementation)

Define
- `dim = num_attention_heads * attention_head_dim` (`inner_dim`)
- `L = num_layers`
- `H = num_attention_heads`, `d = attention_head_dim`
- per-sample joint seq length `s_i = img_s_i + txt_s_i`
- `img_tot = Σ_i img_s_i`, `txt_tot = Σ_i txt_s_i`, `B = batch_size`

Two multiplier conventions are inherited verbatim from upstream verl's LLM counter and used throughout:

- **Factor 6 for dense linears (fwd+bwd).** Each MAC is 2 FLOPs (one multiply + one add); backward computes both `∂L/∂x` and `∂L/∂w`, which is 2× the forward cost, so fwd+bwd is 3× forward. Combined: `2 × 3 = 6 · params · tokens`. Matches `_estimate_qwen2_flops` upstream.
- **Factor 12 for non-causal attention (fwd+bwd).** Per layer per head, attention runs two matmuls (`Q·Kᵀ` and `softmax(·)·V`), each costing `2·s²·d` (no causal-mask halving), so forward attention is `4·s²·d·H·L`. Times 3 for fwd+bwd gives `12·s²·d·H·L`. Matches `_estimate_qwen3_vit_flop` (the ViT path); the LLM path uses `6·` instead because of the causal-mask halving.

The MLP hidden size is assumed to be `4·dim` (Qwen-Image's `intermediate_size`); the per-block `8·dim²` parameter term below counts the two `dim ↔ 4·dim` linears under that assumption. Architectures with a different expansion ratio override this in their own estimator (§3.6).

One denoising-step forward+backward then costs:

```
# Per-block token-scaling dense linears (asymmetric routing: image tokens
# only flow through the image stream and text tokens only through the text
# stream, so each stream contributes 3·dim² (qkv) + dim² (out) + 8·dim²
# (gelu MLP, hidden = 4·dim) = 12·dim² PER STREAM).
dense_block_N_per_stream = 12 * dim**2

# Per-sample modulation linears (img_mod + txt_mod = 12·dim², applied to
# the timestep embedding once per sample, not per token).
mod_block_N = 12 * dim**2

# Patch / encoder / proj-out linears (apply to image or text tokens only).
img_in_N   = in_channels         * dim
txt_in_N   = joint_attention_dim * dim
proj_out_N = patch_size**2 * out_channels * dim

# Dense terms, factor 6 = 2 FLOPs/MAC × 3 (fwd+bwd):
img_dense  = 6 * (L * dense_block_N_per_stream + img_in_N + proj_out_N) * img_tot
txt_dense  = 6 * (L * dense_block_N_per_stream + txt_in_N)               * txt_tot
mod_flops  = 6 * L * mod_block_N * B

# Non-causal joint attention, factor 12 = 2 FLOPs/MAC × 2 matmuls × 3 (fwd+bwd):
attn_flops = 12 * L * H * d * Σ_i (img_s_i + txt_s_i)**2

# Per-call total (single train_batch / infer_batch call):
flops_per_call = (img_dense + txt_dense + mod_flops + attn_flops) \
                 * num_timesteps * cfg_passes
```

`num_timesteps` is the size of the denoising loop in `DiffusersFSDPEngine.forward_backward_batch` (e.g. `data["all_timesteps"].shape[1]` for FlowGRPO; `1` for DPO). `cfg_passes` is `1` (no-CFG / guidance-distilled) or `2` (True-CFG / standard CFG), resolved by `resolve_cfg_passes` (§3.7).

For forward-only paths (`infer_batch` and reference policy log-prob), the caller divides MFU by 3 — identical to upstream. The `/3` removes the backward contribution from the factor-6 dense convention: forward-only is `2 · params · tokens` (factor 2 = 2 FLOPs/MAC, no backward), the formula bakes in factor 6 = 2 · 3, so `/3` recovers the forward-only count. The same `/3` applies to the factor-12 attention term (`12 / 3 = 4` matches forward-only attention: `2 × 2 · s²·d` with no backward).

```python
if forward_only:
    final_metrics["mfu"] /= 3.0
```

The same skeleton is reused for SD3, Flux, etc. via additional entries in the registry; each only needs to express its own per-block param count and embedding linears. Architectures missing from the registry log a warning once and return 0, exactly like `_estimate_unknown_flops` upstream.

### 3.3 MFU formula and reporting

Given `flops_per_call` from §3.2, the elapsed wall time `delta_time` returned by `_postprocess_output`'s timer, and `peak_FLOPS` from `verl.utils.flops_counter.get_device_flops()` (e.g. `119.5 · 1e12` for L20X bf16):

```
achieved_FLOPS_per_device = flops_per_call / (delta_time × world_size)
MFU                       = achieved_FLOPS_per_device / peak_FLOPS
                          = flops_per_call / (delta_time × world_size × peak_FLOPS)
```

`MFU = 1.0` means every GPU in the DP group is running at the device's advertised peak. The `/world_size` factor mirrors upstream's LLM path: `_postprocess_output` consumes the all-gathered `flops_per_call` (across the DP group) and then divides by `torch.distributed.get_world_size()` to recover per-GPU achieved compute. The diffusion path reaches the same invariant via `_allgather_diffusion_flops_meta` on `latent_seqlens` / `prompt_seqlens` before `estimate_flops` runs (§3.4, item 4).

The metric is published under the same keys upstream uses:

```
actor/mfu                # actor training, fwd+bwd
actor_infer/mfu          # actor log-prob recompute, fwd only (MFU /= 3)
ref/mfu                  # reference policy log-prob, fwd only (MFU /= 3)
```

### 3.4 Where the counter is called

`verl_omni/workers/engine_workers.py:TrainingWorker`:

1. **`__init__`**: when `model_type in {"diffusion_model", "diffusion_dp_model"}`, construct `DiffusionFlopsCounter(architecture, transformer_config_dict)` instead of leaving `self.flops_counter = None`.

2. **`_postprocess_output`** receives a small new bundle of fields, all produced by **architecture-agnostic** helpers in `verl_omni.utils.diffusion_flops_counter` (pure Python, unit-testable without Ray / a worker fixture):

   - `latent_seqlens: list[int]` — per-sample image- or video-latent token count, produced by `get_latent_seqlens(data)` which understands:
     - `(B, L, C)` 3-D and `(B, C, H, W)` 4-D image latents.
     - `(B, C, T, H, W)` 5-D video latents (Wan, HunyuanVideo, LTX, CogVideoX).
     - The FlowGRPO `all_latents` time-stacked variants (4-D / 5-D / 6-D).
     - Returns `0` for shapes it cannot interpret, matching the "no info → 0 FLOPs" graceful-degradation of the upstream LLM counter.
   - `prompt_seqlens: list[int]` — per-sample text-encoder token count (derived from `prompt_embeds_mask.sum(-1)`; falls back to the dense `prompt_embeds.shape[1]` when no mask is supplied; `0` for unconditional / class-conditioned models).
   - `num_timesteps: int` — `1` for DPO, `data["all_timesteps"].shape[1]` for FlowGRPO-family algorithms.
   - `cfg_passes: int` — produced by `resolve_cfg_passes(pipeline_cfg, transformer_cfg)`, which consults in order:
     1. Explicit `pipeline.cfg_passes` override (for custom rollouts).
     2. `transformer.guidance_embeds == True` → `1` (Flux-style guidance-distilled models).
     3. `pipeline.true_cfg_scale > 1.0` → `2` (Qwen-Image "True CFG").
     4. `pipeline.guidance_scale > 1.0` → `2` (Wan / SD3 / standard CFG).
     5. Otherwise `1` (unconditional / class-conditioned / explicit no-CFG / `guidance_scale = 1.0`).

   These are extracted in `train_batch`/`infer_batch` directly from the TensorDict (the same place that already extracts `global_token_num` for the LLM path), then passed to `_postprocess_output` as a single `flops_meta: dict` to keep the signature small.

3. **`_postprocess_output` branches by counter type** (no wrapper layer). The LLM path keeps its existing positional signature `estimate_flops(global_token_num, delta_time, images_seqlens=…)` and the diffusion path calls `estimate_flops(delta_time=…, **diffusion_flops_meta)` with the four diffusion-specific kwargs. Both branches share the same MFU normalisation (`/world_size` and, when `forward_only=True`, the additional `/3`).

4. **Global vs per-rank invariant.** Upstream's `_postprocess_output` feeds the counter `global_token_num` (allgathered across DP via `train_mini_batch`) and then divides MFU by `torch.distributed.get_world_size()` to recover per-GPU achieved compute. The diffusion path mirrors this: `_collect_diffusion_flops_meta` produces per-rank `latent_seqlens` / `prompt_seqlens`, and `_allgather_diffusion_flops_meta` concatenates them across the DP group before `estimate_flops` runs. Failing to do so would double-count, reporting world_size× higher MFU than reality.

### 3.5 Wiring through the trainer

The actor training metrics already flow back via `update_actor` → `_postprocess_output` → `actor/mfu`. No trainer change is needed there.

For `actor_infer/mfu` and `ref/mfu`, the diffusion trainer currently discards the `metrics` field returned by `infer_actor_batch` / `infer_ref_batch` (it only reads `log_probs`/`prev_sample_mean`). `ray_diffusion_trainer.py` extracts these metrics and merges them into the per-step metric dict alongside `actor/*`, mirroring the upstream LLM trainer's `metrics.update({"perf/mfu/actor_infer": ...})` step.

### 3.6 How to add a new architecture

The counter is intentionally a small registry pattern so adding Wan2.2, Z-Image, SD3, Flux, or any future DiT is a *single function* change.

```python
# verl_omni/utils/diffusion_flops_counter.py
@register_diffusion_flops_estimator("WanPipeline")  # value of model_index.json _class_name
def estimate_wan_flops(
    config: Mapping[str, Any],
    latent_seqlens: Sequence[int],   # per-sample video-latent token count (T*H*W)
    prompt_seqlens: Sequence[int],  # per-sample text-encoder token count
    delta_time: float,
    *,
    num_timesteps: int,
    cfg_passes: int,
) -> float:
    # The two metadata extractors are reused as-is:
    # - latent_seqlens already encodes T*H*W via get_latent_seqlens.
    # - cfg_passes already accounts for guidance_scale > 1 / guidance_embeds.
    # The estimator only needs to know its own attention topology:
    #   - Wan2.2 uses self-attn on image + cross-attn to text (NOT joint).
    #   - SD3 / Flux use joint full attention (like Qwen-Image).
    #   - Z-Image concatenates text features as extra tokens → joint attention.
    ...
```

What the contributor has to write:

| Component | Where to look in the diffusers code | Encoded in the estimator as |
| --- | --- | --- |
| Per-block linear param count | `__init__` of the block module (`WanTransformerBlock`, `ZImageTransformerBlock`, …) | `dense_block_N_per_stream` (joint) or `(dense_block_self_N, dense_block_cross_N)` (cross-attn variants) |
| Attention topology | `forward` of the block module — joint vs self+cross | `attn_flops = 12·L·H·d · Σ s²` (joint) vs `6·L·H·d · Σ img_s²  +  6·L·H·d · Σ img_s·txt_s` (self+cross) |
| Embedding / proj-out linears | `patch_embedding`, `proj_out` modules | `img_in_N`, `proj_out_N` (per arch) |
| MoE sparsity (if any) | `num_experts` / `num_experts_per_tok` on the transformer config | Multiply MLP FLOPs by `num_experts_per_tok / num_experts` |

What the contributor does **not** have to write:

- Latent → seqlen extraction (`latent_seqlens_from_latent_shape` handles 3-D / 4-D / 5-D / 6-D layouts including video and FlowGRPO time-stacked variants).
- CFG-pass detection (`resolve_cfg_passes` handles `true_cfg_scale`, `guidance_scale`, `guidance_embeds`, and explicit pipeline overrides).
- Distributed all-gather (`_allgather_diffusion_flops_meta` does the cross-rank concatenation regardless of the architecture).
- Forward-only divisor (`_postprocess_output` divides by 3 generically).
- Device peak-FLOPS lookup (reuses `verl.utils.flops_counter.get_device_flops`).

The result: a new architecture is a single new `@register_diffusion_architecture` class plus a matching unit test in `tests/utils/test_diffusion_flops_counter_on_cpu.py`. No trainer or worker changes are required.

### 3.7 Non-CFG compatibility

The design supports four CFG variants without any per-arch code:

| Pipeline configuration | Resolved `cfg_passes` | Models |
| ---------------------- | --------------------- | ------ |
| `true_cfg_scale > 1.0` | 2 | Qwen-Image with True-CFG |
| `guidance_scale > 1.0`, `guidance_embeds=False` | 2 | Wan2.2, SD3, most standard DiT pipelines |
| `guidance_scale > 1.0`, `guidance_embeds=True` | 1 | Flux (guidance-distilled) |
| `true_cfg_scale == 1.0` *or* `guidance_scale == 1.0` *or* unconditional / class-conditioned | 1 | Qwen-Image non-CFG, unconditional DiT, class-cond DiT |

The non-CFG path uses the same code path; `cfg_passes = 1` is a no-op multiplier inside `estimate_*_flops`. The unit test `TestNonCfgQwenImageFlops` pins this behaviour so future estimators in the registry can rely on it.

## 4. Non-goals

- **Rollout MFU** (vllm-omni's diffusion sampling). vllm-omni does not run inside the `TrainingWorker.Timer` block, so attributing time to a single rank is brittle; we leave this to a follow-up.
- **Activation / KV-cache memory MFU.** Out of scope.
- **VAE / text-encoder FLOPs.** They are loaded inside vllm-omni rollout only; the training engine sees only the DiT, so this RFC tracks only the DiT.

## 5. Risks / open questions

- **MoE / sparse DiTs** are not yet shipped. The §2 table treats them as a per-arch concern (e.g. Wan2.2-MoE would multiply MLP FLOPs by `num_experts_per_tok / num_experts`); the initial PR only ships a dense Qwen-Image estimator. The registry pattern makes adding an MoE estimator a follow-up PR with no core changes.
- **Qwen-Image-Edit / Img2Img / Inpaint / ControlNet variants** are deferred. They share the same `QwenImageTransformer2DModel` denoiser but the edit/img2img pipelines concatenate reference-image latents to the noise latents along the sequence dim (`torch.cat([latents, image_latents], dim=1)`), so the effective image-side seqlen is roughly doubled. The current registry warns + reports MFU=0 for these `_class_name`s (pinned by `test_unknown_architecture_warns_once_and_returns_zero`) rather than under-counting silently. Adding support is a follow-up: register the pipeline alias and extend `_collect_diffusion_flops_meta` to sum noise-side + reference-side image seqlens (ControlNet additionally adds the ControlNet backbone's FLOPs). No core changes required.
- **CFG with gradient detachment.** If a future loss path detaches the negative branch, our `cfg_passes` accounting becomes a slight over-estimate (≤ 2×). We document the assumption and expose `cfg_passes` as an explicit override on the counter (`pipeline.cfg_passes`).
- **Variable text seq length under SP padding.** The counter uses `prompt_embeds_mask.sum(-1)` (the *unpadded* per-sample length). The model, on the other hand, runs on prompt embeds padded to a multiple of `sp_size` (`_pad_embeds_for_sp` in `DiffusersFSDPEngine`). The counter therefore *slightly under-counts* attention FLOPs in the presence of SP padding (≤ `sp_size − 1` extra tokens per sample, typically < 2% of `txt_s`). A future revision can pad the text-side seqlens to match if this becomes material.

## 6. Test plan

CPU-only unit tests under `tests/utils/test_diffusion_flops_counter_on_cpu.py`:

1. **Architecture dispatch**: registering a stub estimator for a synthetic architecture is picked up by `DiffusionFlopsCounter`; unknown architectures warn once and contribute 0 FLOPs.
2. **Numerical scaling** (`TestQwenImageFlopsScaling`):
   - Per-step FLOPs scales linearly with `num_timesteps`.
   - Per-step FLOPs scales linearly with `cfg_passes` (including `cfg_passes=1` for non-CFG runs, see `TestNonCfgQwenImageFlops`).
   - Dense FLOPs are linear in `(img_tot, txt_tot)`; attention FLOPs are quadratic in `(img_s + txt_s)` summed per sample.
   - The full formula matches a hand-rolled reference implementation (`_reference_qwen_image_flops`) within 1e-9 across multiple resolutions / batch sizes.
3. **Parameter-count grounding** (`TestQwenImageFlopsParamCount`):
   - The per-stream dense-block parameter count baked into the formula matches a freshly-instantiated `QwenImageTransformerBlock.numel()`.
   - The per-token dense FLOPs predicted by the formula matches `6 · transformer.numel() · img_tot` (LLM-counter convention) within 1% — a ground-truth sanity check that the formula did not double- count or omit any linear.
4. **Distributed wiring** (`TestDPGlobalConsistency`):
   - Sum of per-rank `latent_seqlens` fed to `estimate_flops` and divided by `world_size` equals the per-rank `latent_seqlens` fed without division. This is the global-vs-per-rank invariant that makes the LLM and diffusion paths share the same `_postprocess_output` / `world_size` divisor without double-counting.
   - Realistic step-time × global-seqlens produces MFU < 1.0 on a 4-GPU H200 (a regression guard against the >100% MFU bug we fixed during the bring-up).
5. **Architecture-agnostic metadata extractors**:
   - `TestGetLatentSeqlens`: 3-D image, 4-D image, 5-D video, and the 4-D / 5-D / 6-D FlowGRPO `all_latents` variants all extract the correct token count; garbage shapes return 0.
   - `TestResolveCfgPasses`: non-CFG, `true_cfg_scale`, `guidance_scale`, `guidance_embeds`, and explicit `cfg_passes` overrides all resolve to the right number of forward passes.
6. **Forward-only divisor**: integration with `_postprocess_output` keeps `mfu` divided by 3 for `forward_only=True` (covered by the existing diffusers FSDP engine test, see below).
7. **Placeholder loss suppression**: `_postprocess_output` correctly drops the placeholder `loss=1.0` when `record_loss=False`.

End-to-end:

- Reuse the existing `tests/workers/test_diffusers_fsdp_engine.py` to assert that `train_mini_batch`'s returned metrics dict contains an `mfu` field for the Qwen-Image tiny model.
- Smoke-run `examples/flowgrpo_trainer/run_qwen_image_ocr.sh` (full FT) and `examples/flowgrpo_trainer/run_qwen_image_ocr_lora.sh` (LoRA), both with `true_cfg_scale=1.0` (non-CFG) for 4 steps each, and verify the JSONL file logger captures `actor/mfu`, `actor_infer/mfu` (`ref/mfu` is omitted by FlowGRPO since it has no KL term, so `_compute_ref_log_prob` is not invoked).

  Measured on 4× NVIDIA H200 (peak `989` TFLOPS bf16, pinned via `VERL_OMNI_DEVICE_FLOPS_TFLOPS=989`) at 512×512, `num_inference_steps=8`, steady-state (steps 2–3 averaged):

  | run     | `actor/mfu` | `actor_infer/mfu` | `update_actor` | `step_time` |
  |---------|-------------|-------------------|----------------|-------------|
  | full FT (SP=1, offload=False)  | 0.223 | 0.327 | 115.59 s | 297.23 s |

  Step 1 is excluded as warmup; steps 2 and 3 give matching MFU within 5%, indicating the metric is stable across steady-state steps.

  Interpretation:

  - `actor_infer/mfu > actor/mfu` in both runs — expected because the forward-only path has no optimizer/weight-sync overhead.
  - The full-FT figure of ~0.327 forward-only and ~0.223 train MFU on H200 sits in a realistic band for this hardware and confirms the FLOPs formula and `peak_FLOPS` lookup are wired correctly (an earlier revision returned >1.0 due to a missing DP all-gather, fixed by `_allgather_diffusion_flops_meta`).
  - The formula counts the full DiT's fwd+bwd FLOPs uniformly for both runs (matching upstream's LLM counter convention). LoRA's reported MFU is therefore a slight over-estimate of the actual achieved compute (its backward skips `dL/dw` for frozen weights), but it lets users see the *relative* throughput from one comparable metric.

  Caveats worth flagging during review:

  - The two scripts differ in `ulysses_sequence_parallel_size` (SP=2 vs SP=1) and `model_dtype` (default vs explicit `bf16`). Time-per-step is dominated by the reward server (Qwen3-VL-8B with `genrm_ocr.py`), not by the DiT, so the headline `step_time` is not a direct measure of DiT throughput. The `update_actor` and `old_log_prob` rows are the clean DiT-only timings.
  - When comparing two runs whose SP / DP topology differs, the per-rank `data.shape[0]` and the post-allgather `latent_seqlens` length follow different paths through `_allgather_diffusion_flops_meta`, which is deliberate. The unit test `TestDPGlobalConsistency` asserts that "per-rank seqlens × world_size = global seqlens" for the two paths so the per-GPU MFU does not depend on the topology when wall time is held constant.

## 7. Rollout

This RFC is implemented in one PR alongside the counter and tests. No configuration flags are introduced — MFU is always reported (matching the upstream LLM behaviour). The PR description lists the test commands and expected output.
