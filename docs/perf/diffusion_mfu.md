(diffusion_mfu)=
# Diffusion FLOPs / MFU

Last updated: 05/31/2026

VeRL-Omni reports **Model FLOPs Utilization (MFU)** for diffusion RL
training using the same three keys upstream
[verl](https://github.com/verl-project/verl) reports for LLM RL — so users
have a single, hardware-agnostic metric to compare across runs,
checkpoints, and clusters. This page describes what is reported, how the
numbers are computed, and how to add an estimator for a new diffusion
architecture.

If you are looking for FlowGRPO-specific training metrics
(`zero_std_ratio`, `ratio_mean`, `pg_clipfrac_*`, ...), see
{ref}`metrics`.

## Reported metrics

The diffusion trainer emits three MFU keys, on the same step cadence as
the rest of the actor metrics:

| Metric             | What is timed                                                                                                                                              |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `actor/mfu`        | Actor `train_batch` — one forward + backward call.                                                                                                         |
| `actor_infer/mfu`  | Actor `infer_batch` — forward only (log-prob recompute on the rollout trajectories).                                                                       |
| `ref/mfu`          | Reference policy `infer_batch` — forward only. Only emitted when a reference policy is constructed (i.e. `actor.use_kl_loss=True` or `algorithm.use_kl_in_reward=True`). |

`MFU = 1.0` means every GPU in the data-parallel group is sustaining the
device's advertised peak FLOPS for the duration of the timed call. The
peak comes from `verl.utils.flops_counter.get_device_flops()`, so the
values are normalised against the same device table upstream verl uses
for LLM MFU — no diffusion-specific device table is introduced.

Absolute MFU values are model-, hardware-, batch-shape-, and
parallelism-dependent; treat them as **relative** numbers for comparing
configurations (LoRA vs full FT, before/after a kernel change, baseline
vs optimisation) on the same setup, not as cross-cluster benchmarks.

> **LoRA caveat.** The formula counts the full DiT's forward+backward
> FLOPs uniformly for both LoRA and full FT. LoRA's reported MFU is
> therefore an over-estimate of the *achieved* compute (its backward
> skips `∂L/∂w` for frozen weights), but it lets you compare relative
> throughput across runs on one metric.

## How FLOPs are computed

### Streams: latent vs prompt

Every diffusion transformer the counter supports has two token streams
with distinct per-block linear groups:

- **Latent stream** — the VAE-encoded tokens the denoiser processes.
  Image latents for T2I, spatiotemporal latents for T2V, audio latents
  for T2A. These flow through the "image-side" linears of each block
  (`to_q/to_k/to_v`, `to_out`, the FFN; named `img_mod` / `img_mlp` in
  the Qwen-Image / SD3 family, `attn1` / `ffn` in Wan). The naming
  matches diffusers' own (`latents`, `image_latents`, `all_latents`) —
  a "latent" is whatever VAE-space tensor goes into the denoiser,
  noisy or not.
- **Prompt stream** — tokens that condition the generation. Typically
  text-encoder tokens after attention masking. These flow through the
  "text-side" or cross-attention linears (`add_q/k/v_proj`,
  `to_add_out`, `txt_mlp` in MM-DiT; `attn2`'s KV path in
  Wan-style cross-attention).

For variants that introduce extra latents, the rule is precise and
local: **if two tensors are concatenated along the sequence dim before
hitting a linear, they belong to the same stream for counting** — there
is no third bucket.

| Pipeline pattern | `latent_seqlens` per sample is | `prompt_seqlens` per sample is |
|---|---|---|
| T2I / T2V / T2A (Qwen-Image, SD3, Flux, Wan2.2, Hunyuan, LTX, AudioLDM-style) | image / video / audio latent tokens only | text-encoder tokens (after mask) |
| Img2Img / Edit / Inpaint (Qwen-Image-Edit, SD3-Img2Img) | denoise-target latent **plus** reference latent — concatenated on the image side before the transformer block, so they share `to_q/k/v` and the FFN | text-encoder tokens |
| ControlNet | denoise-target latent **plus** ControlNet conditioning latent (same image-side concat) | text-encoder tokens |
| Img2Vid (Wan2.2-I2V) | video latent tokens only | text tokens **plus** vision-encoder tokens — the reference image is encoded by a separate encoder and concatenated to the text-encoder output, so both go through the cross-attention KV |
| Class-conditioned / unconditional (DiT class-cond) | image latent tokens | 0 (no prompt stream) |

The joint attention term inside `estimate_flops` uses
`(latent_seqlens[i] + prompt_seqlens[i]) ** 2` per sample — the
"concatenated" length you asked about comes out of this product, it is
not stored separately. For Wan-style **self-attn + cross-attn** the
self-attention term uses `latent_seqlens[i] ** 2` and the cross-attention
term uses `latent_seqlens[i] * prompt_seqlens[i]`. Either way, the two
per-stream totals carry enough information; no third "joint" field is
needed.

### Per-call FLOPs (Qwen-Image reference implementation)

Qwen-Image's transformer block has **two parallel residual streams**
(image tokens, text tokens) and the only place they interact is a single
**joint full attention** that runs on the concatenated sequence. The
diffusers source for `QwenImageTransformerBlock.__init__` makes the
asymmetry explicit (file:
`diffusers/models/transformers/transformer_qwenimage.py`):

```python
class QwenImageTransformerBlock(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim, ...):
        # ---- Image stream (image-side linears) --------------------
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))   # per-sample
        self.img_norm1 = nn.LayerNorm(dim, ...)
        self.attn = Attention(                                             # JOINT attention
            query_dim=dim,
            added_kv_proj_dim=dim,           # has its own KV proj for text stream
            dim_head=attention_head_dim, heads=num_attention_heads,
            processor=QwenDoubleStreamAttnProcessor2_0(),
        )
        self.img_norm2 = nn.LayerNorm(dim, ...)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, ...)              # image-only
        # ---- Text stream (text-side linears) ----------------------
        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))   # per-sample
        self.txt_norm1 = nn.LayerNorm(dim, ...)
        # NOTE: text stream has NO separate attention module —
        # the joint attention above handles both streams.
        self.txt_norm2 = nn.LayerNorm(dim, ...)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, ...)              # text-only
```

In the corresponding `forward`:

```python
# Image stream consumes img_mlp; runs on img_tot tokens per call.
img_modulated, img_gate2 = self._modulate(self.img_norm2(hidden_states), img_mod2, ...)
hidden_states = hidden_states + img_gate2 * self.img_mlp(img_modulated)

# Text stream consumes txt_mlp; runs on txt_tot tokens per call.
txt_modulated, txt_gate2 = self._modulate(self.txt_norm2(encoder_hidden_states), txt_mod2)
encoder_hidden_states = encoder_hidden_states + txt_gate2 * self.txt_mlp(txt_modulated)
```

In other words, `img_mlp` is **never applied to text tokens** and
`txt_mlp` is **never applied to image tokens** — they have disjoint
inputs, so the FLOPs accounting for them must use disjoint token totals.

#### Per-block FLOPs contribution

The block has three groups of linears, each scaling with a different
"token total":

| Module             | Token scope                  | Per-block params         | FLOPs term it generates |
|--------------------|------------------------------|--------------------------|-------------------------|
| `img_attn_qkv`, `img_attn_out`, `img_mlp` | image tokens only (`img_tot`) | $4 \cdot \mathrm{dim}^2 + 8 \cdot \mathrm{dim}^2 = 12 \cdot \mathrm{dim}^2$ | $6 \cdot L \cdot 12\,\mathrm{dim}^2 \cdot \mathrm{img\_tot}$ |
| `txt_added_kv`, `txt_added_q`, `txt_added_out`, `txt_mlp` | text tokens only (`txt_tot`) | $4 \cdot \mathrm{dim}^2 + 8 \cdot \mathrm{dim}^2 = 12 \cdot \mathrm{dim}^2$ | $6 \cdot L \cdot 12\,\mathrm{dim}^2 \cdot \mathrm{txt\_tot}$ |
| `img_mod`, `txt_mod`             | per-sample (`B`)             | $6 \cdot \mathrm{dim} \cdot \mathrm{dim}$ each | $6 \cdot L \cdot 12\,\mathrm{dim}^2 \cdot B$ |
| `attn` (joint QK·V matmul)       | joint seq (`img_s + txt_s`)  | (no extra weights)       | $12 \cdot L \cdot H \cdot d \cdot \sum_i (\mathrm{img}\_s_i + \mathrm{txt}\_s_i)^2$ |

The joint attention adds **no extra weights** in this row because its QKV
projections are already counted in the two stream rows above
(`img_attn_qkv` on the image side, `added_kv_proj` + `added_q` on the
text side); only the data-dependent $\mathrm{softmax}(QK^\top) \cdot V$
matmuls show up here.

#### Closed-form formula

Define:

- $\mathrm{dim} = \mathrm{num\_attention\_heads} \cdot \mathrm{attention\_head\_dim}$
- $L = \mathrm{num\_layers}$, $H = \mathrm{num\_attention\_heads}$, $d = \mathrm{attention\_head\_dim}$
- $\mathrm{img\_tot} = \sum_i \mathrm{img}\_s_i$, $\mathrm{txt\_tot} = \sum_i \mathrm{txt}\_s_i$, $B = \mathrm{batch\_size}$

```python
img_dense   = 6 * (L * 12*dim**2 + in_channels*dim + patch**2*out_channels*dim) * img_tot
txt_dense   = 6 * (L * 12*dim**2 + joint_attention_dim*dim) * txt_tot
mod_flops   = 6 * L * 12*dim**2 * B           # img_mod + txt_mod (per-sample, not per-token)
attn_flops  = 12 * L * H * d * sum_i (img_s_i + txt_s_i)**2

flops_per_call = (img_dense + txt_dense + mod_flops + attn_flops) \
                 * num_timesteps * cfg_passes
```

The leading `6 *` factor on dense terms is `2 FLOPs/MAC × 3 (fwd+bwd)`;
the `12 *` factor on `attn_flops` adds another `× 2` for the two
non-causal attention matmuls ($Q \cdot K^\top$ and $\mathrm{softmax} \cdot V$).
Forward-only callers divide the resulting MFU by 3 in
`_postprocess_output` to remove the backward contribution. This matches
upstream verl's `_estimate_qwen3_vit_flop` (non-causal); the dense
convention is also identical to `_estimate_qwen2_flops`.

The extra terms in `img_dense` (the patch-embed input projection and the
patch-unembed output projection) and in `txt_dense` (the text-encoder
projection into the joint dim) are the **non-block** weights applied once
per token at the input and output of the DiT — `img_dense` and `txt_dense`
roll them in for completeness. They are small relative to the $L \cdot
12 \mathrm{dim}^2$ term but the counter still tracks them so absolute
FLOPs match a hand-rolled `model.numel()` reference (the
`TestQwenImageFlopsParamCount` regression test asserts this).

Per-call multipliers:

- `num_timesteps` — denoising-loop depth.
  `data["all_timesteps"].shape[1]` for FlowGRPO-family algorithms; `1`
  for diffusion DPO.
- `cfg_passes` — `1` (no-CFG / guidance-distilled) or `2` (True-CFG /
  standard CFG), resolved per pipeline by `resolve_cfg_passes`.

### MFU formula

Given `flops_per_call` and the elapsed wall time `delta_time` returned by
the worker's timer:

```python
peak_FLOPS  = get_device_flops()                                      # device peak (TFLOPS scaled by 1e12)
achieved    = flops_per_call / (delta_time * world_size)
MFU         = achieved / peak_FLOPS
if forward_only:
    MFU /= 3.0                                                        # remove backward contribution
```

The `/world_size` divisor mirrors upstream's LLM path:
`_postprocess_output` consumes the all-gathered `flops_per_call` (across
the DP group) and divides by `torch.distributed.get_world_size()` to
recover per-GPU achieved compute. On the diffusion side this is reached
via `_allgather_diffusion_flops_meta` gathering `latent_seqlens` and
`prompt_seqlens` across the DP group *before* `estimate_flops` runs.

## Adding a new architecture

Adding a new architecture is **one class with one required method**.
Subclass `DiffusionModelFlops`, implement `estimate_flops`, and
register it. The base-class `get_latent_seqlens` and
`get_prompt_seqlens` extractors cover the standard `(B, C, *spatial)`
layouts (including FlowGRPO rollout-stacked variants), so most new
T2I / T2V / T2A models do not write any data-plumbing code.

### Step 1 — Identify the pipeline class name

Open the model directory's `model_index.json` and read the top-level
`_class_name`. That string is the registry key.

```json
{
  "_class_name": "WanPipeline",
  "_diffusers_version": "0.32.0",
  "transformer": ["diffusers", "WanTransformer3DModel"]
}
```

`DiffusionModelConfig.architecture` is set to this value automatically
when the model is loaded, so the counter dispatches on it without any
config plumbing.

### Step 2 — Read the diffusers transformer-block source

Open the corresponding transformer block in the diffusers source (for
Wan: `diffusers/models/transformers/transformer_wan.py`). The attention
topology and the per-block linear weights are what you need.

For Wan2.2, the relevant section is:

```python
class WanTransformerBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, ...):
        # 1. Self-attention on image tokens only.
        self.attn1 = WanAttention(dim=dim, heads=num_heads,
                                  dim_head=dim // num_heads, ...)
        # 2. Cross-attention from image tokens to text encoder.
        self.attn2 = WanAttention(dim=dim, heads=num_heads,
                                  dim_head=dim // num_heads,
                                  added_kv_proj_dim=added_kv_proj_dim, ...)
        # 3. Feed-forward on image tokens.
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, ...)
```

Two facts matter for the estimator:

- **Attention topology**: `attn1` is **self-attention on the image
  stream only** (cost $\propto \mathrm{img}\_s^2$). `attn2` is
  **cross-attention from image to text** (cost $\propto \mathrm{img}\_s
  \cdot \mathrm{txt}\_s$). This is different from Qwen-Image's joint
  full attention $\propto (\mathrm{img}\_s + \mathrm{txt}\_s)^2$.
- **Per-block linear weights**:
  - Self-attn: $\mathrm{QKV}$ on image ($3 \cdot \mathrm{dim}^2$) plus
    output projection ($\mathrm{dim}^2$) → $4 \cdot \mathrm{dim}^2$.
  - Cross-attn: $\mathrm{Q}$ on image ($\mathrm{dim}^2$), $\mathrm{KV}$
    on the text stream ($2 \cdot \mathrm{dim} \cdot
    \mathrm{added\_kv\_proj\_dim}$), and the output projection
    ($\mathrm{dim}^2$). If `added_kv_proj_dim == dim` this simplifies
    to $4 \cdot \mathrm{dim}^2$; otherwise compute it from the config.
  - FFN: $\mathrm{dim} \to \mathrm{ffn\_dim} \to \mathrm{dim}$ → $2
    \cdot \mathrm{dim} \cdot \mathrm{ffn\_dim}$ (i.e. **not** the $8
    \cdot \mathrm{dim}^2$ assumption Qwen-Image makes; Wan uses an
    explicit `ffn_dim`).

### Step 3 — Write the architecture class

```python
# verl_omni/utils/diffusion_flops_counter.py

@register_diffusion_architecture(
    "WanPipeline",
    "WanPipelineWithLogProb",   # alias, if you ship a custom rollout class
)
class WanFlops(DiffusionModelFlops):
    """Wan2.2 DiT FLOPs estimator (self-attn + cross-attn topology)."""

    # latent_seqlens and prompt_seqlens are inherited from the base class.
    # Wan's (B, C, T, H, W) video latents and FlowGRPO's (B, T_steps, C,
    # T, H, W) rollout-stacked variant are both handled by the default
    # extractor — the latent stream here is the video latent tokens.

    def estimate_flops(
        self,
        latent_seqlens: Sequence[int],
        prompt_seqlens: Sequence[int],
        delta_time: float,
        *,
        num_timesteps: int,
        cfg_passes: int,
    ) -> float:
        num_heads  = int(self.config["num_attention_heads"])
        head_dim   = int(self.config["attention_head_dim"])
        num_layers = int(self.config["num_layers"])
        ffn_dim    = int(self.config["ffn_dim"])
        added_kv   = int(self.config.get("added_kv_proj_dim") or self.dim)
        dim        = self.dim

        # latent_s = tokens flowing through attn1 + ffn (the image-side
        # linears in Wan; the latent stream here is the video latents).
        latent_tot = sum(int(s) for s in latent_seqlens)
        prompt_tot = sum(int(s) for s in prompt_seqlens)
        batch      = max(len(latent_seqlens), len(prompt_seqlens))

        # Per-block linear param counts.
        self_attn_n = 4 * dim * dim                           # QKV + out, latent-side
        cross_q_n   = 1 * dim * dim                           # Q from latent stream
        cross_kv_n  = 2 * dim * added_kv                      # KV from prompt stream
        cross_o_n   = 1 * dim * dim
        ffn_n       = 2 * dim * ffn_dim                       # explicit ffn_dim, no 4x assumption

        # Dense FLOPs.
        latent_dense = self.compute_dense_flops(num_layers * (self_attn_n + ffn_n), latent_tot)
        cross_dense  = self.compute_dense_flops(
            num_layers * (cross_q_n + cross_o_n), latent_tot
        ) + self.compute_dense_flops(
            num_layers * cross_kv_n, prompt_tot
        )
        mod_flops    = self.compute_dense_flops(num_layers * (6 * dim * dim), batch)  # per-sample timestep embed

        # Attention FLOPs. Factor 12 = 2 FLOPs/MAC * 2 matmuls * 3 (fwd+bwd).
        self_attn_flops = 12 * num_layers * num_heads * head_dim * sum(int(s) ** 2 for s in latent_seqlens)
        cross_attn_flops = 12 * num_layers * num_heads * head_dim * sum(
            int(l) * int(p)
            for l, p in zip(latent_seqlens, prompt_seqlens, strict=False)
        )

        flops_per_call = (
            latent_dense + cross_dense + mod_flops + self_attn_flops + cross_attn_flops
        ) * num_timesteps * cfg_passes
        return flops_per_call / delta_time / 1e12              # → TFLOPS achieved
```

What you did **not** have to write:

- **Latent → seqlen extraction.** The base class default reads
  `data["image_latents"]` (training) or `data["all_latents"]`
  (FlowGRPO rollout-stacked) and returns the product of the spatial
  dims. Wan's `(B, C, T, H, W)` produces `T*H*W` tokens per sample
  out of the box; the rollout-stacked `(B, T_steps, C, T, H, W)`
  collapses to the same per-sample count. MM-DiT-family pipelines
  (Qwen-Image, SD3, Flux, ...) call `diffusers._pack_latents` *before*
  the transformer, reshaping `(B, C, H, W)` into a packed
  `(B, L, C')` (or `(B, T_steps, L, C')` after FlowGRPO stacking) with
  `L = (H/p) * (W/p)` and `C' = C * p**2 == in_channels`;
  `QwenImageFlops.get_latent_seqlens` overrides the default to detect
  this layout via `shape[-1] == in_channels` and return `L` per
  sample, so subclasses inheriting from `QwenImageFlops` get the
  packed handling for free.
- **Prompt → seqlen extraction.** The default reads
  `prompt_embeds_mask` (nested or dense) and falls back to dense
  `prompt_embeds.shape[1]` or zeros for unconditional models.
- **CFG-pass detection.** `resolve_cfg_passes` already covers Wan's
  `guidance_scale > 1` → 2 passes, including the
  `guidance_embeds=True` short-circuit for guidance-distilled variants.
- **Distributed all-gather.** `TrainingWorker._allgather_diffusion_flops_meta`
  is topology-agnostic.
- **Forward-only divisor.** `_postprocess_output` applies the `/3` after
  `estimate_flops` returns.
- **Device peak lookup.** The counter reuses
  `verl.utils.flops_counter.get_device_flops()`.

#### Sidebar — overriding `get_latent_seqlens` for Img2Img / Edit / ControlNet

Image-edit and ControlNet variants concatenate reference latents to the
denoise-target latents along the sequence dim before the transformer
block, so the reference tokens flow through the **same** image-side
linears (`to_q/k/v`, `to_out`, the FFN) as the denoise targets. They
therefore belong on the latent stream — the effective
`latent_seqlens[i]` becomes
`denoise_target_token_count + reference_token_count` per sample, not a
separate field. Subclass the parent T2I class and override
`get_latent_seqlens` only; `estimate_flops` is inherited:

```python
@register_diffusion_architecture("QwenImageEditPipeline")
class QwenImageEditFlops(QwenImageFlops):
    def get_latent_seqlens(self, data: Any = None, config: Optional[Mapping[str, Any]] = None) -> list[int]:
        # `super()` already handles the diffusers-packed (B, L, C')
        # layout for the denoise-target stream; the reference stream
        # arrives in the same packed shape, so its L is just shape[-2].
        base = super().get_latent_seqlens(data, config)
        ref = data.get("reference_image_latents")
        if ref is None:
            return base
        ref_per_sample = int(ref.shape[-2])
        return [b + ref_per_sample for b in base]
```

The same pattern applies to Img2Img, Inpaint, and ControlNet — just
swap the `reference_*` key for whichever your pipeline stores the
extra latents under. For Img2Vid models that concatenate
vision-encoder tokens to the text-encoder output instead, override
`get_prompt_seqlens` in the same way (add the encoded reference-image
token count to each per-sample entry).

### Step 4 — Add a unit test

```python
# tests/utils/test_diffusion_flops_counter_on_cpu.py

WAN_CONFIG: dict = {
    "_class_name": "WanTransformer3DModel",
    "num_attention_heads": 16,
    "attention_head_dim": 128,
    "num_layers": 30,
    "ffn_dim": 8192,
    "added_kv_proj_dim": 2048,
}

class TestWanFlopsScaling:
    def test_linear_in_num_timesteps(self):
        counter = DiffusionFlopsCounter("WanPipeline", WAN_CONFIG)
        kw = dict(latent_seqlens=[512] * 2, prompt_seqlens=[64] * 2,
                  delta_time=1.0, cfg_passes=1)
        est_a, _ = counter.estimate_flops(num_timesteps=10, **kw)
        est_b, _ = counter.estimate_flops(num_timesteps=30, **kw)
        assert math.isclose(est_b / est_a, 3.0, rel_tol=1e-9)

    def test_quadratic_in_latent_seqlen(self):
        counter = DiffusionFlopsCounter("WanPipeline", WAN_CONFIG)
        kw = dict(prompt_seqlens=[64], delta_time=1.0,
                  num_timesteps=1, cfg_passes=1)
        small, _ = counter.estimate_flops(latent_seqlens=[256], **kw)
        large, _ = counter.estimate_flops(latent_seqlens=[512], **kw)
        # Self-attn is quadratic, dense is linear → ratio is between 2 and 4.
        assert 2.0 < large / small < 4.0
```

Mirror the pattern in `TestQwenImageFlopsScaling` for fuller coverage
(hand-rolled reference comparison, `cfg_passes=2`, batch-shape sweep).
For architectures whose weights are tractable to enumerate, also add a
`TestArchFlopsParamCount` test that asserts the per-block parameter
count baked into the estimator matches `block.numel()` on a
freshly-instantiated tiny model.

### Step 5 — Verify on a smoke run

Use any existing diffusion-RL launch script (e.g.
`examples/flowgrpo_trainer/run_qwen_image_ocr.sh` adapted to your
pipeline). Look for the three keys in your logger output:

```text
{"actor/mfu": 0.XX, "actor_infer/mfu": 0.XX, ...}
```

(Any in-range value confirms the counter is wired up. Absolute numbers
depend on model, hardware, batch shape, and parallelism.)

If `actor/mfu` is `0` or missing, check:

1. `DiffusionModelConfig.architecture` matches the string you passed
   to `@register_diffusion_architecture`. The pipeline class name in
   `model_index.json` is the source of truth.
2. The transformer config file exists at
   `<local_path>/transformer/config.json`. The counter warns and
   degrades to `0` when this file is missing.
3. The config fields your `estimate_flops` reads (`num_layers`,
   `ffn_dim`, ...) actually appear in the diffusers config. Print
   `counter.config` to confirm.
4. The default `get_latent_seqlens` finds your latents. The default
   looks for `data["image_latents"]` then `data["all_latents"]`. If
   your pipeline stores its latent-stream tensor under a different key
   (e.g. `data["audio_latents"]`), override `get_latent_seqlens` in
   the architecture class — same pattern as the Edit sidebar above.

If `actor/mfu > 1.0`, the most common cause is forgetting to gather
`latent_seqlens` / `prompt_seqlens` across the DP group — but
`_allgather_diffusion_flops_meta` handles that generically, so it
should not happen unless you added a new metadata field that bypasses
the gather. The regression test `TestDPGlobalConsistency` guards
against this for the shipped path.

## Caveats and limitations

- **LoRA over-estimates achieved compute.** As noted above, the formula
  treats LoRA and full FT identically. Use the absolute number for
  *relative* comparisons across runs, not as a hardware benchmark.
- **SP padding undercounts attention.** The counter feeds
  `prompt_embeds_mask.sum(-1)` (the unpadded length) into the formula,
  but the model runs on prompt embeds padded to a multiple of `sp_size`
  by `_pad_embeds_for_sp`. The undercounted text-side seqlen is at most
  `sp_size - 1` tokens per sample; the corresponding share of
  `flops_per_call` depends on the model's text-vs-image work ratio
  but is dominated by the image stream and joint attention.
- **CFG with gradient detachment.** If a future loss path detaches the
  negative-CFG branch's backward, `cfg_passes` becomes a slight
  over-estimate ($\le 2\times$). The pipeline can override the
  detection with an explicit `pipeline.cfg_passes: 1` field.
- **Image-edit / Img2Img / Inpaint / ControlNet variants are not yet
  estimated.** These pipelines concatenate reference latents to the
  denoise-target latents along the sequence dim, so the effective
  `latent_seqlens` is larger than the spatial-dim product of the
  denoise-target tensor alone; the current registry warns + reports
  MFU=0 for them rather than under-counting silently. See [Adding a
  new architecture](#adding-a-new-architecture) for the override
  pattern.
- **Rollout FLOPs are out of scope.** vLLM-Omni runs the rollout
  decoder outside the `TrainingWorker.Timer` block and on possibly
  different hardware; attributing FLOPs there is a follow-up.

## Further reading

- [Upstream `verl.utils.flops_counter`](https://github.com/verl-project/verl/blob/main/verl/utils/flops_counter.py) — the LLM-side counter and the `get_device_flops` table.
- {ref}`metrics` — FlowGRPO-specific metrics (`zero_std_ratio`, `ratio_mean`, `pg_clipfrac_*`).
- [`docs/perf/profiler.md`](profiler.md) — `nsys` / `torch.profiler` recipes when MFU alone is not enough to localise a regression.
