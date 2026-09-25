# Named diffusion media artifacts

Last updated: 09/10/2026.

All registered in-tree diffusion adapters emit named media. The strategy,
agent loops, rewards and V0/V1 exporters consume declarations rather than
identifying images, videos or latents from tensor rank or channel size.
The public `generate(**kwargs)` RPC is unchanged.

## Contract

`pipelines/rollout_media.py` retains `MediaSpec` and `DiffusionIOSpec`:

- `MediaSpec(modality, representation, layout, sample_rate=None, fps=None)`.
  Modality is `image`, `video` or `audio`; representation is `latent` or
  `decoded`. **The tensor owns its dtype**, not the configuration.
- `DiffusionIOSpec(artifacts=...)` maps available artifact names to `MediaSpec`.

`pipelines/rollout_artifacts.py` defines:

- `MediaArtifact(spec, data, context="")`: one sample, with no implicit request
  batch dimension. Context records the pipeline/request for downstream errors.
- `validate_artifacts`: checks returned versus expected names, duplicates,
  requested-but-absent outputs, declarations, rank, channels and dtype before
  normalizing decoded axes.

Decoded visual data is uint8 `[0,255]`. Audio is a floating waveform with a
positive sample rate. Decoded video requires positive finite FPS. Canonical
layouts are image `CHW`, video `TCHW` and audio `CT`. Conversion from an explicitly
declared `HWC`, `CTHW`, `THWC` or mono `T` is supported at the adapter boundary;
consumers do not identify axes by looking for a dimension of size 3.

Native/packed latents retain their axes, floating dtype and values. Do not
reinterpret a packed sequence as image channels or normalize a training latent
into a decoded-video layout. A codec's VAE input layout and the final sampler's
packed layout are distinct facts.

### Available outputs versus requested outputs

Each adapter declares available names next to its implementation:

```python
from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec

class MyPipeline(...):
    diffusion_io_spec = DiffusionIOSpec(artifacts={
        "image_preview": MediaSpec("image", "decoded", "CHW"),
        "image_latent": MediaSpec("image", "latent", "LC"),
    })
```

The runtime packet declares the outputs actually emitted and explicit `primary`,
`preview` and `audio` selectors. `preview=None` means no preview was produced;
export skips it, never substitutes `responses` or latent tensors. A non-null
selector naming a missing artifact is an error.

The current visual adapters select a native primary for `output_type=latent`
and a decoded primary for `image`/`pt`/`np`/`pil`/`both`. These format spellings
are compatibility inputs, not separate tensor-layout contracts.

To require previews with a latent primary, configure the names explicitly:

```bash
+actor_rollout_ref.rollout.pipeline.output_type=latent \
actor_rollout_ref.rollout.pipeline.requested_outputs='[image_preview]'
```

For audiovisual scoring, request `video_preview` and `audio` as appropriate.
The setting is mirrored to validation and model pipeline configs; validation can
override its own list. Unknown/duplicate requests fail before generation when
an adapter declaration is available. Packed request batches take the union of
all requests' requirements, not just those of the first request.

Requested names express **required availability**, not an exclusive output set.
Qwen/SD3/FLUX/Boogu/Wan can avoid preview decoding when it is not requested and the
primary is latent. LTX decodes its joint pair when either decoded stream is
required. Pinned H3 and Bagel forwards still decode; no selective-decoding
optimization is claimed for those models.

## Implementing an adapter

`pipelines/diffusion_rollout_output.py` separates algorithm data from media:

1. Construct native trajectory fields and `prompt_embeddings` / `rl` groups
   with `rollout_output` or `with_rollout_data`.
2. Attach media with `with_visual_artifacts` (batched VAE visuals and native
   latents), `with_batched_media_artifacts` (explicit B-prefixed streams), or
   `with_media_artifacts` (one sample).
3. Ensure the engine's postprocessor preserves the named envelope. Use
   `wrap_rollout_postprocessor` for a media-only upstream processor; do not run
   VAE normalization/PIL conversion on already-canonical artifacts a second time.

For example, after an image sampler returns packed `BLC` latents and its VAE
returns floating `BCHW` pixels in `[-1,1]`:

```python
return with_visual_artifacts(
    rollout_output_with_training_data,
    decoded=vae_pixels,          # None only when no preview is requested
    latents=final_packed_latents,
    latent_layout="LC",
    decoded_layout="CHW",
    pixel_range="minus_one_one",
    output_type=output_type,
    requested=requested_outputs,
    context=f"pipeline=MyPipeline, request_id={request_id}",
)
```

`pixel_range` is an adapter-owned fact, not detected from dtype or extrema.
`zero_one` and `uint8` are also supported. Capture final latents **before** VAE
casts/unpacking/normalization; never substitute the last saved SDE transition.
Algorithm-owned `latents_clean`, trajectories, reference latents and replay
fields keep their existing representations. Internal engine warmup is not a
serving response: Qwen step hooks and LTX still execute decoder warmup but
return no media packet for the reserved dummy request. Real nonfinite pixels
continue to fail validation.

Raw condition media and engine-prepared views are not aliases. In particular,
Qwen-Image-Edit encodes the raw `multi_modal_data.image` view whose processor
grid matches the already-tokenized placeholders; the engine's separately resized
`additional_information.condition_images` must not be compared against it or
substituted for it. VAE condition tensors/sizes use their own explicit fields.

Request batching uses an explicit leading batch axis for payload samples,
trajectory fields and the two training groups. Other metadata, including the
artifact declaration, remains shared. A mismatched leading size is an error,
not a signal to guess that a training tensor was shared.

## Engine and TensorDict transport

The pinned formatter does not put every visual tensor in `multimodal_output`.
The adapter packs all named tensors under one modality-keyed payload entry,
with declarations/selectors in `metadata.media_artifacts`. Batched helpers use
an explicit list of per-sample mappings so request splitting cannot slice a
latent's native axes accidentally.

`DiffusionStrategy` combines complementary `multimodal_output` and `images`
sources and rejects conflicting data/dtypes. A tensor-only `images` fallback
requires an explicit `final_output_type` matching the named primary. Unlabelled
legacy tensor/tuple results are rejected; the strategy does not guess a tuple's
roles, select the first media key or cast floats into pixels.

`DiffusionOutput` exposes `artifacts`, `primary_artifact` and `preview_artifact`.
The temporary `diffusion_output` / `DataProto.responses` view is a projection of
that named primary. Training fields and the audio projection remain for their
existing consumers; retaining a projection does not add another interpretation.

At the agent-loop boundary, tensors become `media_artifact__<name>` fields.
`media_artifact_specs`, `primary_artifact`, `preview_artifact` and
`media_artifact_context` are tensor-free metadata. Ordinary batching and
TransferQueue preserve these fields, including an explicitly absent preview.
Both reward managers reconstruct artifacts and validate that `responses` really
matches the declared primary, including decoded floating audio. They do not
infer its representation from a global training configuration.

## Consumers and failure policy

- CLAP selects decoded `audio` and its actual rate.
- ImageBind selects decoded `video_preview` and/or `audio` according to its mode.
- Image scorers/frame samplers use the selected decoded preview and canonical
  CHW/TCHW axes. The standalone tensor API requires an explicit media kind.
- SD3 latent HTTP scoring selects `image_latent` and requires native `CHW`;
  it does not accept packed `LC` by guessing that a dimension equals 16.
- V0/V1 rollout dumps, validation and W&B select decoded previews, preserve
  artifact FPS (including fractional rates), and never reinterpret latent data.

Schema, selection, primary-projection and metadata errors fail before scoring or
best-effort I/O. `ArtifactContractError` is fatal even for an optional weighted
sub-reward. Encoder/filesystem/W&B failures still warn/skip or record the existing
per-sample fallback. V1 permits only one pending dump, skips with a warning when
it is busy, and copies only retained previews/audio so row views do not pin a
whole batch or unused training latents in the background queue.

## Layout evidence and GPU checks

The adapters and pinned VAE/postprocessor code establish these distinct layouts:

| Family | Final sampling artifact (per sample) | VAE input | Decoded visual before canonical boundary |
| --- | --- | --- | --- |
| Qwen image / edit / DPO / NFT / dual / mix | packed `LC` | `NCTHW`, image T=1 | `NCHW` after selecting the image frame |
| SD3, Boogu | `CHW` | `NCHW` | `NCHW` |
| FLUX DanceGRPO | packed `LC` | `NCHW` after unpacking | `NCHW` |
| Bagel | packed `LC` | `NCHW` after unpatchifying | RGB PIL image |
| Wan | `CTHW` | `NCTHW` | `NCTHW` |
| LTX | video `CTHW`, audio `CTF` | `NCTHW` / `NCTF` | `NTCHW` from `VideoProcessor(..., output_type="pt")` |
| H3 NFT / FlowGRPO | video `CTHW`, audio `CLT` | `NCTHW` / `CLT` | uint8 `NTHWC` from `_prepare_minimax_h3_video_output` |

H3 and LTX decoded audio is `NCT` at their decoder boundary. H3 uses 32000 Hz;
LTX reads the vocoder's actual `output_sampling_rate` (BWE checkpoints can be
48000 Hz, not the old 24000 Hz default).

FLUX DanceGRPO was aligned when rebasing onto the newer main branch. Its pinned
packing/unpacking, named outputs, per-request splitting and optional preview
union are CPU-tested with mocked encoders/denoising/VAE. Real FLUX VAE and GPU
rollout validation remain pending; the earlier GPU matrix does not cover it.

Reproducible GPU checks live under `tests/special_e2e/`:

- `check_diffusion_vae_layouts.py --models-json <case-to-local-path.json> --output <report.json>`
  loads actual VAE implementations/local weights, records implementation class,
  parameter count and input/output shapes, and validates canonical conversion.
- `check_named_diffusion_rollout.py --model <path> --architecture <name> --algorithm <name> --output-dir <dir>`
  exercises the real engine through production request lowering and output
  parsing; `--num-requests` and `--step-execution` cover the two batching modes.
  `--require-request-batch` additionally checks that producer context contains
  every request ID from a real packed forward. Use `--tokenizer-path` for model
  layouts such as Boogu's `processor/`, and `--deploy-config` with BAGEL's
  single-stage deployment recipe rather than its default two-stage AR pipeline.
- Existing training recipes cover actor updates, ordinary agent loops and V1/TQ.
  V1 requires **both** `trainer.use_v1=True` and `transfer_queue.enable=True`;
  invoking `main_diffusion_v1` alone can fall back to V0.

Set `CUDA_VISIBLE_DEVICES` explicitly and do not stop unrelated Ray/GPU jobs.
Tiny-model plumbing, real-VAE layout verification and full-model training quality
are different evidence. Record the exact checkpoint, pin, mode and result in the
PR; a CPU pass or a VAE-only probe is not an end-to-end training pass.
