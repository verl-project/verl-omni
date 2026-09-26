# Offline diffusion model publishing

Last updated: 09/26/2026

`verl_omni.model_merger` converts an existing FSDP actor checkpoint into a local
transformer or self-contained inference pipeline. It follows verl's
`ModelMergerConfig`, `merge_and_save()` and `cleanup()` lifecycle without routing
diffusion configs through Transformers `AutoConfig`. The CLI entrypoint only
parses and dispatches operations; shared arguments and configuration live in
`base_model_merger.py`, while one `FSDPModelMerger` owns reconstruction and
architecture-specific packaging.

## Supported architectures

One exporter handles every Diffusers-style training architecture currently
registered in the repository, independently of the training algorithm:

| Architecture | Canonical transformer | `transformer` output | `pipeline` output |
| --- | --- | --- | --- |
| Qwen-Image | QwenImageTransformer2DModel | Yes | Yes |
| Qwen-Image-Edit Plus | QwenImageTransformer2DModel | Yes | Yes, including processor |
| SD3 / SD3.5 | SD3Transformer2DModel | Yes | Yes, including all three text encoders/tokenizers |
| Flux | FluxTransformer2DModel | Yes | Yes |
| Wan / Wan2.2 | WanTransformer3DModel | Yes | Yes, preserving an optional base `transformer_2` |
| LTX-2 | LTX2VideoTransformer3DModel | Yes | Yes, including audio VAE, connectors and vocoder |
| MiniMax H3 | MiniMaxH3Transformer3DModel | Yes | Yes, converted to the native fused H3 package |
| Boogu-Image | BooguImageTransformer2DModel (`boogu-image`) | Yes | Yes, through the canonical external pipeline |

`--output_format transformer` writes a standalone component loadable with its
canonical class's `from_pretrained()`. `--output_format pipeline` (default)
replaces the selected complete transformer and preserves all other base
components and assets. Missing actor parameters are **never** filled from the base.

For standalone MiniMax H3 output, the base is the canonical Diffusers actor
transformer. For complete Pipeline output, the base is a native MiniMax H3
package: the exporter converts Diffusers names and QKV/GEGLU layouts to the
fused `MiniMaxH3DiTModel` schema, synthesizes and checks `rope.inv_freq`, and
preserves the remaining native audio/video/text components. Boogu resolves the
installed canonical external classes. Both native H3 releases and some Boogu
releases carry Python assets, so copying those local assets requires
`--trust-remote-code`.

BAGEL is deliberately excluded: its training class derives from
`NonDiffusersModelBase` and needs a different native publishing contract.

Supported checkpoint representations:

- One-dimensional FSDP2 DTensor `Shard(d)` and `Replicate`, including uneven and
  empty shards. All mesh coordinates, shapes, placements and dtypes are checked.
- Ordinary full tensors in a one-rank checkpoint.
- Ordinary full-shaped tensors in a multi-rank checkpoint only when every replica
  is exactly equal. Plain dim-0 shards are not guessed or concatenated.

LoRA checkpoints are fused into a complete transformer; see
[LoRA checkpoints](#lora-checkpoints). Not supported yet: standalone LoRA adapter
export, FSDP1 `ShardedTensor`,
HSDP/FSDP+TP, quantized weights, unaudited custom pipelines, architectures
outside the audited table, BAGEL and Omni publishing. Standard Transformers can
continue using `python -m verl.model_merger`; delegation through this entrypoint
is a follow-up. A training engine named `diffusers` does not establish that its
custom model uses a Diffusers publishing layout.

## Inputs

Prepare three non-overlapping local directories: the saved actor, the compatible
base pipeline (or standalone transformer for component output), and an **absent**
output directory whose parent already exists.

```text
actor/
  fsdp_config.json
  model_world_size_2_rank_0.pt
  model_world_size_2_rank_1.pt
  huggingface/config.json

base/
  model_index.json
  transformer/config.json
  transformer/diffusion_pytorch_model.safetensors[.index.json]
  # Native MiniMax H3 instead uses model.safetensors[.index.json].
  vae/...
  text_encoder/...
  tokenizer/...
  scheduler/...
```

The transformer config saved during training must agree with the base's
behavior-affecting config. The exporter constructs a tiny-memory meta-device
schema and checks exact trained keys and shapes. The base must contain standard
safetensors files/indexes. Frozen assets are copied, not downloaded. Hub-cache
file symlinks are dereferenced into regular output files and directory symlinks
are rejected. Python assets are rejected unless `--trust-remote-code` is set;
this is required for local native MiniMax H3 and custom Boogu releases. Component
export reads only the selected config and safetensors, omitting unused assets.

Only trusted checkpoint files may be opened: torch checkpoint deserialization is
pickle-based. `--trust-checkpoint` explicitly acknowledges this; it is **not a
sandbox**. Do not modify source/base files while exporting. Source fingerprints
are checked again before publication.

## CLI and Python

```bash
python -m verl_omni.model_merger merge \
  --backend fsdp \
  --local_dir "$ACTOR_CHECKPOINT" \
  --target_dir "$OUTPUT" \
  --base_model "$BASE_PIPELINE" \
  --trust-checkpoint

python -m verl_omni.model_merger test \
  --backend fsdp \
  --test_hf_dir "$OUTPUT"
```

For a standalone Diffusers component:

```bash
python -m verl_omni.model_merger merge \
  --backend fsdp \
  --local_dir "$ACTOR_CHECKPOINT" \
  --target_dir "$OUTPUT" \
  --base_model "$DIFFUSERS_TRANSFORMER" \
  --output_format transformer \
  --trust-checkpoint
```

Component output may also select a component from a full base pipeline. The
output contains `config.json`, standard Diffusers safetensors and the manifest,
not a misleading `model_index.json`.

Architecture selection is automatic and fail-closed. Complete pipelines use
`model_index.json['_class_name']`; standalone components use
`config.json['_class_name']`. The actor config must match the selected canonical
training class. Qwen-Image and Qwen-Image-Edit share one standalone transformer,
so their component artifacts need no synthetic parent-pipeline choice.

Wan pipelines always treat the actor checkpoint as the repository-standard
`transformer` training component. A base `transformer_2` and the
`boundary_ratio` / `expand_timesteps` options are copied unchanged, so a dual-Wan
output contains both transformers without a Wan-specific CLI choice. Publishing
a separately trained `transformer_2` will require authoritative component
metadata in the checkpoint; one actor checkpoint is never duplicated into both
slots.

For a complete native MiniMax H3 package, point `--base_model` at the
T2VA/FL2VA/Ref2VA pipeline directory and explicitly trust its local component
code:

```bash
python -m verl_omni.model_merger merge \
  --backend fsdp \
  --local_dir "$ACTOR_CHECKPOINT" \
  --target_dir "$OUTPUT" \
  --base_model "$MINIMAX_H3_PIPELINE" \
  --trust-checkpoint \
  --trust-remote-code
```

The CLI does not expose an algorithm choice. FlowGRPO, NFT and distribution
matching use the same publisher when they save the same complete transformer.

```python
from verl_omni.model_merger import ModelMergerConfig, merge_model, validate_artifact

result = merge_model(ModelMergerConfig(
    operation="merge",
    backend="fsdp",
    local_dir=checkpoint_dir,
    target_dir=output_dir,
    base_model=base_pipeline_dir,
    trust_checkpoint=True,
))
validate_artifact(result.output_dir)
```

The actor configuration defaults to `<local_dir>/huggingface`; use
`--hf_model_config_path` to select another local Hugging Face config directory.
`--hf_upload_path ORG/REPO` uploads a successfully published artifact, and
`--private` requests a private repository. Upload happens only after local
publication and validation; an upload failure does not delete the local artifact.

The default dtype is **preserve**. `--dtype float32`, `float16` or `bfloat16`
explicitly casts checkpoint-derived floating tensors, except declared fp32
islands. Integer/bool buffers and copied frozen base weights are never cast.
Non-finite source values and overflow during casting fail export.

`--max_shard_size` is an output safetensors accumulation budget **in bytes**
(default 2 GiB). An individual larger tensor gets its own shard. Rank archives
are mmap-loaded on CPU and tensors are reconstructed one at a time; this avoids
retaining a complete merged transformer, but it is **not a hard RSS limit**.
Mapped-page residency, reconstruction, serializer/verification copies and source
metadata consume additional memory. MiniMax H3 QKV conversion temporarily holds
three source projections plus the fused output tensor. No fallback to eager
loading is performed for unsupported serialization.

## LoRA checkpoints

A checkpoint whose directory contains `lora_train_meta.json` (written by
`FSDPCheckpointManager` for LoRA training) is published as a complete model: the
selected adapter is folded into its base weights as
`W + (lora_alpha / r) * B @ A`, computed in fp32 and cast back to the base weight
dtype before the `--dtype` policy. The output layouts, verification and loaders
are the same as for full training; no PEFT/LoRA support is needed to load it.

```bash
python -m verl_omni.model_merger merge \
  --backend fsdp \
  --local_dir "$LORA_ACTOR_CHECKPOINT" \
  --target_dir "$OUTPUT" \
  --base_model "$BASE_PIPELINE" \
  --adapter_name default \
  --trust-checkpoint
```

Two checkpoint layouts are accepted:

- **Full state** (default save): `*.base_layer.*` base tensors, adapter tensors and
  all other transformer tensors. Base weights come from the checkpoint and the
  renamed tensor set must match the transformer schema exactly.
- **LoRA-only** (`checkpoint.save_lora_only=True`): only adapter tensors. Every
  base tensor is read from `--base_model`, which must therefore be the base used
  for training. A native MiniMax H3 pipeline base uses the fused native layout and
  is rejected here; export the standalone transformer from a Diffusers H3 base.

`--adapter_name` (default `default`) selects exactly one adapter. Other adapters,
such as DiffusionNFT's `old`, are not fused and are listed in the manifest's
`lora_fusion.excluded_adapters`. `lora_train_meta.json` records the `default`
adapter's positive integer `r` and `lora_alpha`; every adapter created by
verl-omni training shares them. A/B pairing, shapes and rank are validated for
all adapters, including excluded ones.

Only linear LoRA A/B weights are supported. Export fails for missing or invalid
metadata, unpaired A/B tensors, rank or shape mismatches, non-linear targets,
DoRA magnitude, LoRA bias or embedding tensors, a missing adapter, and partial or
mixed tensor sets. rsLoRA and PEFT `alpha_pattern` are not recorded in the
metadata and cannot be detected: verl-omni training never enables them, but an
external `lora_adapter_path` that used them would be fused with the wrong scale.

The manifest `lora_fusion` record lists the adapter, `r`, `lora_alpha`, scaling,
base-weight source, fused modules and excluded adapters, and the source
fingerprint includes `lora_train_meta.json`. VeOmni LoRA checkpoints are not
supported. Design and follow-up standalone adapter export are tracked in
[RFC #675](https://github.com/verl-project/verl-omni/issues/675).

## Verification and failure semantics

Before publication the exporter:

1. Verifies pipeline components, transformer configs, base headers and exact
   trained tensor coverage.
2. Reconstructs DTensors and checks their slices against the original shards;
   replica copies must agree exactly.
3. Reopens every written safetensors shard and checks exact post-cast values,
   shapes and dtypes against its intended tensors.
4. Checks unchanged copied assets, input immutability, output indexes and SHA256
   inventories.

The portable `merge_manifest.json` records the selected architecture, source/base
fingerprints, output tensor/file inventories, dtype and verification outcomes.
Known location-only config metadata (`_name_or_path`, `name_or_path`) is removed
with a recorded deterministic transform. No source directory is recorded in the
manifest. SHA256 validates consistency, not publisher authenticity.

The verl-style `test --test_hf_dir` operation checks published files, checksums,
tensor metadata and indexes without loading source rank checkpoints. The Python
helper remains named `validate_artifact()` because that is its precise action.
It does not rerun training or certify
past source round-trip claims if the artifact and manifest are both replaced.

Publication uses an exclusive lock and owned sibling staging. Linux
`renameat2(RENAME_NOREPLACE)` prevents replacing even an empty directory created
concurrently. Other platforms fail explicitly. Errors clean only this export's
staging/lock, never the source or another exporter's files. Existing targets are
not overwritten; automatic crash recovery/resume is not implemented.

Runtime validation is recorded as `not_run`: there is no implicit generation or
GPU use during export. A two-rank CPU/Gloo integration test trains every available
tiny transformer for one real forward/backward/AdamW step under FSDP2, saves it
through `FSDPCheckpointManager`, merges the resulting rank-local DTensors, reloads
the published component or pipeline, and matches the trained actor's fixed-input
forward output. The MiniMax H3 case additionally loads every converted tensor
through the native vLLM-Omni `MiniMaxH3DiTModel.load_weights()` path.

CPU tests also exercise standalone genuine DTensor serialization, fresh-process
CLI export, tiny complete pipeline reload and transformer-forward parity across
the eight architecture identities above.
All eight also have dtype/fp32-island and incomplete-state checks. Seven
Diffusers-compatible pipelines exercise their canonical complete
`from_pretrained()` loader; MiniMax H3 exercises exact native schema coverage,
QKV interleaving, GEGLU reordering, rope synthesis, copied-component integrity,
and both single-rank and real two-rank DTensor conversion. Wan additionally
covers preservation of its second transformer and scalar options. A registry coverage test
fails if a new Diffusers training architecture is added without exporter coverage.

Run the matrix with the project's venv and optional `boogu-image` dependency:

```bash
TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 OMP_NUM_THREADS=1 \
  python -m pytest tests/model_merger/test_architectures_on_cpu.py \
  tests/model_merger/test_lora_fusion_on_cpu.py -v
```

The LoRA tests cover all eight architectures with full-state and LoRA-only
checkpoints, both single-rank and two-rank DTensors. They compare fused tensors
with PEFT's own `merge()`, reload without adapter layers, check fixed-input
forward parity and exercise the dtype/fp32-island policy. Pipeline tests cover
both layouts for all seven Diffusers-compatible pipelines and verify copied
frozen components, including Wan's second transformer. MiniMax H3 tests check
full-state native conversion and rejection of LoRA-only native pipeline export.

Each architecture also trains `default` with a frozen `old` adapter for one
two-rank FSDP2 step, saves both layouts through `FSDPCheckpointManager`, and
checks the published model against the trained LoRA actor's forward output.
The H3 LoRA case additionally checks the native DiT loader. Optional Boogu cases
are reported explicitly as skipped when its package is unavailable.

Without `boogu-image`, only its cases are skipped; this is not evidence that Boogu
was tested. Local verification used Diffusers 0.40.0 and canonical Boogu source
revision `25f8f888298224a94e5ec2abafb98abea9031a0d` on PYTHONPATH, without changing
the shared venv. Boogu's declared dependency ranges differ from that venv; source
execution is not dependency-resolver or clean-install validation.

These tests use random tiny weights on CPU; they are not real-weight/GPU or
production image-quality evidence. Real checkpoint save/export/load remains a
release acceptance gate.

For a manual loader check after export:

```python
from diffusers import QwenImagePipeline

pipeline = QwenImagePipeline.from_pretrained(output_dir, local_files_only=True)
```

See [RFC #596](https://github.com/verl-project/verl-omni/issues/596) for the
subsequent LoRA, Transformers reuse, BAGEL and stage-specific work packages.
