# Qwen3-Omni Thinker GSPO recipes

Last updated: 09/21/2026

This directory contains both FSDP2 and Megatron recipes. For non-Megatron
setup, data preparation and training instructions, see the
[parent GSPO guide](../README.md). The launchers below contain each recipe's
defaults and accept CLI overrides.

| Recipe | Backend / platform | Launcher |
| --- | --- | --- |
| GSM8K LoRA | FSDP2 / GPU | [Thinker LoRA](run_qwen3_omni_thinker_gspo_lora_v1.sh) |
| MMK12 LoRA | FSDP2 / GPU | [MMK12](run_qwen3_omni_thinker_gspo_lora_mmk12_v1.sh) |
| MMK12 LoRA, separate-async | FSDP2 / GPU | [MMK12 separate-async](run_qwen3_omni_thinker_gspo_lora_mmk12_separate_async_v1.sh) |
| MMK12 LoRA | FSDP2 / NPU | [MMK12 NPU](run_qwen3_omni_thinker_gspo_lora_mmk12_v1_npu.sh) |
| MMK12 LoRA with on-policy distillation | FSDP2 / NPU | [MMK12 OPD](run_qwen3_omni_thinker_gspo_lora_mmk12_v1_opd_npu.sh) |
| AVQA LoRA | FSDP2 / GPU | [AVQA LoRA](run_qwen3_omni_thinker_gspo_lora_avqa_v1.sh) |
| AVQA full-parameter | FSDP2 / NPU | [AVQA NPU](run_qwen3_omni_thinker_gspo_npu_avqa_v1.sh) |
| NExT-QA full-parameter | FSDP2 / NPU | [NExT-QA NPU](run_qwen3_omni_thinker_gspo_npu_nextqa_v1.sh) |
| AudioMCQ full-parameter, separate-async | Megatron / GPU | [AudioMCQ](run_qwen3_omni_megatron_audiomcq_separate_async.sh) |

The remaining sections describe the **Megatron AudioMCQ** recipe, its dependency
prerequisites and validation limits. For model-adapter development, see the
[Megatron integration notes](../../../docs/contributing/integrating_an_omni_model.md#21-megatron-training-adapters).

## AudioMCQ with Megatron and V1 separate-async rollout

This recipe trains all Thinker language-model parameters (LoRA rank zero),
freezes the vision/audio towers, and generates text conditioned on audio with
standalone vLLM-Omni replicas. It uses `trainer.v1.trainer_mode=omni_separate_async`.
The toy smoke and full-model run both select the shared
`verl_omni/trainer/config/omni_megatron_trainer.yaml`; the public launcher adds
only the AudioMCQ recipe overrides, and the toy adds its small-model overrides.
`OmniMegatronEngine` follows verl's Megatron LM forward flow and selects its
model-specific config preparation and forward through the registered pipeline
adapter. The Qwen3-Omni pipeline owns the support checks, config mapping and
explicit BSHD model call that passes audio tensors and lets the Thinker build M-RoPE.
It uses BSHD, PP1 and CP1;
the development audio bridge does not implement packed sequences, dynamic
micro-batching is disabled, and MTP, dynamic CP and router replay are rejected.
Optimizer, old-policy snapshots, losses and weight export remain
upstream implementations.
An engine-private config view exposes the nested Thinker text dimensions to
upstream Megatron helpers without changing the worker/rollout HF configuration.

## Environment and data

Use the repository's pinned verl/vLLM-Omni runtime and install the audio extra
(`uv pip install -e '.[audio]'`). Megatron also requires a compatible Megatron-Core,
Transformer Engine, and Megatron-Bridge with Qwen3-Omni audio forward/export
support. The recipe selects `use_mbridge=true`, `vanilla_mbridge=false`.
The development audio bridge is
[`hbhflw2000/Megatron-Bridge@fe22f9d2`](https://github.com/hbhflw2000/Megatron-Bridge/commit/fe22f9d20bc32d3f09fd08dd58d9ad701d885d42),
with Megatron-Core `e41b37002cd8df1cd97c93e3e0876cf0850f72f8`.
With Transformers 5.13+, also apply the registration fix from upstream
[Megatron-Bridge #4876](https://github.com/NVIDIA-NeMo/Megatron-Bridge/commit/039156328f9587ccb5a9c8c9e6adf30e63f2cf6a)
to that older bridge revision; otherwise native ASR auto-registration collides
while importing the bridge, before any Omni model is initialized.
The same older bridge also calls an encoder method named
`_get_feat_extract_output_lengths` when trimming audio features. Transformers 5
moved it to the modeling module and changed its return value from a tuple to
output lengths. The bridge must support that API (including the encoder's
`n_window`) before training; successful checkpoint loading alone does not test
this path. Keep this dependency fix in the bridge, not a global runtime patch
in verl-omni. A tested bridge revision with both compatibility fixes is required
before publishing the recipe as reproducible with Transformers 5.
These are external prerequisites, not implementations vendored by this recipe;
install them into the same environment as verl. Native Transformer Engine and
FlashAttention extensions must be built for that environment's PyTorch version.

The full-model TransferQueue path also needs the equal-length 3D position-ID
layout repair tracked by [verl #7901](https://github.com/verl-project/verl/pull/7901).
The development run used the implementation from closed
[verl #7767](https://github.com/verl-project/verl/pull/7767), head `a965a838`.
TransferQueue 0.1.8 can carry `[4, sequence_length]` position IDs whose jagged
layout is inconsistent; the old verl helper changes only `_ragged_idx` without
rebuilding values and offsets. This can fail before the model forward even
though the Thinker ultimately constructs its own M-RoPE. Keep the repair in
verl rather than duplicating it in the Omni trainer.

### Merge prerequisites

This full-model recipe is not reproducible from the repository pins yet. The
following dependency work must land before this PR can be treated as runnable
from a clean checkout:

1. `verl-project/verl` must replace its `megatron-bridge==0.5.2` and paired
   Megatron-Core pins with a tested upstream pair that supports Qwen3-Omni
   Thinker conversion, audio forward/export, Transformers 5 registration, and
   the current audio-length API. Then this repository must bump
   `.github/verl_pin.txt` to that verl revision.
2. verl #7901, or an equivalent replacement for closed verl #7767, must land
   with regression coverage for equal-length multimodal position IDs, followed
   by the same verl pin bump here.

The 150-step acceptance run used the development dependency overrides described
above; it validates this integration path but is not evidence that the public
pins already satisfy these prerequisites. The tiny-random smoke validates
audio transport, optimizer steps, and weight synchronization only. It does not
exercise the full-model TransferQueue position-ID failure and must not be used
as evidence that the dependency issue is fixed.

Keep the recipe's `limit_mm_per_prompt.image=1` even for audio-only data. With
both image and video limits zero, the pinned vLLM-Omni creates vision deepstack
buffers on `meta` but still consumes them during audio/text profiling. Keeping
vision resident avoids that device mismatch at the cost of extra inference
memory. This does not add image samples or unfreeze either tower. The shared
server's existing frontend multimodal-cache reset must also run after sleep;
direct `AsyncOmni.sleep()` alone does not cover that server lifecycle.

Download the [AudioMCQ-StrongAC-GeminiCoT dataset and audio assets](https://huggingface.co/datasets/Harland/AudioMCQ-StrongAC-GeminiCoT)
separately. Its [dataset card](https://huggingface.co/datasets/Harland/AudioMCQ-StrongAC-GeminiCoT/blob/main/README.md)
lists Apache-2.0; check the terms of the underlying audio sources as well.
Prepare a local `data.jsonl` containing `question`, `choices`, `answer`,
`audio_path`, and optional `source_dataset`/`id` fields:

```bash
python examples/gspo_trainer/data_process/audiomcq.py \
  --input-jsonl /data/AudioMCQ/data.jsonl \
  --audio-root /data/AudioMCQ \
  --output-dir /data/audiomcq-prepared \
  --validation-size 256 --seed 42
```

Conversion checks file existence, labels and path containment; it does not
decode the entire audio corpus. Missing/invalid rows are counted in
`dataset_info.json`. Validation holds out 256 unique audio assets; questions
sharing an asset stay in the same split. Existing output files are never
overwritten; use a new output directory for each conversion. Audio paths must
be accessible at the same location on all nodes.

Previously audited AudioMCQ parquets with `prompt`, `audios`, and structured
`reward_model.ground_truth` can be used directly to preserve their exact split.
The scorer accepts exact option text or an option letter inside `<answer>` and
reports `content_correct` and `format_valid`. It preserves the development
recipe's reward semantics, including its handling of repeated answer tags.

## Toy smoke (4 GPUs)

```bash
bash tests/special_e2e/run_qwen3_omni_megatron_audiomcq_smoke.sh
```

Builds the existing multimodal tiny-random Qwen3-Omni checkpoint and short
synthetic PCM WAVs locally. Uses 2 Megatron training GPUs plus a standalone TP2
replica on 2 GPUs, four optimizer steps, sync every two steps, and validation
before training and every two steps. Keeping two updates per sync exercises
V1's old-policy parameter save/restore. Synthetic tones test audio transport
only. Random-model correctness and nonzero reward are **not** acceptance gates.
Inspect finite losses/logprobs, successful optimizer steps, weight transfers,
and validation completion. A zero gradient is permitted when every reward and
advantage is zero.
The toy uses `top_k=1` with positive temperature: unrestricted sampling from its
tiny random vocabulary can emit input-side audio markers in the response,
creating fictitious audio segments that cannot be matched to input features.
This structural-test setting does not change the full-model sampling defaults
and is not evidence of stochastic sampling quality or a learning curve.

## Full-model run (32 GPUs)

After allocating four 8-GPU nodes and starting a Ray cluster, run once on the
head. The defaults request 4 training and 4 standalone rollout GPUs per node,
actor TP4/EP4/PP1 (expert TP1), rollout TP4, and 150 steps with validation every
10 steps. The toy overrides actor TP/EP to one; do not use its unsharded-expert
topology for the 30B run:

```bash
MODEL_PATH=/models/Qwen3-Omni-30B-A3B-Instruct \
TRAIN_FILE=/data/audiomcq-prepared/train.parquet \
VAL_FILE=/data/audiomcq-prepared/validation.parquet \
OUTPUT_DIR=/persistent/audiomcq \
bash examples/gspo_trainer/qwen3_omni/run_qwen3_omni_megatron_audiomcq_separate_async.sh \
  ray_kwargs.ray_init.address=auto
```

The launcher records the command, Git revision, resolved configuration, console
log and TensorBoard events in a unique run directory. Use local scratch for
high-frequency writes and archive once afterward on fragile shared filesystems.
TensorBoard and worker bootstrap environment variables are explicitly forwarded
through Ray's per-job runtime environment, including for pre-started clusters.
Hydra overrides are forwarded unchanged; keep
`data.train_batch_size == parameter_sync_step * actor.ppo_mini_batch_size`.

V1 also uses hybrid replicas on the training pool for its initial sampling
window and validation. Although hybrid switching during training is disabled,
sleep/wake and colocated weight loading still need to work. Prefix caching is
disabled. The old development `fully_async_policy` run does not validate these
V1 lifecycle paths or Decoupled PPO (`bypass_mode=false`). A successful toy smoke
establishes structural coverage, not full-model learning or TP4 numerical
parity; evaluate the need for a new full-model run after reviewing the changes.
