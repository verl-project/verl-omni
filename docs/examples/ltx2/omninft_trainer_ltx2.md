# LTX-2.3 audio-video OmniNFT

Last updated: 09/17/2026

This recipe trains LoRA adapters for
`diffusers/LTX-2.3-Diffusers@8eee8edcf067e838b843f926ec4d4cc9b2be1aaf`
text-to-audio-video generation on 16 Ascend NPUs. It reuses the standard
LTX-2.3 diffusers actor, vLLM-Omni rollout pipeline, worker placement, and native
reward-model lifecycle. OmniNFT adds the dual-modal direct-preference loss and
routes independent reward components to its video and audio branches.

The configuration uses FSDP2, sequence-parallel size one, and four TP=4
rollout replicas. Each replica serves one sample at a time, with rollout CPU
offload disabled. The recipe sets `actor_rollout_ref.rollout.max_num_seqs=1`.

## Paper feature coverage

The [OmniNFT paper](https://arxiv.org/abs/2605.12480) introduces three core
techniques. This integration currently implements one of them:

| Technique | Status |
|-----------|--------|
| Modality-wise advantage routing | Implemented |
| Layer-wise gradient surgery | Not implemented |
| Region-wise loss reweighting | Not implemented |

The training results in this example cover modality-wise advantage routing.
They do not include layer-wise gradient surgery or the paper's attention-based
region-wise video-loss weighting.

## Prepare data

Convert the OmniNFT VGGSound metadata to the standard RLHF parquet schema:

```bash
python3 examples/omninft_trainer/data_process/prepare_data.py
```

By default, the converter downloads `train_metadata_20k.jsonl` and
`test_metadata.jsonl` from OmniNFT revision
`fb9237f6e74edf0d0f2a683f4d975b79fde588fe` and writes `train.parquet` and
`test.parquet` under `data/omninft/vggsound/verl_omni`. This is the same
revision used by the pinned reward reference source. Use `--train_file` and
`--val_file` for local metadata files. Samples from the same prompt retain their shared `uid` for group-wise
advantage normalization. Reward workers preserve input order through the existing
single-sample scoring path; a separate reward sample ID is unnecessary.
`MultiVisualRewardManager` records each raw component, and `preserve_components`
assembles these into named columns before OmniNFT normalization and modality
routing. Cross-sample reward inference batching is disabled; model-specific
frame and audio-window processing is retained.

CLAP uses the existing `clap.py::compute_score` with an executor-owned model,
`prompt_key: audio`, scale/offset of `0.5`, and score bounds `[0, 1]`. Without
these options, it retains the existing cached cosine scorer. The supplied model
path selects the checkpoint; optional `model_kwargs` and `processor_kwargs`
are passed to the Transformers loaders. HPSv3 uses the existing
`hpsv3_reward.py::compute_score_hpsv3` with `prompt_key: video`, `num_frames: 5`,
`top_fraction: 0.3`, `score_cap: 15.0`, and `reward_scale: 1.0`. Its managed
model enables `use_sequential_position_ids` to preserve the reference reward's
position handling while using the standard Transformers multimodal forward.
Both paths share model loading and input preparation. Without these options,
HPSv3 retains interval sampling, mean aggregation, scale `0.1`, and cached
cross-request batching. Managed inference batches frames within each sample.
AudioBox, VideoAlign, and DeSync are separate modules directly under
`verl_omni/utils/reward_score/`. Like CLAP and HPSv3, each separates score
computation from executor-owned model loading/inference. Their scorers accept
standard reward arguments and optionally read media metadata from a single-sample
batch; they do not modify that batch. AudioBox exposes `axis_weights` and
`score_scale`; VideoAlign exposes `prompt_key`, `score_weights`, `score_means`,
and `score_stds` (in VQ/MQ/TA order). The recipe sets these choices explicitly.
Checkpoint/source versions are selected when downloading assets, rather than
checked through revision labels in the scorers.

HPSv3 and VideoAlign share the standard Qwen2-VL multimodal reward forward.
Transformers builds and merges visual embeddings; explicit sequential position
IDs preserve the original reward models' decoder behavior. VideoAlign always
uses this position policy; HPSv3 enables it through the recipe option above.
The reward head receives inputs cast to its parameter dtype. No visual-module
wrapping or model-layout aliases are needed. Numerical parity with the reference
checkpoints still requires accelerator validation.

## Prepare model assets

Download the pinned LTX-2.3 base model, install the optional reward packages,
and download the pinned reward assets:

```bash
bash examples/omninft_trainer/download_models.sh
```

The script writes the Hugging Face cache and reward assets under `outputs` by
default, matching the launcher's default paths. Set `MODEL_ROOT` and
`REWARD_ROOT` consistently for another location. If the LTX-2.3 base model is
already available, run `download_reward_models.sh` directly to prepare only
the rewards. Review the licenses of the reward repositories and checkpoints
before use.

The five required components keep the scoring definitions used by OmniNFT:

| Component | Signal | Routed to |
|-----------|--------|-----------|
| VideoAlign | video-text quality and alignment | video, weight 1.0 |
| HPSv3 | five-frame top-30% visual preference | video, weight 1.5 |
| AudioBox | duration-weighted audio aesthetics | audio, weight 0.5 |
| CLAP | paired audio-text cosine score | audio, weight 1.0 |
| DeSync | audio-video synchronization | video and audio, weight 1.0 |

Scores remain separate through reward execution. For each component, OmniNFT
centers scores within a prompt group and, with the checked-in recipe, divides
by that component's full-batch standard deviation (`correction=0`) before
applying the routing weights. A missing required score fails the training step.
The routed actor input is `reward_prob[B,T,2]`; its final dimension is ordered
as video, audio. The shared DiffusionNFT engine executes old, current, and
reference policy forwards, while the OmniNFT adapter only packs and unpacks the
two modalities.

Each named reward model owns an upstream native worker group. The executor loads
the model on wake and closes it on sleep, releasing accelerator memory between
reward phases.

## Launch on Ascend NPU

```bash
bash examples/omninft_trainer/ltx2/run_ltx2_3_omninft_lora_npu_bs32.sh
```

The launcher composes the example-local
`examples/omninft_trainer/ltx2/ltx2_omninft.yaml`, which extends the shared
diffusion trainer configuration and owns the stable algorithm, adapter, loss,
rollout, and five-reward definitions. The shell owns machine-dependent paths,
device placement, run size, and environment setup; trailing Hydra arguments
override either layer.

Useful environment overrides include `DATA_DIR`, `TRAIN_FILE`, `VAL_FILE`,
`MODEL_ROOT`, `MODEL_PATH`, `REWARD_ROOT`, `OUTPUT_DIR`, `NUM_GPUS`, `ROLLOUT_TP`,
`ROLLOUT_N`, `ROLLOUT_MAX_NUM_SEQS`, `TOTAL_TRAINING_STEPS`, and `WANDB_MODE`.
The default device placement is VideoAlign `[0,1,8,9]`, HPSv3
`[2,3,10,11]`, AudioBox `[4,12]`, CLAP `[5,13]`, and DeSync
`[6,7,14,15]`; the corresponding `*_DEVICES` variables can be adjusted
without changing reward or training semantics.

The recipe uses a batch size of 32 and learning rate `3e-5`. Training uses 20
denoising steps at 256x384, and validation uses 40 denoising steps at 256x384.
Training replay applies video/audio CFG values of 1.

See [the example README](../../../examples/omninft_trainer/README.md) for the
data contract, lifecycle details, and exact routing configuration.
