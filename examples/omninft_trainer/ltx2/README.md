# LTX-2.3 audio-video OmniNFT

Last updated: 10/16/2026

This recipe trains LoRA adapters for
`diffusers/LTX-2.3-Diffusers`
text-to-audio-video generation on 16 Ascend NPUs. It reuses the standard
LTX-2.3 diffusers actor, vLLM-Omni rollout pipeline, worker placement, and native
reward-model lifecycle. OmniNFT adds the dual-modal direct-preference loss and
routes independent reward components to its video and audio branches.

The configuration uses FSDP2, sequence-parallel size one, and four TP=4
rollout replicas. Each replica serves one sample at a time, with rollout CPU
offload disabled. The recipe sets `actor_rollout_ref.rollout.max_num_seqs=1`.

The actor uses `model_type=diffusion_nft_model` with `algorithm=omni_nft`.
The shared NFT engine delegates modality-specific training inputs and loss
assembly to the registered LTX adapter, which overrides the defaults on `DiffusionModelBase`.
The adapter prepares transformer inputs and
applies its instance-scoped NPU RMSNorm workaround before forward execution.
Rollout returns named video/audio artifacts with explicit FPS and sample rate,
alongside the clean latent tokens needed for training.

For algorithm coverage, reward curves, and reference/trained video comparisons,
see the [OmniNFT trainer overview](https://github.com/verl-project/verl-omni/blob/main/examples/omninft_trainer/README.md).

Set up the repository by following the
[NPU installation guide](../../../docs/start/install_npu.md), then run the commands
below from the repository root. The launcher accepts environment variables for
local model, reward-model, data, and output paths.

## Prepare data

Convert the native OmniNFT VGGSound
[`train_metadata_20k.jsonl`](https://github.com/zghhui/OmniNFT/blob/fb9237f6e74edf0d0f2a683f4d975b79fde588fe/dataset/vggsound/train_metadata_20k.jsonl)
and
[`test_metadata.jsonl`](https://github.com/zghhui/OmniNFT/blob/fb9237f6e74edf0d0f2a683f4d975b79fde588fe/dataset/vggsound/test_metadata.jsonl)
to the standard RLHF parquet schema:

```bash
python3 examples/omninft_trainer/data_process/prepare_data.py
```

By default, the converter reads both files from the pinned OmniNFT revision
`fb9237f6e74edf0d0f2a683f4d975b79fde588fe` and writes `train.parquet` and `test.parquet` to
`./data/omninft/vggsound/verl_omni`. Local JSONL paths can be supplied with `--train_file` and `--val_file`. Each row
keeps the joint generation prompt, stable prompt-group `uid`, separate video/audio reward prompts, and native source
metadata. The standard `RLHFDataset` reads these files.

## Prepare model assets

Download the LTX-2.3 model with the Hugging Face CLI. The destination below
matches the launcher's default model path. To use an existing checkpoint,
set `MODEL_PATH` to its local directory instead.

```bash
hf download diffusers/LTX-2.3-Diffusers \
  --local-dir ./outputs/LTX-2.3-Diffusers
```

Install the optional reward dependencies and download the reward checkpoints
and their Qwen2-VL base models. Run these commands in the training environment;
set `REWARD_ROOT` to another directory if needed.

```bash
pip install audiobox_aesthetics==0.0.4 qwen-vl-utils==0.0.14 timm==1.0.27
export REWARD_ROOT="$PWD/outputs/omninft-rewards"

hf download KlingTeam/VideoReward \
  --local-dir "$REWARD_ROOT/VideoReward"
hf download MizzenAI/HPSv3 \
  --local-dir "$REWARD_ROOT/HPSv3"
hf download facebook/audiobox-aesthetics \
  --local-dir "$REWARD_ROOT/audiobox-aesthetics"
hf download laion/clap-htsat-unfused \
  --local-dir "$REWARD_ROOT/checkpoints/clap-htsat-unfused"
hf download zghhui/OmniNFT-Reward-Series synchformer/synchformer_state_dict.pth \
  --local-dir "$REWARD_ROOT"
hf download Qwen/Qwen2-VL-2B-Instruct \
  --local-dir "$REWARD_ROOT/Qwen2-VL-2B-Instruct"
hf download Qwen/Qwen2-VL-7B-Instruct \
  --local-dir "$REWARD_ROOT/Qwen2-VL-7B-Instruct"
```

DeSync also imports Synchformer code from a trusted OmniNFT checkout. Prepare
it at the revision used by this recipe:

```bash
git clone https://github.com/zghhui/OmniNFT.git "$REWARD_ROOT/OmniNFT-reference"
git -C "$REWARD_ROOT/OmniNFT-reference" checkout --detach fb9237f6e74edf0d0f2a683f4d975b79fde588fe
```

For an existing checkout at this revision, set `DESYNC_SOURCE_ROOT` to its
location. Review the licenses of the reward repositories and checkpoints before use.

## Launch

```bash
WANDB_MODE=offline \
bash examples/omninft_trainer/ltx2/run_ltx2_3_omninft_lora_npu.sh
```

The example uses a batch size of 32, eight rollouts per prompt, 256 generated
samples per step, four TP=4 rollout replicas with one sample per
replica, LoRA rank/alpha 32/64, and a learning rate of `3e-5`. Rollout CPU
offload is disabled. Actor mini/micro batch sizes are 32/8. Validation runs
every 10 steps with 16 prompts, and checkpoints are saved every 50 steps.
Override `DATA_DIR`, `MODEL_PATH`, `REWARD_ROOT`, or `OUTPUT_DIR` for local
paths. The recipe sets `actor_rollout_ref.rollout.max_num_seqs=1`.
Training batches and eight rollouts per prompt use separate engine requests.

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

## Reward routing

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

### Scorer configuration

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
