# Qwen3-TTS GRPO with an audio reward

Last updated: 09/11/2026.

This example full-parameter tunes the codec-0 policy of
`Qwen/Qwen3-TTS-12Hz-0.6B-Base`. It uses verl's stock GRPO advantage,
vanilla PPO policy loss, and optional direct reference-model KL. The other
15 codec codebooks and code2wav stage remain frozen but are retained so every
candidate can be decoded and scored as audio.

The launcher follows the V1 omni-model integration guide: it calls
`verl_omni.trainer.main_omni` and expresses the recipe as CLI overrides on the
standard `omni_trainer` config, without a model-specific Trainer or config tree.

SpeechJudge-BTRM is one possible pointwise scorer. SpeechJudge's published vLLM
entry point targets the pairwise generative GRM, while BTRM uses a scalar reward
head with Transformers. This example therefore keeps reward inference behind the
generic audio HTTP protocol instead of adding a SpeechJudge-specific Trainer
path. The scorer may run in a separate environment from the Transformers 5.x
vLLM training stack.

## Algorithm background

This recipe applies the paper's TTS GRPO flow to Qwen3-TTS: grouped codec-token
rollouts are decoded, scored, converted to group-relative advantages, and
replayed with optional reference KL. It optimizes codec-0, the autoregressive
policy sequence described by the Qwen3-TTS architecture, while retaining all 16
codebooks for replay and waveform decoding. The HTTP scorer is configurable, so
this is not an exact reproduction of the paper's CER-and-NLL reward. See the
references below for the algorithm and multi-codebook design details.

## Install

Install the engine before the training stack:

```bash
uv pip install -e ".[gpu]" --torch-backend=auto
uv pip install "vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@$(cat .github/vllm_omni_pin.txt)"
uv pip install -e ".[omni,train,dev]"
uv pip install --no-deps --reinstall \
  "qwen-tts @ git+https://github.com/QwenLM/Qwen3-TTS.git@$(cat .github/qwen_tts_pin.txt)"
```

The pinned Qwen3-TTS revision is the upstream Transformers 5 support change
from Qwen3-TTS PR #360. Its package metadata requires Transformers 5.15.1 or
newer, while this repository intentionally caps Transformers at 5.14.1. The
`--no-deps` flag preserves that repository-wide cap; the `omni` extra owns the
runtime dependencies, including `torchaudio==2.11.0` to match vLLM's Torch pin,
and CI tests the exact Qwen3-TTS revision from `.github/qwen_tts_pin.txt` on this
stack. The released `qwen-tts==0.1.1` source targets Transformers 4.57.3 and
cannot be imported unchanged here. The adapter registers the upstream config
and model with `AutoConfig` and `AutoModelForTextToWaveform`; it does not carry
a local Transformers compatibility layer. The system `sox` executable is also
required by qwen-tts.

## Data

Training and validation parquet rows use the normal verl format:

```python
{
    "data_source": "tts",
    "prompt": [{"role": "user", "content": "Text to synthesize"}],
    "reward_model": {"style": "model", "ground_truth": "Text to synthesize"},
    "extra_info": {"id": "stable-id", "split": "train"},
}
```

Use disjoint prompts. The default recipe evaluates the same complete 100-row
validation parquet at step 0 and every 20 updates. It uses the rollout engine's
global seed; model-specific per-request seed derivation is intentionally outside
this integration.

The concatenated replay layout also requires one fixed speaker embedding JSON.
Generate it once with the official Qwen3-TTS Base model's
`extract_speaker_embedding` API from a 24 kHz reference recording, then reuse
the same file for the entire run.

## Audio scorer protocol

The configured endpoint receives one JSON request per candidate:

```json
{
  "protocol_version": "1",
  "waveform_f32_base64": "...",
  "num_samples": 24000,
  "sample_rate": 24000,
  "prompt": "Text to synthesize",
  "metadata": {"id": "stable-id"}
}
```

It must return `{"score": 1.25}` and may include additional scalar metrics.
The client retries only transient network, timeout, HTTP 408/429, and 5xx
failures. Missing, malformed, or non-finite results stop the run instead of
being converted to a valid zero reward.

For SpeechJudge-BTRM, deploy the official
[`AmphionTeam/SpeechJudge`](https://github.com/AmphionTeam/SpeechJudge) code and
[`RMSnow/SpeechJudge-BTRM`](https://huggingface.co/RMSnow/SpeechJudge-BTRM)
checkpoint in a separate environment, then expose its pointwise score through
this protocol. The official [`main_grm_vllm.py`](https://github.com/AmphionTeam/SpeechJudge/blob/master/infer/main_grm_vllm.py)
runs a different, pairwise generative GRM path; the BTRM entry point is
[`main_btrm.py`](https://github.com/AmphionTeam/SpeechJudge/blob/master/infer/main_btrm.py).
Pin the SpeechJudge source revision and runtime versions in the service
deployment. SpeechJudge-BTRM is licensed CC-BY-NC-4.0.

## Train

```bash
MODEL_PATH=/path/to/Qwen3-TTS-12Hz-0.6B-Base \
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/fixed-validation-100.parquet \
SPK_EMBED_PATH=/path/to/speaker.json \
SCORER_URL=http://scorer-host:18080/score \
OUTPUT_DIR=/path/to/output \
bash examples/grpo_trainer/qwen3_tts/run_qwen3_tts_grpo.sh
```

The example defaults are `B=4`, `G=8`, `lr=1e-6` with 10 warmup steps and a
constant schedule, direct `low_var_kl` with coefficient `0.12`, two GPUs, and
500 updates. These are recipe values, not algorithm requirements.
`norm_adv_by_std_in_grpo` remains at the upstream default. The actor and
reference keep persistent parameters in FP32, while FSDP uses BF16 parameters
for forward and backward computation with FP32 gradient reduction and buffers.
The actor's AdamW state therefore remains FP32, and rollout inference remains
BF16.

For a two-update implementation smoke test:

```bash
TOTAL_TRAINING_STEPS=2 TEST_FREQ=-1 SAVE_FREQ=-1 RESUME_MODE=disable \
OUTPUT_DIR=outputs/qwen3_tts_grpo_smoke \
bash examples/grpo_trainer/qwen3_tts/run_qwen3_tts_grpo.sh \
  trainer.val_before_train=false trainer.log_val_generations=0
```

This smoke proves rollout, finite audio reward, optimizer update, and
post-update weight sync only. It is not evidence that GRPO improves held-out
speech quality; that requires the complete fixed-validation curve and paired
human listening evaluation.

The CI-oriented wrapper at
[`tests/special_e2e/run_qwen3_tts_grpo_smoke.sh`](../../../tests/special_e2e/run_qwen3_tts_grpo_smoke.sh)
creates deterministic fixtures, uses an in-process CPU duration reward, and runs
two updates with a pinned tiny-random checkpoint rebuilt for the 16-codebook
actor contract.

## Hindi SFT-LoRA recipe

The Hindi launcher reuses the generic path above and changes only the published
task recipe. Its defaults follow the published 400-update experiment documented
in the
[`akashicmarga/qwen3-tts-hindi-lora-grpo`](https://huggingface.co/akashicmarga/qwen3-tts-hindi-lora-grpo)
model card without claiming a bitwise MLX reproduction.

| Setting | Value |
|---|---:|
| Initial policy and KL reference | Base model with the public Hindi SFT LoRA merged |
| Train / validation prompts | 863 / 100 IndicVoices-R Hindi Read rows |
| Prompt batch / group size | 4 / 4 |
| Maximum response | 240 codec frames |
| Learning rate | `5e-6`, 10 warmup steps |
| KL | stock `k3`, coefficient `0.08` |
| Trainable parameters | rank-8 LoRA on talker attention and MLP projections |
| Training length | 400 updates |
| Complete 100-prompt validation | step `0/20/40/.../400` |
| Checkpoints | step `20/40/.../400` |

The public SFT adapter was trained in MLX. Convert and merge it into the pinned,
non-quantized Transformers checkpoint before GRPO so both the initial policy
and frozen reference start from the same Hindi SFT weights. GRPO then creates a
new zero-initialized LoRA. The merged checkpoint contains no live SFT adapter.

Download the pinned assets:

```bash
hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --revision 5d83992436eae1d760afd27aff78a71d676296fc \
  --local-dir /path/to/qwen3-tts-base
hf download akashicmarga/qwen3-tts-hindi-lora adapters.safetensors \
  --revision a8718cac15b5a40bd4926b47d0854162ff32010a \
  --local-dir /path/to/hindi-sft-mlx
hf download openai/whisper-large-v3-turbo \
  --revision 41f01f3fe87f28c78e2fbf8b568835947dd65ed9 \
  --local-dir /path/to/whisper-large-v3-turbo
```

Create the SFT starting checkpoint:

```bash
python examples/grpo_trainer/qwen3_tts/data_process/merge_hindi_sft_lora.py \
  --base-model /path/to/qwen3-tts-base \
  --source /path/to/hindi-sft-mlx/adapters.safetensors \
  --output-dir /path/to/qwen3-tts-hindi-sft
```

The merge command checks the SHA-256 of every runtime file from the pinned Base
revision and the published adapter, all 462 adapter tensors, PEFT topology, and
checkpoint dtype. It uses the pinned Transformers 5 Qwen-TTS source directly
and adds no compatibility layer.

IndicVoices-R is gated and CC-BY-4.0. Accept its terms and authenticate before
building the pinned split:

```bash
hf auth login
python examples/grpo_trainer/qwen3_tts/data_process/prepare_indicvoices_hindi_grpo.py \
  --output-dir /path/to/indicvoices_hindi_grpo
```

The builder uses `normalized` text when present and otherwise uses `text`. It
filters the first 1,000 `Read` rows to duration 1-14 seconds and SNR at least 20
dB, which must yield 863 training prompts. It holds out the next 100 passing
rows, rejects prompt leakage, and records content and parquet hashes in
`manifest.json`.

Run Whisper outside Ray and expose it through the already-supported audio JSON
protocol. A separate process prevents its model dependencies and device
selection from leaking into reward-loop workers:

```bash
CUDA_VISIBLE_DEVICES=0 python \
  examples/grpo_trainer/qwen3_tts/whisper_cer_scorer.py \
  --model /path/to/whisper-large-v3-turbo \
  --device cuda:0 --dtype float16 --port 18080
```

In another shell, launch training with one fixed speaker embedding:

```bash
MODEL_PATH=/path/to/qwen3-tts-hindi-sft \
TRAIN_FILE=/path/to/indicvoices_hindi_grpo/train.parquet \
VAL_FILE=/path/to/indicvoices_hindi_grpo/validation.parquet \
SPK_EMBED_PATH=/path/to/speaker.json \
SCORER_URL=http://127.0.0.1:18080/score \
NUM_GPUS=2 \
bash examples/grpo_trainer/qwen3_tts/run_qwen3_tts_hindi_grpo.sh
```

For a short implementation smoke without validation:

```bash
TOTAL_TRAINING_STEPS=2 TEST_FREQ=-1 SAVE_FREQ=1 RESUME_MODE=disable \
OUTPUT_DIR=outputs/qwen3_tts_hindi_grpo_smoke \
bash examples/grpo_trainer/qwen3_tts/run_qwen3_tts_hindi_grpo.sh \
  trainer.val_before_train=false trainer.log_val_generations=0
```

The scorer returns `1 - min(1, CER)` and reports both raw `cer` and capped
`cer_capped`. CER is an ASR proxy for intelligibility, not a complete
naturalness or speaker-identity metric. A convergence claim must use the full
fixed-validation curve and paired listening checks, not the short smoke run.

## References

- Chang Liu, Ya-Jun Hu, Ying-Ying Gao, Shi-Lei Zhang, and Zhen-Hua Ling.
  [Group Relative Policy Optimization for Text-to-Speech with Large Language
  Models](https://arxiv.org/abs/2509.18798), 2025.
- Hangrui Hu et al. [Qwen3-TTS Technical
  Report](https://arxiv.org/abs/2601.15621), 2026.
- QwenLM. [Qwen3-TTS PR #360: Support Transformers
  5](https://github.com/QwenLM/Qwen3-TTS/pull/360), 2026.
- Dong Zhang et al. [SpeechAlign: Aligning Speech Generation to Human
  Preferences](https://arxiv.org/abs/2404.05600), 2024.
