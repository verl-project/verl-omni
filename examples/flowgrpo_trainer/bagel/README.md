# BAGEL-7B-MoT FlowGRPO training

Last updated: 10/09/2026

[BAGEL-7B-MoT](https://github.com/ByteDance-Seed/BAGEL) is a
Mixture-of-Transformers model supporting both image understanding and
generation.  Unlike Qwen-Image, BAGEL is a **non-diffusers** model — it
cannot be loaded by diffusers and uses its own weight-loading path via
``NonDiffusersModelBase``.  See
[How to Integrate a Non-Diffusers Model for FlowGRPO Training](../../../docs/contributing/integrating_a_non_diffusers_model.md)
for the integration architecture.

## Prerequisites

- Install VeRL-Omni (see [installation guide](../../../docs/start/install.md)).

- 4 GPUs or 8 NPUs. Run commands from the repository root.

- Download the checkpoint:

  ```bash
  huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT --local-dir ~/models/ByteDance-Seed/BAGEL-7B-MoT
  ```

## OCR training

We use an OCR (optical character recognition) dataset that provides
ground-truth text for evaluating image-generation quality.  Prompts are
stored in standard chat-message format for the agent loop (see
``bagel_ocr.py``).

### Prepare the dataset

Preprocess the raw OCR data into parquet:

```bash
export WORKSPACE=${WORKSPACE:-$HOME}

python3 examples/flowgrpo_trainer/data_process/bagel_ocr.py \
  --model_path ~/models/ByteDance-Seed/BAGEL-7B-MoT \
  --input_dir ~/data/ocr \
  --output_dir $WORKSPACE/data/ocr/bagel
```

This produces ``$WORKSPACE/data/ocr/bagel/train.parquet`` and
``test.parquet``.

### Run training

For GPU:
```bash
bash examples/flowgrpo_trainer/bagel/run_bagel_ocr_lora.sh
```

For NPU:  
```bash
bash examples/flowgrpo_trainer/bagel/run_bagel_ocr_lora_npu.sh
```

The launch script uses a [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
reward model with vLLM rollout (TP=4) and the ``genrm_ocr.py`` custom reward
function.

## PickScore training

PickScore evaluates image-text alignment using a
[CLIP-based model](https://huggingface.co/yuvalkirstain/PickScore_v1).  The
reward function lives entirely in ``verl_omni/utils/reward_score/pickscore_reward.py``
— there is **no** separate vLLM reward model deployment, so the GPU is shared
between the actor and the reward computation.

### Prepare the dataset

The raw PickScore dataset (``train.txt`` / ``test.txt``) should be downloaded
from the [flow_grpo repository](https://github.com/yifan123/flow_grpo/tree/main/dataset/pickscore).

Preprocess for BAGEL:

```bash
python3 examples/flowgrpo_trainer/data_process/bagel_pickscore.py \
  --model_path ~/models/ByteDance-Seed/BAGEL-7B-MoT \
  --input_dir ~/data/pickscore \
  --output_dir $WORKSPACE/data/pickscore/bagel
```

This produces ``$WORKSPACE/data/pickscore/bagel/train.parquet`` and
``test.parquet``.

### Run LoRA training

```bash
bash examples/flowgrpo_trainer/bagel/run_bagel_pickscore_lora.sh
```

Key configuration differences from OCR:
- No ``reward.reward_model.*`` flags — PickScore runs as a custom reward
  function on the rollout GPU.
- Higher ``noise_level`` (``1.3`` vs ``0.7``) and SDE window
  (``sde_window_size=2``, ``range=[0,7]``) to provide sufficient exploration
  for text-alignment learning.

### Run full-weight (non-LoRA) training

A full-weight training variant is available that trains the entire
generation pathway (``moe_gen`` parameters) while keeping the understanding
pathway frozen:

```bash
bash examples/flowgrpo_trainer/bagel/run_bagel_pickscore.sh
```

Key differences from the LoRA variant:

| Aspect | LoRA | Full-weight |
|---|---|---|
| Script | ``examples/flowgrpo_trainer/bagel/run_bagel_pickscore_lora.sh`` | ``examples/flowgrpo_trainer/bagel/run_bagel_pickscore.sh`` |
| Strategy | default | ``fsdp2`` (required for mixed ``requires_grad``) |
| Trainable params | Low-rank adapters on ``*_moe_gen`` | All ``moe_gen`` parameters (``requires_grad`` set by ``configure_trainable_params``) |
| ``lora_rank`` / ``lora_alpha`` | 64 / 128 | N/A |
| ``sde_window_size`` | 2 | 3 (more exploration for full-weight) |

**Why FSDP2 is required.** FSDP1 does not natively support mixed
``requires_grad`` within a single wrapped module — some parameters frozen,
others trainable.  FSDP2 handles this correctly and also reshards layer
parameters after forward, reducing peak memory during gradient
checkpointing.  The understanding pathway (``moe_und``) is not a LoRA
wrapper replacement but simply has ``requires_grad=False`` set by the
``configure_trainable_params`` hook.

## Key differences from Qwen-Image

| Aspect | Qwen-Image | BAGEL-7B-MoT |
|---|---|---|
| Model loading | diffusers | Custom ``from_pretrained`` via ``NonDiffusersModelBase`` |
| Architecture | Auto-detected | Explicit: ``+actor_rollout_ref.model.architecture=OmniBagelForConditionalGeneration`` |
| Deploy config | Not needed | ``bagel_deploy_config.yaml`` (single-stage topology) |
| LoRA targets | ``*_proj`` layers | ``*_proj`` + ``*_moe_gen`` (MoT dual-pathway) |
| LoRA weight sync | Adapter tensors | Merged full weights (``lora.merge=True``): the rollout engine's fused MoT layout cannot bind ``*_moe_gen`` adapters |
| FSDP prefixes | ``transformer_blocks.`` | ``layers.`` |
| CFG | Standard true CFG | 3-branch (gen / text-uncond / img-uncond) with global renormalisation |
| Timestep convention | ``t / 1000`` | Raw sigma with SD3-style shift of 3.0 |

## Further reading

- [How to Integrate a Non-Diffusers Model for FlowGRPO Training](../../../docs/contributing/integrating_a_non_diffusers_model.md) — full integration guide using BAGEL as the worked example
- [vLLM-Omni BAGEL docs](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/online_serving/bagel/)

## AlphaGRPO

The experimental single-turn [AlphaGRPO](https://github.com/huangrh99/AlphaGRPO)
GPU recipe trains BAGEL thinking and image generation together. It combines a clipped
sequence-ratio text objective with image FlowGRPO, DVReward and the thinking-tag
format reward. Multi-turn self-reflective refinement and image editing are not
supported by this adapter.

A 128px native BAGEL smoke run verified FSDP2 joint LoRA updates and merged weight
sync using controlled advantages. The full recipe with the 30B judge and quality
comparisons has not been validated.

### Data and training

Use the official `alphagrpo20k` train/test JSONL files or your own question-decomposed
prompts. Each record contains `prompt`, `semantic_questions`, and `quality_questions`.
Questions can be strings or dictionaries with a `question` field. Obtain the
official files with Git LFS; LFS pointer files are not datasets.

From the repository root:

```bash
python examples/flowgrpo_trainer/data_process/bagel_dvreward.py \
  --model_path /path/to/BAGEL-7B-MoT \
  --input_dir /path/to/alphagrpo20k \
  --output_dir /path/to/dvreward

DATA_DIR=/path/to/dvreward \
  bash examples/flowgrpo_trainer/bagel/run_bagel_alphagrpo_lora.sh \
  actor_rollout_ref.model.path=/path/to/BAGEL-7B-MoT \
  actor_rollout_ref.model.tokenizer_path=/path/to/BAGEL-7B-MoT
```

Use `run_bagel_dvreward_lora.sh` for image-only FlowGRPO with DVReward.

The AlphaGRPO wrapper trains both understanding and generation LoRA projections
with FSDP2. It inherits the existing four-GPU OCR recipe, including reward-model
placement. It selects the official launcher's Qwen3-VL-30B-A3B-Instruct judge.
This resource layout has not yet been validated with DVReward. A smaller judge
can be selected by a trailing `reward.reward_model.model_path=...` override;
that changes the reward model and is not the official reward setup.

For each question, the scorer uses first-token top-5 log-probabilities to compute
`P(yes) / (P(yes) + P(no))`. Semantic and quality questions are averaged separately,
then combined as `sqrt(semantic_score * quality_score)`. Both component scores are
returned as metrics. HTTP failures propagate; they are not converted into zero rewards.

### Gotchas

- Both question groups must be non-empty. This avoids assigning perfect scores
  to absent questions. Question records are scored in order, without filtering
  `is_valid`, matching the official scorer's behavior.
- Prompts exceeding the token limit are rejected rather than truncated, so the
  image condition still matches the questions used to score it.
- Unlike the reference's JPEG transport, this scorer uses the repository's PNG
  data-URI helper. When neither yes nor no appears in top-5 tokens, a literal
  yes/no answer is used; any other answer is rejected rather than treated as no.
- AlphaGRPO uses the reference planning system prompt. Its three image contexts
  are system + prompt + thinking, system only, and system + prompt. Replay uses
  exact native token IDs; decoding is used only for reward scoring and logging.
- Text likelihoods use the configured temperature before nucleus filtering,
  matching the reference likelihood convention. Sequence ratios stay paired with
  each sample's own advantage; broadcasting them into a cross-sample matrix would
  cancel or mix group-relative learning signals.
- Only sampled thinking actions enter the text loss. A forced end token after
  reaching the thinking budget conditions the image but is excluded from the
  text loss. The text objective runs once per micro-batch, with image-step-count
  compensation before the shared gradient-accumulation division.
- Native BAGEL records log-probs only for noisy SDE steps. Window slicing applies
  to the full latent and timestep trajectories; slicing log-probs again drops
  training steps when the window starts after zero.
- The single-turn reward adds `1` when thinking starts with `<think>` and ends
  with `</think>`, matching the reference T2I task. `dvreward`, component scores
  and `thinking_format_score` are logged separately.
- Rollout currently requires TP=CFG=SP=1 and full-request execution. Actor
  sequence parallelism and timestep staging are unsupported. Merged LoRA sync
  routes the language head separately from the transformer body.
- `pipeline.max_sequence_length` must hold the system prompt, native prompt and
  `max_think_tokens + 2` framing budget. Overflow is rejected. Negative prompts
  are unsupported because text CFG uses the planning system context.
- Image-only DVReward uses the original FlowGRPO adapter; enabling its upstream
  `think=true` path would not provide joint text training or matching replay. Use
  `model.algorithm=alphagrpo` and the AlphaGRPO recipe for thinking. This is an
  integration, not evidence of a stable quality or speed improvement.
