# BAGEL-7B-MoT FlowGRPO training

[BAGEL-7B-MoT](https://github.com/ByteDance-Seed/BAGEL-7B-MoT) is a
Mixture-of-Transformers model supporting both image understanding and
generation.  Unlike Qwen-Image, BAGEL is a **non-diffusers** model — it
cannot be loaded by diffusers and uses its own weight-loading path via
``NonDiffusersModelBase``.  See
[docs/contributing/integrating_a_non_diffusers_model.md](../../docs/contributing/integrating_a_non_diffusers_model.md)
for the integration architecture.

## Prerequisites

- Install VeRL-Omni (see [docs/start/install.md](../../docs/start/install.md)).

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

### Run training

```bash
bash examples/flowgrpo_trainer/bagel/run_bagel_pickscore_lora.sh
```

Key configuration differences from OCR:
- No ``reward.reward_model.*`` flags — PickScore runs as a custom reward
  function on the rollout GPU.
- Higher ``noise_level`` (``1.3`` vs ``0.7``) and SDE window
  (``sde_window_size=2``, ``range=[0,7]``) to provide sufficient exploration
  for text-alignment learning.

## Key differences from Qwen-Image

| Aspect | Qwen-Image | BAGEL-7B-MoT |
|---|---|---|
| Model loading | diffusers | Custom ``from_pretrained`` via ``NonDiffusersModelBase`` |
| Architecture | Auto-detected | Explicit: ``+actor_rollout_ref.model.architecture=OmniBagelForConditionalGeneration`` |
| Deploy config | Not needed | ``bagel_deploy_config.yaml`` (single-stage topology) |
| LoRA targets | ``*_proj`` layers | ``*_proj`` + ``*_moe_gen`` (MoT dual-pathway) |
| FSDP prefixes | ``transformer_blocks.`` | ``layers.`` |
| CFG | Standard true CFG | 3-branch (gen / text-uncond / img-uncond) with global renormalisation |
| Timestep convention | ``t / 1000`` | Raw sigma with SD3-style shift of 3.0 |

## Further reading

- [integrating_a_non_diffusers_model.md](../../docs/contributing/integrating_a_non_diffusers_model.md) — full integration guide using BAGEL as the worked example
- [vLLM-Omni BAGEL docs](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/online_serving/bagel/)
