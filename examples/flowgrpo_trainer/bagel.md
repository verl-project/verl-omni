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

- 4 GPUs.  Run commands from the repository root.

- Download the checkpoint:

  ```bash
  huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT --local-dir ~/models/ByteDance-Seed/BAGEL-7B-MoT
  ```

## Prepare the dataset

We use the same [PickScore dataset](https://github.com/yuvalkirstain/PickScore)
as the official flow_grpo BAGEL config.  Prompts are stored in standard
chat-message format for the agent loop (see ``bagel_pickscore.py``).

First, download the raw prompt files from the flow_grpo repository:

```bash
wget -P ~/data/pickscore/ \
  https://raw.githubusercontent.com/yifan123/flow_grpo/main/dataset/pickscore/train.txt
wget -P ~/data/pickscore/ \
  https://raw.githubusercontent.com/yifan123/flow_grpo/main/dataset/pickscore/test.txt
```

Then preprocess them into parquet:

```bash
export WORKSPACE=${WORKSPACE:-$HOME}

python3 examples/flowgrpo_trainer/data_process/bagel_pickscore.py \
  --input_dir $WORKSPACE/data/pickscore \
  --output_dir $WORKSPACE/data/pickscore/bagel \
  --model_path ~/models/ByteDance-Seed/BAGEL-7B-MoT
```

This produces ``$WORKSPACE/data/pickscore/bagel/train.parquet`` and
``test.parquet``.

## Run training

```bash
bash examples/flowgrpo_trainer/run_bagel_flowgrpo_lora.sh
```

The launch script enables decoupled rollout correction
(``calculate_log_probs=true``, ``rollout_is=sequence``).

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
