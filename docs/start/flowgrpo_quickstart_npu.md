# Quickstart: FlowGRPO training on Qwen-Image OCR dataset with Ascend NPU

Last updated: 05/09/2026

Post-train a diffusion image generation model with FlowGRPO on Atlas 800T A2.

## Introduction

This guide launches FlowGRPO LoRA training for `Qwen-Image` OCR generation on Ascend NPU.

## Prerequisite

Prepare an Atlas 800T A2 server with 8 NPUs, and install the necessary software stack.

1. Install CANN by following the [Ascend CANN installation guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0003.html?OS=openEuler&InstallType=local).

2. Install VeRL-Omni and its dependencies as described in the [installation guide](install.md#install).

3. Install the FlowGRPO-specific reward dependency:

```bash
uv pip install Levenshtein
```

## Launch Training

Refer to [flowgrpo_quickstart](flowgrpo_quickstart.md) for details on the OCR dataset format, preprocessing commands, and general FlowGRPO task descriptions. The launch script for Ascend NPU is located at `examples/flowgrpo_trainer/run_qwen_image_ocr_lora_npu.sh`.

Run the FlowGRPO training script for Ascend NPU from the repository root:

```bash
bash examples/flowgrpo_trainer/run_qwen_image_ocr_lora_npu.sh
```

The script executes:

```bash
python3 -m verl_omni.trainer.main_diffusion
```

Checkpoints are saved to:

```bash
checkpoints/${trainer.project_name}/${trainer.experiment_name}
```

TensorBoard logs are saved to:

```bash
tensorboard_log/${trainer.project_name}/${trainer.experiment_name}
```

To enable logging with Weights & Biases (WandB), modify `examples/flowgrpo_trainer/run_qwen_image_ocr_lora_npu.sh` and set:

```bash
trainer.logger='["console", "wandb"]'
```
