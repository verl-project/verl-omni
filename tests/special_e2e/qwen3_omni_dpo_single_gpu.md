# Single-GPU Qwen3-Omni offline DPO validation

The smoke recipe defaults trainable LoRA weights to FP32 via `LORA_DTYPE`.
The BF16 default failed when FSDP2 assigned reduced FP32 gradients to BF16 adapter parameters.
The model forward precision remains controlled separately by its existing configuration.

```bash
NUM_GPUS=1 TOTAL_TRAINING_STEPS=50 LORA_DTYPE=float32 \
  MODEL_PATH=/path/to/qwen3-omni-tiny DATA_DIR=/path/to/dpo-data \
  bash tests/special_e2e/run_qwen3_omni_multimodal_offline_mllm_dpo_lora_smoke.sh \
  trainer.total_epochs=100 trainer.save_freq=10
```

Validated with the existing tiny checkpoint builder and image/video/audio datasets,
FSDP2 ref-in-actor, one NVIDIA L20, torch2.13.0+cu130, Transformers5.14.1,
and verl revision fefb080262e1c015a0ea05f958822a6a512dc795.
All50 training steps completed; all logged metrics were finite and gradient norms
were nonzero (3.472–3.918). Image/video/audio validation ran each step.
Checkpoints at steps10,20,30,40,50 contained24 finite LoRA tensors;
all24 changed between every adjacent pair. Runtime298.41s including initialization
and cleanup. This is a tiny-model execution test, not a convergence or model-quality
claim. The recipe's commented post-training judge generation was not exercised.
