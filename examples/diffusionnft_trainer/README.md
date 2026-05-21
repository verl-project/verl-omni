# DiffusionNFT Qwen-Image OCR LoRA Example

This example runs DiffusionNFT with Qwen-Image LoRA rollout and the OCR reward.

Before running the OCR example, activate the repository environment and install the OCR reward dependency if it is missing:

```bash
source /mnt/andy/gitlocal/verl-omni/.venv/bin/activate
uv pip install Levenshtein
```

The default script is configured for a 4-step smoke run over 2 epochs and accepts normal Hydra overrides.
