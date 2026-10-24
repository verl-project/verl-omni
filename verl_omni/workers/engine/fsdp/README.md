# FSDP diffusion engines

Last updated: 10/24/2026

## Gotchas

- The shared `workers/engine/lora_export.py` must materialize and clone exported
  tensors while the LoRA merge context remains open. `DTensor.full_tensor()` can alias local storage on one rank, and dtype
  conversion may return the same tensor. Restoring base weights otherwise
  silently changes tensors awaiting transport.
