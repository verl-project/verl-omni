# FSDP diffusion engines

Last updated: 10/09/2026

## Gotchas

- Materialize and clone exported tensors while the LoRA merge context remains
  open. `DTensor.full_tensor()` can alias local storage on one rank, and dtype
  conversion may return the same tensor. Restoring base weights otherwise
  silently changes tensors awaiting transport. BAGEL AlphaGRPO consumes weights
  one at a time to avoid retaining another complete model on the rollout GPU.
