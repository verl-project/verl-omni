# MiniCPM-o vLLM kernel parity (experimental)

Last updated: 11/26/2026.

Opt-in mode so the FSDP2 actor and the eager vLLM-Omni rollout compute the same
log-probs for the same MiniCPM-o thinker weights and tokens. This is **kernel
alignment** on the HF Qwen3 text decoder (batch-invariant vLLM kernels, shared
LoRA merge). It is not the AVQA separate-async quality work, and it is not
[true on-policy](true_on_policy.md) for Qwen3-Omni.

## Enable

```bash
export VERL_OMNI_VLLM_PARITY=1
export VLLM_BATCH_INVARIANT=1
bash examples/gspo_trainer/minicpm/run_minicpmo_4_5_thinker_gspo_lora_gsm8k_vllm_parity_v1.sh
```

The recipe also sets `actor_rollout_ref.rollout.enforce_eager=true` and
`tensor_model_parallel_size=1`. Leave the production AVQA scripts alone: they
use CUDA graphs and TP=2 and were not measured in this mode.

## What it changes

| Piece | Where | Behavior when `VERL_OMNI_VLLM_PARITY=1` |
|---|---|---|
| Qwen3 text decoder | `MiniCPMThinkerAdapter.configure_model` → `install_qwen3_parity` | HF layers keep module identity for FSDP2; forward uses vLLM residual dataflow and batch-invariant kernels |
| LoRA merge | `lora_export._LoraMerge.merge` / actor `lora_linear` | Both call `merge_lora_weight` so sync streams the weight the actor multiplies by |
| Log-probs | `OmniFSDPEngine.prepare_model_outputs` | vLLM V1 sampler math (`fp32` logits / temperature → Triton `log_softmax` → gather) without patching the verl pin |

Off by default: unset `VERL_OMNI_VLLM_PARITY` (or any value other than `1`) and
training is unchanged.

## Scope and honesty about measurements

Supported intent: MiniCPM-o 4.5 thinker, text-only LoRA (GSM8K-style), FSDP2,
eager rollout, TP=1.

A private RSI release on older pins
(`verl-omni@3ce6010d`, `verl@b0e665e5`, `vllm-omni@9ec1efe5`) reported
0 mismatched / 169967 holdout tokens and ~0.86× step time versus the unpatched
baseline under those pins. **Those numbers are not a claim about this PR or
current main.** Pins have moved; hook sites were retargeted (LoRA sync now
goes through `export_lora_for_sync` / `_LoraMerge`, and log-probs are overridden
in `OmniFSDPEngine` instead of upstream `transformer_impl.py`). Re-verify
mismatch counts on current pins before treating a run as zero-mismatch.

## Related

- [True On-Policy](true_on_policy.md) — Qwen3-Omni full-parameter bitwise path (different mechanism)
- [Deterministic post-training](deterministic_post_training.md) — broader determinism knobs
- Recipe: `examples/gspo_trainer/minicpm/run_minicpmo_4_5_thinker_gspo_lora_gsm8k_vllm_parity_v1.sh`
