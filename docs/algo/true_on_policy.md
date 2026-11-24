# True On-Policy for the Qwen3-Omni Thinker

Last updated: 11/23/2026.

With `actor_rollout_ref.model.true_on_policy=true`, the log-probs the vLLM-Omni rollout reports for the Qwen3-Omni thinker are bitwise equal to the log-probs the FSDP actor recomputes for the same tokens with the same weights. PPO's old policy and the sampling policy are then the same distribution, with no numerical gap to correct for.

## What 0-diff means

For every training step, over the valid response tokens only:

- rollout and actor use the same weight version, the same token history and the same sampling distribution;
- the raw log-probs have the same dtype, are all finite, and differ in zero elements. No `exp`, no tolerance.

The check is repeated after a real optimizer update and weight sync. verl reports it as `training/rollout_logprobs_mismatch_count` (with `training/rollout_logprobs_diff_max` and `_mean`); verl's `training/rollout_probs_diff_*` compare probabilities after `exp` and hide 1-ULP differences.

## Enable

```bash
python3 -m verl_omni.trainer.main_omni \
    actor_rollout_ref.model.true_on_policy=True \
    actor_rollout_ref.actor.fsdp_config.batch_invariant=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.model.lora_rank=0 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode=ar \
    +actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name=qwen3_omni_moe \
    [other config overrides...]
```

## Contract

The trainer refuses to start unless all of these hold, and lists every violation in one error:

| Setting | Required |
|---|---|
| `actor_rollout_ref.rollout.temperature` | `1.0` |
| `actor_rollout_ref.rollout.top_p` | `1.0` |
| `actor_rollout_ref.rollout.top_k` | `-1` |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `1` |
| `actor_rollout_ref.rollout.calculate_log_probs` | `true` |
| `actor_rollout_ref.rollout.enable_prefix_caching` | `false` |
| `actor_rollout_ref.model.lora_adapter_path` | `null` |
| `actor_rollout_ref.model.use_remove_padding` | `true` |
| `actor_rollout_ref.actor.strategy` | `fsdp` or `fsdp2` |
| `actor_rollout_ref.actor.fsdp_config.batch_invariant` | `true` |
| `actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size` | `1` |
| `trainer.v1.trainer_mode` | `omni_sync` |

`actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name` must name a rollout pipeline that supports the mode (`qwen3_omni_moe`); the rollout server refuses to start without one, because nothing else would turn on `hf_numerics`. Prefix caching is required off because every validated run had it off.

With LoRA (`actor_rollout_ref.model.lora_rank > 0`) three more settings are required:

| Setting | Required |
|---|---|
| `actor_rollout_ref.model.lora.merge` | `true` |
| `actor_rollout_ref.model.target_parameters` | `null` |
| `actor_rollout_ref.actor.strategy` | `fsdp2` |

`actor_rollout_ref.model.override_config.attn_implementation` may be left unset or set to `hub_fa3`, and `override_config.experts_implementation` unset or set to `vllm_fused_moe`; anything else is an error. CUDA graphs on the rollout (`enforce_eager=false`) are supported.

`actor_rollout_ref.actor.fsdp_config.batch_invariant=true` also exports `VLLM_BATCH_INVARIANT=1` to the rollout. Do not add `actor_rollout_ref.rollout.full_determinism=true` for this: it seeds every request identically, so the `n` samples of a prompt come out the same and group-relative advantages vanish.

`algorithm.rollout_correction.bypass_mode=true` is allowed. It skips the actor recompute and uses the rollout log-probs as the old policy, which is equivalent once they are bitwise equal. The mismatch metric needs the recompute, so validate a setup with `bypass_mode=false` first.

## How it works

Each source of difference is fixed on the side that owns it. The HF layers keep their own `forward`.

| Piece | Where | What |
|---|---|---|
| attention | verl-omni | `hub_fa3`: an attention function registered with Transformers' `AttentionInterface`. It derives `cu_seqlens` from the packed position ids and calls the FlashAttention-3 varlen kernel from the `kernels` hub (`kernels-community/vllm-flash-attn3`), which is the kernel vLLM runs on Hopper and brings its own backward. `OmniModelConfig` selects it automatically |
| MoE experts | verl-omni | `vllm_fused_moe`: an experts implementation registered with Transformers' `ExpertsInterface`. Its output is vLLM's `fused_experts` kernel, the one the rollout runs, which is batch-invariant under `VLLM_BATCH_INVARIANT=1`; the kernel has no backward, so the gradient is taken from Transformers' `grouped_mm` experts evaluated on the same inputs. `OmniModelConfig` selects it automatically |
| rope frequencies | verl-omni | the thinker's text rotary embedding is replaced with one that multiplies positions and fp32 inverse frequencies elementwise, as the rollout does. It keeps the frequencies in fp32 when `fsdp_config.model_dtype=bfloat16` casts the model, and avoids the fp32 matmul some Transformers releases use, which the batch-invariant kernels run at TF32 |
| RMSNorm, residual add, rope, MoE routing | vLLM-Omni | thinker-stage option `additional_config: {hf_numerics: true}`: HF operation order and rounding inside torch custom ops, so torch.compile cannot re-fuse them and they are still captured into CUDA graphs. The experts stay on `fused_experts`. verl-omni writes the option into the thinker stage of the generated deploy config |
| batch-invariant actor ops, log-prob formula | verl | `actor_rollout_ref.actor.fsdp_config.batch_invariant`: installs vLLM's batch-invariant kernels in the actor and computes log-probs with the vLLM sampler formula (fp32 `log_softmax` + gather); verl-omni's entrypoint exports `VLLM_BATCH_INVARIANT=1` to the rollout under the same flag |
| LoRA | verl-omni | with `lora.merge=true` the rollout runs merged weights `W + B @ A * scaling` and has no LoRA ops. The actor's LoRA linears forward with the same merged weight instead of PEFT's `base(x) + B(A(x)) * scaling`, and the gradient still reaches `A` and `B`. Export and forward share one merge: the whole base weight and adapter factors, in the forward dtype, multiplied on the training device. FSDP mixed precision hands the forward bf16 copies of fp32 adapters, and a bf16 matmul rounds differently on CPU than on GPU, so merging the fp32 factors, or shards that are offloaded to CPU, gives the rollout different weights |
| gate metric | verl | `training/rollout_logprobs_mismatch_count` |

Dense GEMMs, the embedding, `lm_head` and synced weights need no change: they came out bitwise equal on both sides in every gate and full-depth run. That is a measured result at the pinned versions, not something the kernels guarantee, so rerun the gate after a vLLM, CUDA or cuBLAS upgrade.

## Scope

Supported: Qwen3-Omni thinker, text-only prompts, full-parameter training or LoRA on linear layers merged on the actor side, rollout TP=1, FSDP/FSDP2 actor (FSDP2 with LoRA), the synchronous trainer, Hopper GPUs.

Not supported:

- LoRA that the rollout applies as an adapter (`lora.merge=false`), LoRA on the fused MoE experts (`target_parameters`), DoRA and other LoRA variants, and adapters loaded from `lora_adapter_path`.
- Multimodal inputs and the vision/audio towers. The actor raises a `ValueError` on the first batch that carries image, video or audio features.
- TP or EP > 1 on the rollout; the vLLM-Omni option refuses them.
- The separate-async trainer (`omni_separate_async`): its rollout replicas can generate with weights the actor has already updated.
- Run-to-run reproducibility of training: only the rollout-vs-actor log-probs within a run are covered.

## Cost

Most of the cost is on the rollout, and most of that comes from vLLM's batch-invariant mode, which `fsdp_config.batch_invariant` turns on for the rollout as well, rather than from the HF-numerics ops. On the 48-layer thinker (one H20, 32 requests of 512 tokens, CUDA graphs) the rollout ran at 2059 tok/s stock, 1128 tok/s under batch invariance and 992 tok/s with `hf_numerics` on top. Batch invariance replaces the cuBLAS GEMMs with a Triton matmul and slows the fused MoE kernel; the HF-order RMSNorm, rope and routing run as many small unfused kernels.

The actor runs the experts twice per MoE layer when it needs gradients, once through `fused_experts` for the output and once through `grouped_mm` for the gradient.

With LoRA the actor rebuilds the merged weight of every LoRA linear in each forward, and the rollout receives full merged weights at every sync, as `lora.merge=true` does without the mode.

`algorithm.rollout_correction.bypass_mode=true` takes the old log-prob recompute out of every step. Keep the mode off for runs that do not need bitwise equality.

## Verify

```bash
bash tests/special_e2e/run_qwen3_omni_true_on_policy_gate.sh
```

Runs a tiny random Qwen3-Omni thinker on 2 Hopper GPUs, with a bf16 actor as in the recipe. It checks `training/rollout_logprobs_mismatch_count == 0` at steps 1 and 2 with a nonzero `actor/grad_norm`, and `actor/old_logprobs_mismatch_count == 0` (the update forward reproduces the recomputed log-probs token for token) with a finite, nonzero `actor/grad_norm`; then reruns step 1 with `bypass_mode=true`, where the same count compares the update forward with the rollout log-probs used as the old policy. A third run trains merged LoRA on the attention projections with fp32 adapters and CPU-offloaded parameters, and applies the first run's checks at steps 2 and 3; at step 1 the LoRA `B` factors are still zero. The tiny model does not show the CPU-versus-GPU rounding of the LoRA merge; `tests/workers/test_lora_export_merge_gpu.py` checks it at the size of a real projection.

CPU tests:

```bash
pytest tests/utils/test_config_on_cpu.py tests/models/test_qwen3_omni_hub_fa3_attention_on_cpu.py \
    tests/models/test_qwen3_omni_vllm_fused_moe_experts_on_cpu.py tests/models/test_qwen3_omni_exact_mrope_on_cpu.py \
    tests/pipelines/test_qwen3_omni_true_on_policy_contract_on_cpu.py tests/pipelines/test_qwen3_omni_thinker_adapter_on_cpu.py \
    tests/workers/config/test_omni_config_on_cpu.py tests/workers/rollout/rollout_vllm/test_vllm_omni_strategy_on_cpu.py
```

`test_qwen3_omni_true_on_policy_contract_on_cpu.py` holds the cross-repository contract on every CPU run: the vLLM-Omni RMSNorm, residual add, rope and MoE routing ops are compared bitwise with the Transformers modules of the actor, and the vLLM, vLLM-Omni and verl interfaces the mode calls are checked at the pinned versions. Rerun the gate above after a pin bump that touches any of them; the CPU job cannot see kernel-level differences.

GPU tests of the experts implementation against vLLM's kernel and the `grouped_mm` backward, and of the LoRA merge shared by export and forward:

```bash
pytest tests/models/test_qwen3_omni_vllm_fused_moe_experts_gpu.py tests/workers/test_lora_export_merge_gpu.py
```
