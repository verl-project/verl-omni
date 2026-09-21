# Qwen-Image FlowGRPO

Last updated: 09/17/2026

See the [FlowGRPO trainer guide](../../../docs/examples/flowgrpo_trainer.md) for installation, OCR data and
reward-model setup.

## Regional compilation

For FSDP2 with `ulysses_sequence_parallel_size=1`, regional compilation can
compile the repeated Qwen-Image transformer blocks to improve training
performance. The benchmark recipe enables the configuration validated with
FA3:

```bash
bash examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_fsdp2_benchmark.sh
```

Set `actor_rollout_ref.model.use_regional_compile=true` to enable the feature.
The benchmark uses `fullgraph=false` because the current FA3 path requires a
graph break, and `dynamic=true` for prompt-dependent input shapes. Append
`actor_rollout_ref.model.use_regional_compile=false` to the benchmark command
to compare against eager execution. Other compiler options can be customized
through `actor_rollout_ref.model.regional_compile_options`; see the
[configuration reference](../../../docs/examples/config.md#actor_rollout_refmodel--diffusionmodelconfig)
for all defaults and current constraints.

## Optional timestep input staging

For Qwen-Image (`QwenImagePipeline`) FlowGRPO or DiffusionNFT with FSDP/FSDP2 on
GPU and `ulysses_sequence_parallel_size=1`, the default-off actor option can
reduce the number of trajectory inputs resident on the device:

```bash
bash examples/flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora.sh \
  actor_rollout_ref.actor.enable_timestep_staging=true
```

The same override works with the
[Qwen-Image DiffusionNFT recipe](../../../docs/examples/diffusionnft_trainer.md).
It keeps the caller's trajectory on CPU, copies shared prompt conditions once
per micro-batch, then transfers the current timestep's inputs. FlowGRPO stages
the current/next latent pair and matching loss fields; DiffusionNFT keeps the
clean latent and shared noise on device and stages per-step noise when present.
Consumed tensor inputs must be on CPU without gradients.

Qwen-Image FlowGRPO/DiffusionNFT on GPU with FSDP/FSDP2 and SP=1 is the validated
scope. Other models, algorithms and backends are not covered by this feature's
validation; there is no model-name or device allowlist in the shared engine.
The trainer's config validation rejects staging with sequence parallelism.
Inference does not stage inputs. Training outputs remain discarded after
backward unless explicitly requested.

Transfers are synchronous, without prefetch or an overlap guarantee. Full CPU
trajectory storage is unchanged. Measure both peak memory and update time:
this trades device memory for transfers, not a guaranteed throughput gain.
See [output lifetime and staging details](../../../docs/algo/performance.md#diffusion-actor-output-retention).
