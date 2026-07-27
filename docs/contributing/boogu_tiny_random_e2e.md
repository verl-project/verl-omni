# Tiny-random BOOGU e2e plan

This note records a practical end-to-end validation plan for BOOGU support under limited hardware.

## Why this exists

The real `Boogu/Boogu-Image-0.1-*` checkpoints are too large for the current WSL machine.

- Current available GPU memory: about `8 GiB`
- Official `vllm-omni` Boogu serving recipe expects roughly `34.6 GiB` on GPU and recommends a `40GB+` card for the full checkpoint

Because of that, the near-term goal is not “run the real BOOGU checkpoint”, but “run the full BOOGU-shaped pipeline with a tiny-random checkpoint”.

## Target

Build a tiny-random BOOGU checkpoint and use it to validate the following BOOGU-specific path end to end:

1. dataset/parquet load
2. BOOGU rollout request preparation
3. BOOGU prompt / negative prompt handling
4. BOOGU diffusion sampling on a tiny checkpoint
5. reward computation with a simple self-contained reward
6. training update
7. artifact generation and image saving

## Scope

This plan is intended to validate the framework path and BOOGU-specific wiring under constrained hardware.

It does **not** try to prove model quality, because tiny-random checkpoints do not carry meaningful semantics.

## Recommended validation shape

### Stage 1: tiny-random BOOGU checkpoint

Build a local checkpoint with the same directory layout as a BOOGU checkpoint:

```text
<tiny-boogu>/
├── model_index.json
├── mllm/
├── processor/
├── scheduler/
├── transformer/
└── vae/
```

Suggested component choices:

- `transformer`: tiny `BooguImageTransformer2DModel` with reduced hidden size / heads / layers
- `scheduler`: BOOGU `FlowMatchEulerDiscreteScheduler`
- `vae`: tiny `AutoencoderKL`
- `mllm`: tiny `Qwen3VLForConditionalGeneration`
- `processor`: copy only lightweight processor/tokenizer assets from a cached BOOGU source model, or a compatible tiny local substitute

### Stage 2: tiny-random BOOGU rollout adapter

Add a BOOGU rollout adapter registered under:

- `BooguImagePipeline` + `flow_grpo`
- optionally `BooguImageTurboPipeline` + `flow_grpo`

The rollout adapter should mirror the existing Qwen-Image / Bagel pattern:

- accept tokenized prompt inputs
- prepare prompt embeds and negative prompt embeds
- switch the scheduler to the SDE/logprob-aware scheduler
- return:
  - final image
  - intermediate latents
  - per-step logprobs
  - timesteps

### Stage 3: minimal e2e training loop

Run a tiny FlowGRPO smoke configuration with:

- `tensor_model_parallel_size=1`
- `n=1` or `n=2`
- very small image size, e.g. `256x256`
- `num_inference_steps=4`
- tiny parquet dataset
- a self-contained reward such as `jpeg_compressibility`
- `trainer.total_training_steps=1~2`

This avoids extra reward-model serving and keeps the validation focused on the BOOGU path.

## Minimal artifacts to save

The tiny-random BOOGU e2e run should save at least:

1. generated sample image(s)
2. a small image grid comparing the saved outputs
3. training loss / reward curve
4. the exact command line used

Recommended output layout:

```text
assets/pr_results/boogu_tiny_random_e2e/
├── generated_0.png
├── generated_1.png
├── grid.png
├── metrics.csv
└── metrics_curve.png
```

## Suggested order of work

1. keep the existing BOOGU CPU tests as the regression safety net
2. keep the existing BOOGU adapter smoke script as the adapter-level validation
3. add the tiny-random BOOGU checkpoint builder
4. add the BOOGU rollout adapter
5. add a minimal `tests/special_e2e/run_flowgrpo_boogu_tiny_random.sh`
6. run it in WSL and save the output images / curves

## Success criteria

The tiny-random BOOGU e2e path is considered validated when:

- the trainer launches successfully
- rollout returns images and logprobs
- the reward path runs without an external reward server
- the update step completes without NaN / crash
- images and metric curves are saved locally

## Notes for PR review

When reporting results in a PR, describe this as:

- BOOGU-specific CPU tests
- BOOGU adapter smoke validation
- tiny-random BOOGU end-to-end framework validation

Do **not** describe tiny-random results as real BOOGU checkpoint quality results.
