# Tiny I2I FlowGRPO: single-GPU smoke validation

Tracking: [#237](https://github.com/verl-project/verl-omni/issues/237).

The existing Qwen-Image-Edit-Plus tiny checkpoint builder and FlowGRPO E2E
completed two training steps on one NVIDIA L20 (46 GB) on 2026-09-14.
This validates the image-conditioning training path with random weights;
it does not establish editing quality, convergence, or full-size model support.

## Reproduce

Install the repository's pinned GPU/training dependencies, including
`qwen-vl-utils` for image preprocessing. From the repository root:

```bash
export MODEL_PATH="${HOME}/models/tiny-random/qwen-image-edit-plus"
python3 tests/special_e2e/build_qwen_image_edit_plus_tiny_random.py \
  --output-dir "${MODEL_PATH}"

CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 \
MODEL_PATH="${MODEL_PATH}" DATA_DIR="${HOME}/data/dummy_image_edit" \
TOTAL_TRAIN_STEPS=2 IMAGE_HEIGHT=128 IMAGE_WIDTH=128 \
COND_HEIGHT=128 COND_WIDTH=128 \
ATTN_BACKEND=native ROLLOUT_ATTN_BACKEND=TORCH_SDPA \
bash tests/special_e2e/run_flowgrpo_qwen_image_edit.sh \
  actor_rollout_ref.rollout.calculate_log_probs=True
```

The builder's default hidden size is 16. The smoke script uses rank-8 LoRA,
four inference steps, an SDE window of two steps, true CFG scale 4, two
responses per prompt, and a one-image microbatch. No reward model is needed:
JPEG compressibility provides the rule reward. The tested path includes
condition-image dataset loading, rollout, reward, old/reference log-prob
recomputation, backpropagation, and actor-to-rollout weight synchronization.

## Recorded results

| Metric | Step 1 | Step 2 |
| --- | ---: | ---: |
| Actor gradient norm | 0.000442505 | 0.001755714 |
| Mean absolute rollout/train log-prob difference | 0.0000166148 | 0.0000761095 |
| Maximum absolute log-prob difference | 0.0000388622 | 0.0001694411 |
| Actor ratio standard deviation | 0 | 0 |

The training command exited successfully at global step 2. Per-timestep
log-prob diagnostics were present, and the final two metric records contained
no NaN or infinity. These log-prob differences are observations for this
configuration, not universal numerical tolerances.

The initial run exposed a singleton-microbatch telemetry bug: `ratio.std()`
uses a sample correction and yields NaN for one ratio. FlowGRPO now reports
the population standard deviation, as FlowDPPO already does. This changes
only the diagnostic; the loss and gradient calculation are unchanged.

Validation after the fix:

```bash
python3 -m pytest -q \
  tests/trainer/diffusion/test_flowgrpo_singleton_metrics_on_cpu.py \
  tests/trainer/diffusion/test_rollout_correction_on_cpu.py
```

Result: **23 passed**, followed by the successful two-step GPU E2E above.
The focused regression covers singleton/two-element ratio statistics and
finite, nonzero gradients.

## Tested dependency snapshot and limits

- Source baseline: `27ff29e410f0610d3a73144d8b102c389df06504`, with the
  singleton ratio-statistic fix above.
- verl: `fefb080262e1c015a0ea05f958822a6a512dc795`.
- vllm-omni: `ded8934626aaad1a3e816c3a1d9d742efc012d93`.
- vllm 0.28.0, torch 2.13.0+cu130, transformers 5.14.1, diffusers 0.40.0,
  qwen-vl-utils 0.0.14.

This is the VLM-based Qwen-Image-Edit path only; FLUX/text-encoder variants,
multiple GPUs, larger resolutions and quality curves were not validated here.
Final validation metrics were `None` because the smoke disables validation.
A diffusion-worker shutdown message appeared after the final training
metrics; the training process exited zero. This is not a warning-free
teardown claim. Completion of this smoke does not close the broader RFC.
