# DiffusionART Trainer

Last updated: 10/12/2026

This example post-trains `Qwen/Qwen-Image` with
[DiffusionART](../../docs/algo/diffusionart.md) (aggregation-first reward
tilting, [arXiv:2610.12572](https://arxiv.org/abs/2610.12572)) on the OCR text
rendering task, using `vllm-omni` rollout and `Qwen/Qwen3-VL-8B-Instruct` as the
OCR reward model.

DiffusionART is an online direct-preference algorithm in verl-omni's taxonomy:
rollouts are deterministic ODE trajectories sampled with the `old` LoRA adapter,
the actor trains from the states those trajectories visited, and the `default`
adapter is updated. There is no likelihood ratio, no differentiable reward, no EMA
schedule and no KL term.

> **Status:** CPU math tests + golden equivalence to the authors' code; no GPU
> run yet. The recipes below are the planned configurations and will be updated
> with curves once they have run. See [Validation status](#validation-status).

Three recipes:

- Qwen-Image OCR (below), the repository's standard smoke task.
- LTX-2.3 text-to-audio-video,
  [`ltx2/run_ltx2_t2av_art_lora.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/diffusionart_trainer/ltx2/run_ltx2_t2av_art_lora.sh)
  (a shell script with explicit hydra overrides, like the other recipes in this repository):
  the OmniNFT LTX-2.3 setting (same VGGSound prompts, five rewards and
  routing weights, LoRA r32/alpha64, lr 3e-5, 32 prompts x 8 samples,
  256x384x121 frames, 20 rollout steps, 150 steps) with only the algorithm
  changed, for a like-for-like comparison. Guidance is computed per modality
  from the routed rewards (VideoAlign/HPSv3 -> video, AudioBox/CLAP -> audio,
  DeSync -> both); the rollout is an ODE; the training forward uses the same CFG
  as the rollout (video 1.5 / audio 3.0) because `v_old` is recomputed by it;
  `ppo_mini_batch_size=16` of 32 prompts gives the two updates per rollout. The
  native VideoAlign / HPSv3 / AudioBox / CLAP / DeSync scorers it uses are on
  main since #663 part 1; ART reads their per-component `reward/<name>` scores
  under the default `weighted_sum` aggregation, so it does not need
  `preserve_components` (still pending #664). Model and data assets: the LTX-2.3
  Diffusers checkpoint and the VGGSound caption parquet are prepared as in the
  in-repo LTX-2.3 FlowGRPO example
  ([`examples/flowgrpo_trainer/ltx2/README.md`](../flowgrpo_trainer/ltx2/README.md));
  the five native scorers load their public checkpoints from `REWARD_ROOT`
  (VideoReward, HPSv3, Qwen2-VL-2B/7B, audiobox-aesthetics, CLAP `clap-htsat-unfused`,
  Synchformer) as listed in the script header.
- MiniMax-H3 text-to-audio-video,
  [`minimax_h3/run_minimax_h3_t2va_art_lora.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/diffusionart_trainer/minimax_h3/run_minimax_h3_t2va_art_lora.sh):
  the DiffusionNFT MiniMax-H3 recipe with only the algorithm swapped (CLAP +
  ImageBind, README VidProM split, LoRA r64/alpha128, lr 3e-4, 10-step rollouts). H3
  denoises video and audio on different sigma grids, so the rollout also publishes
  `all_segment_timesteps` and the loss uses each modality's own flow time.

## Installation

Follow the [installation guide](../../docs/start/install.md), then install the OCR
reward dependency:

```bash
uv pip install -e ".[ocr]"
```

The script is configured for a single node with `4` GPUs.

## Prepare the dataset

Obtain the raw OCR dataset from the original Flow-GRPO repository
(https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr), place it under
`$WORKSPACE/data/ocr` (`WORKSPACE` defaults to `$HOME`) and preprocess it:

```bash
python3 examples/flowgrpo_trainer/data_process/qwenimage_ocr.py \
  --input_dir $WORKSPACE/data/ocr \
  --output_dir $WORKSPACE/data/ocr
```

This produces `$WORKSPACE/data/ocr/train.parquet` and `$WORKSPACE/data/ocr/test.parquet`.

## Launch

```bash
WORKSPACE=/path/to/workspace bash examples/diffusionart_trainer/qwen_image/run_qwen_image_ocr_lora.sh
```

[`qwen_image/run_qwen_image_ocr_lora.sh`](https://github.com/verl-project/verl-omni/blob/main/examples/diffusionart_trainer/qwen_image/run_qwen_image_ocr_lora.sh)
mirrors the DiffusionNFT Qwen-Image OCR recipe (data, reward, LoRA targets,
512x512, 10 rollout steps, 16 samples per prompt, lr `3e-4`) and changes only the
algorithm dispatch and the DiffusionART knobs:

```bash
actor_rollout_ref.model.algorithm=diffusion_art
actor_rollout_ref.model.model_type=diffusion_art_model
actor_rollout_ref.actor.diffusion_loss.loss_mode=diffusion_art
actor_rollout_ref.model.policy_state_adapters='["default","old"]'
actor_rollout_ref.rollout.rollout_adapter=old
actor_rollout_ref.rollout.algo.noise_level=0.0
algorithm.trainer_type=direct_preference
algorithm.sample_source=online
algorithm.old_policy_decay_schedule=copy
```

The paper's two disjoint updates per rollout batch are expressed through
`ppo_mini_batch_size = data.train_batch_size / 2` (`12` of `24` prompts here); the
trainer logs a warning for any other split so the one-update ablation stays
possible.

### Pairwise K=2 arm (paper Sec. 4.4)

`actor_rollout_ref.rollout.n=2 algorithm.art_pairwise_reward=true` turns the LTX
recipe into the paper's pairwise setting (one win/lose pair per prompt and
modality; 64 rollouts per step instead of 256). Keep `ppo_mini_batch_size=16`
(prompts) for the two updates per rollout, and note the micro-batching
constraint: each rank receives `ppo_mini_batch_size * n / dp` samples per mini
batch and the engine asserts (after the first rollout) that this is divisible by
`ppo_micro_batch_size_per_gpu`. At 8 GPUs that is `16 * 2 / 8 = 4` samples per
rank, so `ppo_micro_batch_size_per_gpu` must divide 4 (`1`, `2` or `4`; run1's
`16` fails). The trainer now checks this at start-up. ART guidance is computed per
sample, so micro batches do not have to hold whole rollout groups.

## Key knobs

| Flag | Default | Meaning |
|---|---|---|
| `actor_rollout_ref.actor.diffusion_loss.art_guidance_gain` | `1.0` | Multiplier on the unit-RMS guidance `G(c)`. The paper uses `1`. |
| `actor_rollout_ref.actor.diffusion_loss.art_loss_space` | `velocity_linear_sigma` | MSE weight `t` (paper) or `velocity_uniform` for weight `1`. |
| `actor_rollout_ref.actor.diffusion_loss.art_mask_metric` | `sde_kl` | Drift proxy for the trust-region mask: reference-SDE KL (paper Eq. 7) or `fixed_mse`. |
| `actor_rollout_ref.actor.diffusion_loss.art_mask_threshold` | `1e-5` | Drift threshold `delta` above which outward updates are masked. |
| `actor_rollout_ref.actor.diffusion_loss.art_reference_noise_level` | `0.8` | `nu` of the reference SDE used by `sde_kl`. |
| `actor_rollout_ref.actor.diffusion_loss.art_corrective_mask_enabled` | `true` | Set `false` to ablate the trust-region mask. |
| `actor_rollout_ref.actor.diffusion_loss.art_video_weight` / `art_audio_weight` | `1.0` / `1.0` | Weights of the per-modality losses (audio-video models). |
| `algorithm.art_pairwise_reward` | `false` | DiffusionART-Pairwise: random disjoint win/lose pairs replace the scalar rewards; ties become restore anchors. |
| `algorithm.art_exclude_final_step` | `true` | Drop the last ODE position from the trainable state pool (paper recipes). |
| `algorithm.art_log_rollout_velocity_gap` | `false` | Log `actor/old_rollout_gap` = mean((v_old - v_rollout)^2); doubles cached-state memory. |
| `algorithm.art_modality_reward_weights` | `null` | `{video: {reward: w}, audio: {reward: w}}` routing; `null` reads `reward_functions.*.routing_weights`. |
| `algorithm.timestep_fraction` | `1.0` | Fraction of visited ODE states trained per rollout. |
| `algorithm.art_sample_log_dir` | `null` | Write per-sample training records (`step_<N>.jsonl`: uid, raw/standardised reward components, routed rewards, group) under this directory. |

## Validation status

- CPU unit tests: `tests/trainer/diffusion/test_diffusionart_math_on_cpu.py`,
  `tests/trainer/diffusion/test_diffusionart_on_cpu.py` and the golden-value test
  `tests/trainer/diffusion/test_diffusionart_golden_on_cpu.py` (fixture generated
  from vanzll/DiffusionART @ `8652c8a`; math matches to fp32 round-off).
- GPU: **not yet run** for either recipe. No convergence curve is available.
