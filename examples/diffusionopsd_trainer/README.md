# DiffusionOPSD Trainer

Last updated: 10/19/2026

This example post-trains `stabilityai/stable-diffusion-3.5-medium` with
[DiffusionOPSD](../../docs/algo/diffusion_opsd.md) on PickScore-SFW prompts,
using `vllm-omni` rollout and PickScore as the reward. DiffusionOPSD is an online
self-distillation method: the `old` LoRA adapter rolls out deterministically,
the actor builds explicit clean-output targets by ascending/descending a
differentiable reward inside a small relative L2 trust region around an
on-policy anchor, and a detached two-branch forward-process loss fits the current
policy to those targets. There is no teacher checkpoint: the behavior policy is
the EMA of the trained policy itself.

This is the paper-v1 configuration: 512×512, 10 deterministic CFG-free rollout
steps, query noise `sigma*=0.278`, `rho=0.10`, `M_tgt=2`, `beta=1.0`, and
`c_adv=5`.

> **Status: GPU validated at 512×512.** The main evidence run uses the paper's
> resolution on two A100-80GB GPUs (one actor/rollout, one native reward): 100
> optimizer steps, group size 8, 4 prompts per update. The endpoint PickScore
> reward rose from **0.797 to 1.122 (+41%)**, peaking at 1.184, at 39.8 s per
> step after warm-up and **61.6 GB peak VRAM** on the actor/rollout GPU.
> `opsd_target_displacement_rel` stayed in [0.0688, 0.0718], inside the
> trust-region bound, with no non-finite gradients. Earlier 256×256 30-step and
> 512×512 45-step runs show the same trend. The differentiable target reward
> matched the native endpoint reward within 0.22%. Deviations from the paper:
> group size 8 instead of 24 and 4 prompts per update instead of 48.

DiffusionOPSD is **not** [DiffusionOPD](../../docs/algo/diffusion_opd.md)
(on-policy teacher distillation) and **not**
[DiffusionNFT](../../docs/algo/diffusionnft.md) (reward-weighted clean-latent
regression); see the [algorithm page](../../docs/algo/diffusion_opsd.md) for the
equations and the differences.

## Installation

Follow the [installation guide](../../docs/start/install.md) to set up the base
environment. PickScore runs through the native named-reward path
(`transformers` + `Pillow`, both base dependencies), so this example needs no
optional reward extra.

The recipe defaults to one node with 2 actor/rollout GPUs and 1 reward GPU
(`NUM_GPUS_ACTOR_ROLLOUT`, `NUM_GPUS_REWARD`). The differentiable target reward is
also loaded inside the actor engine, so keep the actor GPUs as the largest ones
available.

## Prepare the dataset

The recipe trains on PickScore-SFW (Pick-a-Pic) prompts and scores the generated
images with PickScore. Download the prompt split used by the original Flow-GRPO
project and convert it to the diffusion parquet schema:

```bash
export WORKSPACE=${WORKSPACE:-$HOME}

mkdir -p "$WORKSPACE/data/pickscore_sfw/raw"
wget -O "$WORKSPACE/data/pickscore_sfw/raw/train.txt" \
  https://raw.githubusercontent.com/yifan123/flow_grpo/main/dataset/pickscore_sfw/train.txt
wget -O "$WORKSPACE/data/pickscore_sfw/raw/test.txt" \
  https://raw.githubusercontent.com/yifan123/flow_grpo/main/dataset/pickscore_sfw/test.txt

python3 examples/flowgrpo_trainer/data_process/sd3_pickscore_sfw.py \
  --input-dir "$WORKSPACE/data/pickscore_sfw/raw" \
  --output-dir "$WORKSPACE/data/pickscore_sfw/sd3"
```

The script reads:

```bash
pickscore_train_path=$WORKSPACE/data/pickscore_sfw/sd3/train.parquet
pickscore_test_path=$WORKSPACE/data/pickscore_sfw/sd3/test.parquet
```

`sd3_pickscore_sfw.py` stores the caption as `ground_truth`; the PickScore
reward function encodes it as the text prompt.

## Run

```bash
bash examples/diffusionopsd_trainer/sd35/run_sd35_medium_opsd_lora_v1.sh
```

Additional Hydra overrides can be appended to the command; they are forwarded
after the recipe defaults.

| Variable | Default | Purpose |
|---|---|---|
| `WORKSPACE` | `$HOME` | Dataset and output root. |
| `NUM_GPUS_ACTOR_ROLLOUT` | `2` | Actor/rollout GPUs, one node. |
| `NUM_GPUS_REWARD` | `1` | GPUs in the dedicated reward pool. One native PickScore replica is placed on each. |
| `PICKSCORE_MODEL_PATH` | `yuvalkirstain/PickScore_v1` | PickScore checkpoint shared by the endpoint and differentiable rewards. |
| `N_RESP_PER_PROMPT` | `8` | Rollout group size (`actor_rollout_ref.rollout.n`). The paper v1 recipe uses 24. |
| `TRAIN_BATCH_SIZE` | `8` | Prompts per training step. The paper v1 recipe uses 48. |
| `IMAGE_RESOLUTION` | `512` | Square rollout resolution. |
| `TOTAL_TRAINING_STEPS` | `100` | Training steps. |

## Reward wiring and alignment

The endpoint reward, used only for the group-normalized loss weight `omega`
(`actor_rollout_ref.actor.diffusion_loss.adv_clip_max=5.0`,
`algorithm.global_std=True`) and for monitoring, goes through the usual reward
stack. This recipe wires the native PickScore named model:

```bash
reward.models.pickscore.backend=native
reward.models.pickscore.executor.model=verl_omni.utils.reward_score.pickscore_reward:PickScoreNativeModel
reward.reward_functions.pickscore.name=compute_score_pickscore_native
```

See [Named Reward Models](../../docs/algo/named_reward_models.md) for the
native-model lifecycle and pool/placement rules.

The **differentiable** reward used to construct the trust-region targets is a
separate module loaded by the `diffusion_opsd_model` engine, selected by
`actor_rollout_ref.actor.diffusion_loss.opsd_reward_kind=pickscore`,
`...opsd_reward_model_path` and `...opsd_reward_dtype=float32`. Only PickScore
v1 is implemented in this PR. Both rewards must load the same checkpoint and
use the same image preprocessing; `PICKSCORE_MODEL_PATH` is forwarded to both,
and `opsd_target_micro_batch=6` bounds the reward-backward batch size.

Both rewards are wired to the same checkpoint through `PICKSCORE_MODEL_PATH`.
On the smoke batch the differentiable target reward scored 0.634889 against the
native endpoint reward's 0.633521, an absolute difference of 0.001369 (0.22%
relative); the pixel preprocessing differs by design (differentiable bicubic
resize vs the PIL processor, mean pixel difference 0.005). The `omega` weights
and the targets therefore refer to the same reward signal.

## What to expect

- The endpoint training reward (`critic/rewards/mean` in this trainer; the
  policy-gradient trainers report it as `critic/score/mean`) and the validation
  reward should rise over training; they are the same PickScore signal used for
  the `omega` weights.
- Target construction returns `opsd_target_displacement_rel` and
  `opsd_target_displacement_neg_rel`
  (`mean(||y_target - y0||_2 / ||y0||_2)`). The positive displacement must lie
  in `(0, rho]` (here `(0, 0.10]`). A value pinned at `rho` means the projection
  is active on every sample, and a value near zero means the reward gradient is
  not reaching the target runtime.
- `sigma*=0.278` is resolved at batch preparation against the last four rollout
  states, and a closest state further than `opsd_query_sigma_tolerance=0.15`
  away raises instead of training on a wrong-noise state.
- `actor/opsd_pos_loss`, `actor/opsd_neg_loss`, `actor/opsd_policy_loss`,
  `actor/opsd_reward_prob_mean` and `actor/total_loss` should stay finite; the
  negative branch is the larger of the two early on.

## Configuration knobs

`actor_rollout_ref.actor.diffusion_loss.loss_mode=opsd` selects the loss; the
`opsd_*` fields live on the same block.

| Knob | Recipe value | Meaning |
|---|---|---|
| `opsd_rho` | `0.10` | Relative L2 trust-region radius; per-sample budget `rho * \|\|y0\|\|_2`. |
| `opsd_n_ascent` | `2` | Reward-gradient steps per branch (`M_tgt`). |
| `opsd_eta` | `1.0` | Fraction of the budget per ascent step before projection. |
| `opsd_branch_beta` | `1.0` | `beta` in the implicit positive/negative velocities. |
| `opsd_query_sigma` | `0.278` | Target noise level `sigma*`; the query uses the closest captured rollout step. |
| `opsd_query_sigma_tolerance` | `0.15` | Maximum allowed gap between `sigma*` and the closest captured state. |
| `opsd_num_train_timesteps` | `1000` | Scheduler timestep scale used to map timesteps to flow time; must match the SD3.5 scheduler (1000). |
| `opsd_eps_gamma` | `1e-5` | Minimum adaptive normalizer in the regression loss. |
| `opsd_reward_kind` | `pickscore` | Differentiable reward used for target construction; only PickScore v1 is implemented. |
| `opsd_reward_model_path` | PickScore checkpoint | Path/ID of the differentiable reward checkpoint; empty uses the scorer default. |
| `opsd_reward_dtype` | `float32` | Dtype of the differentiable reward model. |
| `opsd_target_micro_batch` | `6` | Samples per reward-backward micro batch during target construction. |

Fixed algorithm settings that must stay consistent with the loss:

| Knob | Recipe value |
|---|---|
| `algorithm.trainer_type` | `direct_preference` |
| `algorithm.sample_source` | `online` |
| `algorithm.old_policy_decay_schedule` | `linear_to_0_5` (`eta_beh = min(1e-3 u, 0.5)`) |
| `algorithm.old_policy_update_interval` | `1` |
| `actor_rollout_ref.model.algorithm` | `opsd` |
| `actor_rollout_ref.model.model_type` | `diffusion_opsd_model` |
| `actor_rollout_ref.model.lora_rank` / `lora_alpha` | `32` / `64` |
| `actor_rollout_ref.model.policy_state_adapters` | `["default","old"]` |
| `actor_rollout_ref.rollout.rollout_adapter` | `old` |
| `actor_rollout_ref.rollout.calculate_log_probs` | `False` |
| `actor_rollout_ref.rollout.pipeline.num_inference_steps` | `10` |
| `actor_rollout_ref.rollout.pipeline.guidance_scale` | `1.0` (CFG-free) |
| `actor_rollout_ref.rollout.algo.noise_level` | `0.0` (deterministic) |
| `actor_rollout_ref.actor.diffusion_loss.adv_clip_max` | `5.0` (`c_adv`) |
| `trainer.use_v1` / `trainer.v1.trainer_mode` | `true` / `sync` |

`ppo_mini_batch_size` equals `data.train_batch_size` and `ppo_epochs` stays at
its default of `1`, so the actor takes exactly one optimizer update per
constructed target set (`M_fit=1`).

## Tested configuration

Validated on a public 8xA100-80GB server; only two GPUs were used (one
actor/rollout, one native reward). The main evidence run is 512×512 with 100
optimizer steps; a 256×256 30-step smoke run and a 512×512 45-step run are
consistent with it.

| Item | Value |
| --- | --- |
| Hardware | 2x NVIDIA A100-SXM4-80GB (actor/rollout on one, native reward on the other) |
| Software | Python 3.10, torch 2.13.0+cu130, vllm 0.31.0, diffusers 0.40.0, transformers 5.14.1, peft 0.21.2 |
| Backbone | `stabilityai/stable-diffusion-3.5-medium` (local diffusers dir, LoRA r32/a64) |
| Trainer | v1 `sync`, `direct_preference` + `online` |
| Main run | **512×512, group 8, 4 prompts per update, 100 steps**, 10 deterministic CFG-free rollout steps, seed 42 |
| Target construction | `sigma*=0.278`, `rho=0.10`, `M_tgt=2`, `beta=1.0`, `c_adv=5`, `opsd_target_micro_batch=6` |
| Endpoint reward | PickScore v1 native named model, same checkpoint as the target reward |
| Reward curve | **0.797 → 1.122 (+41%), peak 1.184**; 256² smoke: 0.707 → 0.899; 512² 45 steps: 0.797 → 1.066 |
| Step time | 120 s first step (warm-up), then 39.8 s average |
| Peak VRAM | **61.6 GB actor/rollout GPU**, 4.8 GB reward GPU |
| Metrics | `opsd_query_sigma=0.2780`; `opsd_target_displacement_rel` in [0.0688, 0.0718] (inside `(0, rho]`); 0 non-finite-gradient steps |

Runtime knobs of the validation runs: `OPSD_MODEL_PATH` and
`PICKSCORE_MODEL_PATH` pointed at local directories,
`trainer.logger='["console"]'`, `save_freq=-1`, `test_freq=-1`,
`HF_HUB_DISABLE_XET=1` and the pyproject-pinned verl commit
`fefb080262e1c015a0ea05f958822a6a512dc795`.

The validation runs deviate from the paper v1 recipe in budget-driven ways:
group size 8 instead of 24 and 4 prompts per update instead of 48. The paper's
8-GPU, 48-group schedule remains the setting for paper-comparable numbers.
