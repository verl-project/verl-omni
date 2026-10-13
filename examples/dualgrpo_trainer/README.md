# DualGRPO Trainer

Last updated: 10/12/2026

This example post-trains `Qwen-Image` with **DualGRPO** on reasoning-heavy text-to-image prompts from [R2I-Bench](https://github.com/PLUM-Lab/R2I-Bench/tree/main/data/prompts). Rollout uses `vllm-omni`. The LLM text encoder and the diffusion transformer (DiT) are updated together from image-grounded rewards.

DualGRPO is the reinforcement-learning stage of [Think-Then-Generate: Reasoning-Aware Text-to-Image Diffusion with LLM Encoders](https://openreview.net/forum?id=RAULvuDNNP) ([arXiv:2601.10332](https://arxiv.org/abs/2601.10332)). In that paradigm, the LLM encoder reasons about a raw user prompt and rewrites it; embeddings of the rewritten prompt condition the DiT. The paper first activates this think-then-rewrite pattern with supervised fine-tuning. This directory runs the DualGRPO stage that follows. The data preprocessor inserts the think-then-rewrite system prompt used at rollout; it does not run that SFT stage.

For the shared FlowGRPO denoising loss used by the DiT update, see [Algorithms - Flow-GRPO](../../docs/algo/flowgrpo.md). For environment setup, see the [installation guide](../../docs/start/install.md). Async reward placement is described in [Async Reward](../../docs/algo/async_reward.md).

## Installation

Follow the [installation guide](../../docs/start/install.md) to set up the base environment, including `vllm-omni`.

The launch scripts target a single node. Colocated scripts (`run_qwen_image.sh`, `run_qwen_image_lora.sh`) put the actor, rollout, and UnifiedReward worker on one `NUM_GPUS` pool. Async scripts (`run_qwen_image_async_reward.sh`, `run_qwen_image_lora_async_reward.sh`) split that into an actor/rollout pool and a reward pool. Visible GPUs must cover both pools.

## How DualGRPO is wired

Qwen-Image is trained as a composite policy: Qwen2.5-VL writes a reasoning trace and a revised prompt, then MMDiT renders that prompt into images. Set `trainer.train_ar=True` so both stages receive a policy-gradient update. The trainer then:

1. Repeats each prompt `actor_rollout_ref.rollout.m` times and samples one reasoning trace per copy (`composite_single_turn_agent`).
2. For each rewritten prompt, samples `actor_rollout_ref.rollout.n` images with a FlowGRPO SDE window (`sde_type=sde`).
3. Scores every image. The sub-reward whose config key is `ar` is averaged across the `n` images of a trace and becomes the encoder outcome reward. The weighted sum of `reward.reward_functions.*.weight` becomes the DiT outcome reward.
4. Estimates group-relative advantages with `algorithm.adv_estimator=flow_grpo`, then updates the encoder on its `m` traces and the DiT on the `m × n` image trajectories with `actor_rollout_ref.actor.diffusion_loss.loss_mode=dual_grpo`.

In the paper, one prompt yields \(J\) reasoning traces and \(K\) images per trace. These scripts set \(J\) with `rollout.m` and \(K\) with `rollout.n`.

The DiT loss follows FlowGRPO. The encoder update is a GRPO-style token update on the reasoning trace, using the mean image reward of that trace.

## Prepare the dataset

Download the R2I-Bench prompt CSVs from:

- https://github.com/PLUM-Lab/R2I-Bench/tree/main/data/prompts

Place them so each file lives at:

```text
$WORKSPACE/data/r2i_bench/prompts/<category>/<category>_<subcategory>.csv
```

`WORKSPACE` defaults to `$HOME`. The converter reads the categories listed in [`data_process/r2i_bench.py`](data_process/r2i_bench.py): commonsense, compositional, concept mixing, logical, numerical, mathematical, and causal. A missing CSV is skipped.

Convert to parquet from the repository root:

```bash
python3 examples/dualgrpo_trainer/data_process/r2i_bench.py \
  --input_dir $WORKSPACE/data/r2i_bench/prompts/ \
  --output_dir $WORKSPACE/data/r2i_bench/qwen_image/
```

This writes:

- `$WORKSPACE/data/r2i_bench/qwen_image/train.parquet`
- `$WORKSPACE/data/r2i_bench/qwen_image/test.parquet`

Each subcategory is split 80/20 into train and test. The split is deterministic unless you pass `--shuffle`. Each row stores the raw prompt as the user message, prepends the DualGRPO prompt-optimizer system prompt (reason, then emit `Revised Prompt:`), and keeps the raw prompt as `reward_model.ground_truth` for scoring.

## Prepare the models

**Policy model (Qwen-Image).** The script uses the Hugging Face Hub ID `Qwen/Qwen-Image` directly — no manual download is required. Hugging Face will cache the weights automatically on first run. To use a local copy instead, edit the `model_name` variable in the script. The text encoder is Qwen2.5-VL. Training here is text-only, so the vision tower is removed before FSDP wrap.

Dual rewards are applied for the AR (LLM) and DiT stages:

- **DiT reward model (UnifiedReward 2.0).** The script defaults to the Hugging Face Hub ID `CodeGoat24/UnifiedReward-2.0-qwen3vl-8b`. No manual download is required — Hugging Face will cache the weights on first run. To use a local copy instead, edit the `DIT_REWARD_MODEL_NAME` variable in the script.

  UnifiedReward scores Alignment, Coherence, and Style on a 1–5 scale. Axes outside that range are treated as a malformed reply. The DiT reward is the mean of the parsed axes, normalized to `[0, 1]`. A reply that cannot be parsed contributes score `0` and still returns the same reward fields.

- **AR reward model (PickScore).** The PickScore reward calls `compute_score_pickscore`. The first call downloads:

  - `yuvalkirstain/PickScore_v1`
  - `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`

  Its config weight is `0.0`, so PickScore stays out of the DiT combined score. The trainer still reads the raw score stored under `reward/ar`, averages it over the `n` images of each reasoning trace, and uses that mean as the AR reward. Keep the reward-function key named `ar`; the trainer looks up `reward/ar` by that name.

## Run training

Launch from the repository root.

**Full weight, colocated reward** (`NUM_GPUS` defaults to 2):

```bash
bash examples/dualgrpo_trainer/qwen_image/run_qwen_image.sh
```

**LoRA, colocated reward** (`lora_rank=8`, `lora_alpha=16`, `exclude_modules=".*visual.*"`, `lora.merge=True`; `NUM_GPUS` defaults to 4):

```bash
bash examples/dualgrpo_trainer/qwen_image/run_qwen_image_lora.sh
```

**Full weight, async reward** (actor/rollout pool defaults to 4 GPUs, reward pool to 2):

```bash
bash examples/dualgrpo_trainer/qwen_image/run_qwen_image_async_reward.sh
```

**LoRA, async reward** (same pools and the same LoRA settings as the colocated LoRA script):

```bash
bash examples/dualgrpo_trainer/qwen_image/run_qwen_image_lora_async_reward.sh
```

AR and DiT must use the same mode: both full weight, or both LoRA. Do not attach LoRA to only one of them.

Extra Hydra overrides can be appended on the command line. Example: `bash examples/dualgrpo_trainer/qwen_image/run_qwen_image_lora.sh trainer.total_training_steps=100`.

### Environment variables

| Variable | Colocated scripts | Async scripts | Description |
| --- | --- | --- | --- |
| `WORKSPACE` | `$HOME` | `$HOME` | Root for parquet files and local model directories |
| `NUM_GPUS` |`4` | fallback for the actor pool | Devices shared by actor, rollout, and optional the colocated UnifiedReward worker |
| `NUM_GPUS_ACTOR_ROLLOUT_REWARD` | `NUM_GPUS`, else `4` | - | Actor and rollout and reward shared GPUs |
| `NUM_GPUS_ACTOR_ROLLOUT` | — | `NUM_GPUS`, else `4` | Actor and rollout GPUs |
| `NUM_GPUS_REWARD` | — | `2` | Dedicated UnifiedReward GPUs |
| `NUM_NODES` | `1` | `1` | Ray nodes |
| `MAX_NUM_SEQS` | `256` | `256` | vLLM-Omni `max_num_seqs` |

`data.train_files` and `data.val_files` are `$WORKSPACE/data/r2i_bench/qwen_image/{train,test}.parquet`.

On the async scripts, `NUM_GPUS` only fills `NUM_GPUS_ACTOR_ROLLOUT` when that variable is unset. `NUM_GPUS_REWARD` stays `2` unless you set it. `reward.reward_functions.ar.device` is `cuda:0` on the colocated scripts and `cuda:4` on the async scripts, matching a 4-GPU actor pool followed by the reward pool. If you change the actor pool size, point that device at a GPU in the reward pool.

### Shared trainer settings

Each script runs `python3 -m verl_omni.trainer.main_diffusion` with:

- `algorithm.adv_estimator=flow_grpo` both AR and DiT use a GRPO-style advantage estimator
- `actor_rollout_ref.model.model_type=diffusion_composite_model` two-stage AR + DiT rollout pipeline
- `actor_rollout_ref.model.algorithm=dual_grpo` 
- `actor_rollout_ref.model.use_remove_padding=True` AR sp must apply
- `actor_rollout_ref.actor.strategy=fsdp2` with parameter and optimizer offload, `model_dtype=bfloat16`, and `use_dynamic_bsz=False`
- `actor_rollout_ref.actor.optim.lr=3e-5` for the DiT and `actor_rollout_ref.model.ar.optim.lr` for the text encoder, with `weight_decay=0.0001`. You can change AR optimizer by setting `actor_rollout_ref.model.ar.optim.*`, otherwise AR uses DiT optimizer configs.
- `actor_rollout_ref.actor.diffusion_loss.loss_mode=dual_grpo` for DualGRPO loss compute
- `actor_rollout_ref.rollout.name=vllm_omni` set rollout engine
- `actor_rollout_ref.rollout.agent.default_agent_loop=composite_single_turn_agent` for two-stage rollout
- `actor_rollout_ref.rollout.m=4` and `actor_rollout_ref.rollout.n=4` are number of AR traces and DiT samples per trace
- `actor_rollout_ref.rollout.pipeline.height=512` and `width=512`, `true_cfg_scale=1.0`
- `actor_rollout_ref.rollout.algo.sde_type=sde`, `noise_level=1.2`, `sde_window_size=2`, `sde_window_range=[0,5]`
- `reward.reward_manager.name=MultiVisualRewardManager` compute AR and DiT rewards
- `+trainer.train_ar=True` train the AR text encoder

Training samples per step are `train_batch_size × rollout.m × rollout.n` images and `train_batch_size × rollout.m` reasoning traces. The full-weight colocated scripts use `train_batch_size=4` (64 images, 16 traces). The other async scripts use `train_batch_size=8` (128 images, 32 traces).

The composite actor supports only two weight modes. AR and DiT must both train full weights, or both train LoRA. LoRA uses one config for both actors:
LoRA training configs are shared between AR and DiT:
- `actor_rollout_ref.model.lora_rank=8`
- `actor_rollout_ref.model.lora_alpha=16`
- `actor_rollout_ref.model.exclude_modules=".*visual.*"`
- `actor_rollout_ref.model.lora.merge=True`: required. The vLLM-Omni rollout engine cannot update AR (text encoder) LoRA weights yet, so both actors sync merged full weights.

### Dual reward config

Similar to the paper's VLM-as-a-judge setup, PickScore measures semantic consistency for the AR stage. UnifiedReward scores Alignment, Coherence, and Style for the DiT stage. `MultiVisualRewardManager` computes both. Colocated scripts set `reward.reward_model.enable_resource_pool=False` and share the actor GPUs. Async scripts set `enable_resource_pool=True` so rollout agent loop can score a finished image while later samples are still generating. See [Async Reward](../../docs/algo/async_reward.md).

```bash
REWARD_ENGINE=vllm
DIT_REWARD_MODEL_NAME=CodeGoat24/UnifiedReward-2.0-qwen3vl-8b

# Enable reward model for DiT reward
reward.reward_model.enable=True
reward.reward_model.enable_resource_pool=False
reward.reward_model.model_path=$DIT_REWARD_MODEL_NAME
reward.reward_model.rollout.name=$REWARD_ENGINE

# Apply multiple rewards
reward.custom_reward_function.path=pkg://verl_omni.reward_loop.reward_manager.multi
reward.custom_reward_function.name=_multi_reward_placeholder
reward.reward_manager.name=MultiVisualRewardManager
reward.reward_manager.module.path=pkg://verl_omni.reward_loop.reward_manager
"+reward.reward_functions.ar.path=pkg://verl_omni.utils.reward_score.pickscore_reward"
'+reward.reward_functions.ar.name=compute_score_pickscore'
'+reward.reward_functions.ar.weight=0.0' # must stay zero
"+reward.reward_functions.dit.path=pkg://verl_omni.utils.reward_score.unified_reward"
'+reward.reward_functions.dit.name=compute_score_unified_reward'
'+reward.reward_functions.dit.weight=1.0' # combined score equals the DiT score
```

## Logging

The scripts log to console and TensorBoard by default. To also log to W&B, set your API key and add `"wandb"` to `trainer.logger`:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
trainer.logger='["console", "tensorboard", "wandb"]'
trainer.project_name=dual_grpo
trainer.experiment_name=qwen_image_dualgrpo
```

Validation images are written when `trainer.test_freq` is reached. `trainer.log_val_generations=8`.

See the [Metrics Documentation](../../docs/start/metrics.md) for diffusion training metrics. With `trainer.train_ar=True`, encoder metrics are logged alongside the DiT metrics.

## Variants

| Variant | Script | Default GPUs | Notes |
| --- | --- | --- | --- |
| Full weight | `examples/dualgrpo_trainer/qwen_image/run_qwen_image.sh` | `NUM_GPUS=4`, shared | SP across all GPUs, rollout TP=2, reward TP across those GPUs, UnifiedReward 2B, `gpu_memory_utilization=0.3`, PickScore on `cuda:0`. |
| LoRA | `examples/dualgrpo_trainer/qwen_image/run_qwen_image_lora.sh` | `NUM_GPUS=4`, shared | SP across all GPUs, rollout TP=2, reward TP across those GPUs, Rank 8 / alpha 16, UnifiedReward 2B, `gpu_memory_utilization=0.3`, PickScore on `cuda:0` |
| Full weight + async reward | `examples/dualgrpo_trainer/qwen_image/run_qwen_image_async_reward.sh` | 4 actor/rollout + 2 reward | Dedicated reward pool, reward TP=2 so one reward worker, UnifiedReward 8B, `gpu_memory_utilization=0.6`, PickScore on `cuda:4` |
| LoRA + async reward | `examples/dualgrpo_trainer/qwen_image/run_qwen_image_lora_async_reward.sh` | 4 actor/rollout + 2 reward | Same LoRA config as the colocated LoRA script. Reward TP=1, so two reward workers. UnifiedReward 8B, `gpu_memory_utilization=0.6`, PickScore on `cuda:4` |


