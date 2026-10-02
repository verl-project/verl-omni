# DualGRPO Trainer

Last updated: 09/30/2026

This example post-trains `Qwen-Image` with **DualGRPO** on reasoning-heavy text-to-image prompts from [R2I-Bench](https://github.com/PLUM-Lab/R2I-Bench/tree/main/data/prompts). Rollout uses `vllm-omni`. The LLM text encoder and the diffusion transformer (DiT) are updated together from image-grounded rewards.

DualGRPO is the reinforcement-learning stage of [Think-Then-Generate: Reasoning-Aware Text-to-Image Diffusion with LLM Encoders](https://openreview.net/forum?id=RAULvuDNNP) ([arXiv:2601.10332](https://arxiv.org/abs/2601.10332)). In that paradigm, the LLM encoder reasons about a raw user prompt and rewrites it; embeddings of the rewritten prompt condition the DiT. The paper first activates this think-then-rewrite pattern with supervised fine-tuning. This directory runs the DualGRPO stage that follows. The data preprocessor inserts the think-then-rewrite system prompt used at rollout; it does not run that SFT stage.

For the shared FlowGRPO denoising loss used by the DiT update, see [Algorithms - Flow-GRPO](../../docs/algo/flowgrpo.md). For environment setup, see the [installation guide](../../docs/start/install.md).

## Installation

Follow the [installation guide](../../docs/start/install.md) to set up the base environment, including `vllm-omni`.

The launch scripts target a single node. `NUM_GPUS` defaults to `2`, and the actor, rollout, and UnifiedReward worker share those devices.

## How DualGRPO is wired

Qwen-Image is trained as a composite policy: Qwen2.5-VL writes a reasoning trace and a revised prompt, then MMDiT renders that prompt into images. Set `trainer.train_ar=True` so both stages receive a policy-gradient update. The trainer then:

1. Repeats each prompt `actor_rollout_ref.rollout.m` times and samples one reasoning trace per copy (`composite_single_turn_agent`).
2. For each rewritten prompt, samples `actor_rollout_ref.rollout.n` images with a FlowGRPO SDE window (`sde_type=sde`).
3. Scores every image. The sub-reward whose config key is `ar` is averaged across the `n` images of a trace and becomes the encoder outcome reward. The weighted sum of `reward.reward_functions.*.weight` becomes the DiT outcome reward.
4. Estimates group-relative advantages with `algorithm.adv_estimator=flow_grpo`, then updates the encoder on its `m` traces and the DiT on the `m × n` image trajectories with `actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo`.

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

- **DiT reward model (UnifiedReward 2.0).** The full-weight script defaults to the Hugging Face Hub ID `CodeGoat24/UnifiedReward-2.0-qwen3vl-4b`; the LoRA script defaults to `CodeGoat24/UnifiedReward-2.0-qwen3vl-8b`. No manual download is required — Hugging Face will cache the weights on first run. To use a local copy instead, edit the `DIT_REWARD_MODEL_NAME` variable in the script.

  UnifiedReward scores Alignment, Coherence, and Style on a 1–5 scale. The DiT reward is the mean of those axes, normalized to `[0, 1]`.

- **AR reward model (PickScore).** The PickScore reward calls `compute_score_pickscore`. The first call downloads:

  - `yuvalkirstain/PickScore_v1`
  - `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`

  Its config weight is `0.0`, so PickScore stays out of the DiT combined score. The trainer still reads the raw score stored under `reward/ar`, averages it over the `n` images of each reasoning trace, and uses that mean as the AR reward. Keep the reward-function key named `ar`; the trainer looks up `reward/ar` by that name.

## Run training

Launch from the repository root.

**Full weight:**

```bash
bash examples/dualgrpo_trainer/qwen_image/run_qwen_image.sh
```

**LoRA** (`lora_rank=8`, `lora_alpha=16`, `exclude_modules=".*visual.*"`, `lora.merge=True`):

```bash
bash examples/dualgrpo_trainer/qwen_image/run_qwen_image_lora.sh
```

### Environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `WORKSPACE` | `$HOME` | Root for parquet files and local checkpoints |
| `NUM_GPUS` | `2` | Devices shared by actor, rollout, and the UnifiedReward worker |
| `NUM_NODES` | `1` | Ray nodes |
| `MAX_NUM_SEQS` | `256` | vLLM-Omni `max_num_seqs` for step-wise continuous batching |

`data.train_files` and `data.val_files` are fixed in the scripts to `$WORKSPACE/data/r2i_bench/qwen_image/{train,test}.parquet`.

Each script runs `python3 -m verl_omni.trainer.main_diffusion` with:

- `algorithm.adv_estimator=flow_grpo`: both AR and DiT use a GRPO-style advantage estimator
- `actor_rollout_ref.model.model_type=diffusion_composite_model`
- `actor_rollout_ref.model.algorithm=dual_grpo`: two-stage AR + DiT rollout pipeline
- `actor_rollout_ref.actor.strategy=fsdp2` with parameter and optimizer offload
- `actor_rollout_ref.actor.diffusion_loss.loss_mode=flow_grpo`: DiT uses a FlowGRPO loss; AR uses a GRPO-style token loss
- `actor_rollout_ref.rollout.name=vllm_omni`: rollout engine
- `actor_rollout_ref.rollout.agent.default_agent_loop=composite_single_turn_agent`: two-stage rollout
- `actor_rollout_ref.rollout.m=2` and `actor_rollout_ref.rollout.n=4`: number of AR traces and DiT samples per trace
- `reward.reward_manager.name=MultiVisualRewardManager`: compute AR and DiT rewards
- `trainer.train_ar=True`: train the AR text encoder

Training samples per step are `train_batch_size × rollout.m × rollout.n` images and `train_batch_size × rollout.m` reasoning traces.

Both the encoder and the DiT share one AdamW scheduler in the current scripts (`actor_rollout_ref.actor.optim.lr=3e-5`). The paper uses separate learning rates (2e-6 for the LLM, 3e-4 for the DiT under FlowGRPO-fast).

### Dual reward config

Similar to the paper's VLM-as-judge setup, PickScore measures semantic consistency and conceptual alignment for the AR stage, while UnifiedReward scores Alignment, Coherence, and Style for the DiT stage (analogous to the paper's aesthetic and physical-consistency signals). `MultiVisualRewardManager` computes both scores, and the trainer or rollout with async reward computation then extracts them in post-processing:

```bash
REWARD_ENGINE=vllm
DIT_REWARD_MODEL_NAME=CodeGoat24/UnifiedReward-2.0-qwen3vl-4b

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
'+reward.reward_functions.ar.weight=0.0' # must set it zero
"+reward.reward_functions.dit.path=pkg://verl_omni.utils.reward_score.unified_reward"
'+reward.reward_functions.dit.name=compute_score_unified_reward'
'+reward.reward_functions.dit.weight=1.0' # must set it non-zero, the combined score is same as dit score
```

## Logging

The scripts log to console and TensorBoard by default. To also log to W&B, set your API key and add `"wandb"` to `trainer.logger`:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
trainer.logger='["console", "tensorboard", "wandb"]'
trainer.project_name=dual_grpo
trainer.experiment_name=qwen_image_dualgrpo
```

Validation images, when `trainer.test_freq` is reached, are written under `validation_data_dir=validation_data`. `trainer.log_val_generations=8`.

See the [Metrics Documentation](../../docs/start/metrics.md) for diffusion training metrics. With `trainer.train_ar=True`, encoder metrics are logged alongside the DiT metrics.

## Variants

| Variant | Script | GPUs | Notes |
| --- | --- | --- | --- |
| Full weight | `examples/dualgrpo_trainer/qwen_image/run_qwen_image.sh` | `NUM_GPUS` | FSDP2 full weights, UnifiedReward-2.0 4B |
| LoRA | `examples/dualgrpo_trainer/qwen_image/run_qwen_image_lora.sh` | `NUM_GPUS` | Rank 8 / alpha 16, visual modules excluded, UnifiedReward-2.0 4B |
