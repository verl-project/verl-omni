# BAGEL-7B-MoT UniGRPO training (trainside, no vLLM)

Last updated: 09/18/2026

[BAGEL-7B-MoT](https://github.com/ByteDance-Seed/BAGEL) is a Mixture-of-Transformers model that supports both image understanding and generation. **UniGRPO** trains one shared transformer to first produce an autoregressive "thinking" chain through the understanding (`und`) experts and then render an image through the generation (`moe_gen`) experts conditioned on the prompt and thinking. PickScore rewards the image, the prompt-group GRPO advantage is shared by the AR and image tracks, and a joint **2-backwards -> 1 optimizer step** update trains both experts with separate learning rates. This recipe reproduces the UniGRPO algorithm described in the [UniGRPO paper](https://arxiv.org/abs/2603.23500) on top of verl-omni's BAGEL FlowGRPO components.

## Trainside rollout (no vLLM)

Unlike the FlowGRPO recipe, UniGRPO samples on the live FSDP actor module through a flat full-parameter bf16 replica synced from the FSDP master at each step (`actor_rollout_ref.rollout.name=trainside`) instead of a vLLM server. This keeps variable-length AR decoding collective-free on each rank and lets the AR and image tracks share one transformer. The training loop is `UniGRPORayTrainer` (`algorithm.trainer_type=unigrpo`): worker `generate` -> PickScore reward -> flow_grpo advantage -> `record_old_logp` (anchors the on-policy ratio to 1 at update 0) -> joint `update_actor`.

The implementation includes:

- Adapter `verl_omni/pipelines/bagel_unigrpo/` (`BagelUniGRPO`, registered as `(OmniBagelForConditionalGeneration, unigrpo)`), `BagelUniPipeline`, `UniGRPOJointUpdater`, and KV-cache AR decoding.
- Engine `UniGRPODiffusersFSDPEngine` (`model_type=diffusion_unigrpo_model`) with per-layer and root-leaf FSDP2 wrapping, per-expert learning rates, and joint forward/backward execution.
- Image loss `UniGRPOLoss` (`loss_mode=unigrpo`) with GRPO-Guard RatioNorm policy gradients and a velocity-MSE regularizer.

## Prerequisites

- Install VeRL-Omni (see the [installation guide](../../../docs/start/install.md)).
- Run commands from the repository root.
- Download the BAGEL checkpoint:

  ```bash
  huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT --local-dir ~/models/ByteDance-Seed/BAGEL-7B-MoT
  ```

- Full fine-tuning of the 14.6B model plus a full bf16 rollout replica per rank requires `param_offload=false`; scale `trainer.nnodes` and `n_gpus_per_node` for your cluster.

## Prepare the dataset

UniGRPO reuses the FlowGRPO PickScore preprocessing, which writes a BAGEL-native `prompt_token_ids` column consumed directly by the trainside pipeline:

```bash
export WORKSPACE=${WORKSPACE:-$HOME}

python3 examples/flowgrpo_trainer/data_process/bagel_pickscore.py \
  --model_path ~/models/ByteDance-Seed/BAGEL-7B-MoT \
  --input_dir ~/data/pickscore \
  --output_dir $WORKSPACE/data/pickscore/bagel
```

This produces `$WORKSPACE/data/pickscore/bagel/train.parquet` and `test.parquet`. The raw dataset (`train.txt` and `test.txt`) is from <https://github.com/yifan123/flow_grpo/tree/main/dataset/pickscore>.

## Run training

```bash
bash examples/unigrpo_trainer/bagel/run_bagel_unigrpo.sh
```

### Key knobs

- `actor_rollout_ref.rollout.n` — GRPO group size G (samples per prompt).
- `data.train_batch_size / actor_rollout_ref.actor.ppo_mini_batch_size` — number of joint updates per step (`num_updates_per_batch`); the default 8 / 4 gives 2 (update 0 on-policy, update 1 off-policy).
- `actor_rollout_ref.actor.optim.param_group_lrs={moe_gen: 3e-5}` with `optim.lr=1e-6` — faster learning rate for the generation experts while the base and understanding parameters use the base learning rate; this requires `optim._target_=...FSDPDiffusionOptimizerConfig`.
- `actor_rollout_ref.actor.diffusion_loss.{clip_ratio,mse_weight,ratio_norm}` — image policy-gradient clip, velocity-MSE weight, and GRPO-Guard RatioNorm toggle.
- `actor_rollout_ref.rollout.pipeline.num_inference_steps` and `rollout.algo.{noise_level,sde_window_size}` — denoising steps, SDE noise level (eta), and number of SDE steps. The AR-decode knobs (`max_new_tokens`, `temperature`, `top_k`, and `top_p`) and the SDE window fraction use `BagelUniPipeline` defaults (1024 / 1.0 / 1024 / 1.0 and `(0.0, 0.2)`).

In-loop validation is skipped because trainside has no vLLM generation path; run evaluation as a separate offline pass over saved checkpoints.
