# FlowGRPO Trainer

This example shows how to post-train `Qwen-Image` (and, in a separate
recipe, `BAGEL-7B-MoT`) with FlowGRPO on an OCR-style image generation
task using `vllm-omni` rollout and a visual generative reward model
(`Qwen3-VL-8B-Instruct` in this example).

For the full installation and quickstart guide, see `docs/start/flowgrpo_quickstart.md`. For algorithm details and rule-based reward training (e.g. JPEG incompressibility), see `docs/algo/flowgrpo.md`.

## Installation

Follow the [installation guide](../../docs/start/install.md) to set up the base environment, then install the FlowGRPO-specific dependency:

```bash
pip install Levenshtein
```

The provided script is configured for a single node with `4` GPUs.

## Prepare the dataset

Obtain the raw OCR dataset from the original Flow-GRPO repository:

- https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr

Place the raw dataset under `$WORKSPACE/data/ocr` (where `WORKSPACE` defaults to `$HOME`), then preprocess it into parquet files:

```bash
python3 examples/flowgrpo_trainer/data_process/qwenimage_ocr.py \
  --input_dir $WORKSPACE/data/ocr \
  --output_dir $WORKSPACE/data/ocr
```

This produces:

- `$WORKSPACE/data/ocr/train.parquet`
- `$WORKSPACE/data/ocr/test.parquet`

## Prepare the models

**Policy model (Qwen-Image):** the script uses the Hugging Face Hub ID `Qwen/Qwen-Image` directly — no manual download is required. Hugging Face will cache the weights automatically on first run. To use a local copy instead, edit the `model_name` variable in the script directly.

**Reward model (Qwen3-VL-8B-Instruct):** the script defaults to the Hugging Face Hub ID `Qwen/Qwen3-VL-8B-Instruct`, so no manual download is required — Hugging Face will cache it automatically on first run. To use a local copy instead, edit the `reward_model_name` variable in the script directly.

## Run training

Launch the example from the repository root:

```bash
bash examples/flowgrpo_trainer/run_qwen_image_ocr_lora.sh
```

Optional KL loss tuning:

- `actor_rollout_ref.actor.use_kl_loss=True`
- `actor_rollout_ref.actor.kl_loss_coef=0.001`

The script runs `python3 -m verl_omni.trainer.diffusion.main_flowgrpo` with:

- `algorithm.adv_estimator=flow_grpo`
- `actor_rollout_ref.model.path=Qwen/Qwen-Image`
- `actor_rollout_ref.model.lora_rank=64`
- `actor_rollout_ref.model.lora_alpha=128`
- `actor_rollout_ref.rollout.name=vllm_omni`
- `reward.custom_reward_function.name=compute_score_ocr`
- `trainer.n_gpus_per_node=4`

## Logging

W&B logging is enabled by default in the example script:

```bash
export WANDB_API_KEY=<your_wandb_api_key>
```

The script sets:

```bash
trainer.logger='["console", "wandb"]'
trainer.project_name=flow_grpo
trainer.experiment_name=qwen_image_ocr_lora
```

Override these values on the command line if you want to log under a different project or run name.

### Diffusion-specific metrics

See the [Metrics Documentation](../../docs/start/metrics.md) for a full description of all diffusion-specific training metrics.

## Variants

For reward models that are expensive to evaluate (e.g., a VLM judge), the reward model can be allocated its own dedicated GPU resource pool and run asynchronously alongside the policy. This avoids blocking policy training on reward computation.

```bash
bash examples/flowgrpo_trainer/run_qwen_image_ocr_lora_async_reward.sh
```

## BAGEL recipe

`run_bagel_flowgrpo.sh` post-trains `BAGEL-7B-MoT` (Mixture-of-Transformers)
with the same OCR reward. BAGEL is registered through the
`verl_omni.pipelines.bagel_flow_grpo` adapter pair as the architecture
`OmniBagelForConditionalGeneration`, and the rollout uses a
single-stage vllm-omni pipeline whose schema is described in
[`bagel_stage_config.yaml`](bagel_stage_config.yaml).

Prerequisites in addition to the Qwen-Image recipe:

- A local copy of `BAGEL-7B-MoT` (HF repo `ByteDance-Seed/BAGEL-7B-MoT`).
- The same `Qwen3-VL-8B-Instruct` reward model and OCR parquet files
  produced above.

Launch:

```bash
export BAGEL_MODEL_PATH=/path/to/BAGEL-7B-MoT
export REWARD_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
export OCR_TRAIN_PATH=$WORKSPACE/data/ocr/train.parquet
export OCR_TEST_PATH=$WORKSPACE/data/ocr/test.parquet

bash examples/flowgrpo_trainer/run_bagel_flowgrpo.sh
```

Notable differences from the Qwen-Image recipe:

- Uses `+actor_rollout_ref.model.architecture=OmniBagelForConditionalGeneration`
  to bypass the `model_index.json` lookup (BAGEL ships as a single
  custom checkpoint, not a `diffusers` pipeline).
- LoRA `target_modules` are the BAGEL MoT generation projections
  (`q_proj_moe_gen`, `k_proj_moe_gen`, `v_proj_moe_gen`,
  `o_proj_moe_gen`).
- Passes the stage-config YAML to vllm-omni via
  `+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_configs_path`.


## Performance

> All experiments were conducted on *NVIDIA H800* GPUs using the OCR reward.

The experiment settings and throughputs are shown in the table below.

| Script | Model | Algorithm | Hybrid Engine | # Cards | Reward Fn | # GPUs for Actor | # GPUs for Rollout | # GPUs for Async Reward | Batch Size | `rollout.n` | lr   | # Val Samples | Training Samples per Step | `ppo_micro_batch_size_per_gpu` | Throughput (Samples / GPU / Seconds) | Time per Step (Seconds) |
| --- | --- | --- | --- | --- | --- | --- | --- |-------------------------| --- | --- |------| --- | --- | --- |------------------------------| --------------------------------|
| `run_qwen_image_ocr_lora.sh` | Qwen-Image | Flow-GRPO | True | 4 | qwenvl-ocr-vllm | 4 | 4 | 0 (sync)                | 32 | 16 | 3e-4 | 1k (full set) | 32×16=512 | 16 | 0.305                        | 420 |
| `run_qwen_image_ocr_lora_async_reward.sh` | Qwen-Image | Flow-GRPO | True | 5 | qwenvl-ocr-vllm | 4 | 4 | 1                       | 32 | 16 | 3e-4 | 1k (full set) | 32×16=512 | 16 | 0.280                        | 360 |

- Validation reward curve (evaluated with `trainer.val_before_train=True`):

<div align="center">
<img width="600" alt="2p_comparison" src="https://github.com/user-attachments/assets/1094beaf-fed9-4661-8a6a-1c3983150648" />
<br>
qwen_image_ocr_lora: corresponding with the script `run_qwen_image_ocr_lora.sh`; 
<br>
qwen_image_ocr_lora_async_reward: corresponding with the script `run_qwen_image_ocr_lora_async_reward.sh`.
</div>
