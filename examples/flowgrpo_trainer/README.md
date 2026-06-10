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
  --output_dir $WORKSPACE/data/ocr/qwen_image
```

This produces:

- `$WORKSPACE/data/ocr/qwen_image/train.parquet`
- `$WORKSPACE/data/ocr/qwen_image/test.parquet`

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

The script runs `python3 -m verl_omni.trainer.main_diffusion` with:

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

Ulysses sequence parallelism shards the sequence dimension across GPUs to reduce per-GPU memory. A ready-to-use 4-GPU SP=2 LoRA example is provided:

```bash
bash examples/flowgrpo_trainer/run_qwen_image_ocr_lora_sp2.sh
```

We have provided a script to enable non-cfg full-weight Qwen-Image OCR training. The example is runnable on 4 NVIDIA H200 GPUs; enabling CFG requires more GPU resources.

```bash
bash examples/flowgrpo_trainer/run_qwen_image_ocr.sh
```


## BAGEL recipe

`run_bagel_flowgrpo_lora.sh` post-trains `BAGEL-7B-MoT` (Mixture-of-Transformers)
with the same OCR reward. BAGEL is registered through the
`verl_omni.pipelines.bagel_flow_grpo` adapter pair as the architecture
`OmniBagelForConditionalGeneration`, and the rollout uses a
single-stage vllm-omni pipeline whose schema is described in
[`bagel_deploy_config.yaml`](bagel_deploy_config.yaml).

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

bash examples/flowgrpo_trainer/run_bagel_flowgrpo_lora.sh
```

Notable differences from the Qwen-Image recipe:

- Uses `+actor_rollout_ref.model.architecture=OmniBagelForConditionalGeneration`
  to bypass the `model_index.json` lookup (BAGEL ships as a single
  custom checkpoint, not a `diffusers` pipeline).
- LoRA `target_modules` are the BAGEL MoT generation projections
  (`q_proj_moe_gen`, `k_proj_moe_gen`, `v_proj_moe_gen`,
  `o_proj_moe_gen`).
- Passes the deploy-config YAML to vllm-omni via
  `+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.deploy_config`. The
  legacy `stage_configs_path` entrypoint is **not** supported: it routes
  through vllm-omni 0.20's deprecated stage-args loader, which silently
  kills the BAGEL `DiffusionWorker` subprocess after warmup. Always use
  the `deploy_config` schema documented at
  [`bagel_deploy_config.yaml`](bagel_deploy_config.yaml).
- Defaults to `trainer.n_gpus_per_node=4` with
  `actor_rollout_ref.rollout.tensor_model_parallel_size=1` (4 TP=1
  rollout replicas), matching the Qwen-Image recipe. Be aware of a
  TOCTOU race in vllm-omni's per-process `MASTER_PORT` picker
  (`OmniDiffusionConfig.__post_init__` →`settle_port` in
  [`vllm_omni/diffusion/data.py`](https://github.com/vllm-project/vllm-omni/blob/main/vllm_omni/diffusion/data.py)):
  every concurrent `vLLMOmniHttpServer` Ray actor independently calls
  `is_port_available(p)` and may pick the same port before any of them
  actually `bind`s. Birthday-paradox collision probability is roughly 4%
  at 4 actors and 18% at 8 in the default 100-port window, and is
  amplified further when retries land inside the prior run's TIME_WAIT
  window (≈60s). If a launch dies during `init_distributed_environment`
  with `EADDRINUSE` on a port in 30005–30105, wait ~60s and re-launch.

An NPU script for Atlas A3 with 16 NPUs is also provided. Before running, set the `ASCEND_HOME_PATH` environment variable (defaults to `/usr/local/Ascend/cann-9.0.0`).

```bash
bash examples/flowgrpo_trainer/run_qwen_image_ocr_npu.sh
```

### VeOmni engine backend

The diffusion trainer defaults to FSDP2. To use [VeOmni](https://github.com/ByteDance-Seed/VeOmni) as the actor/reference engine instead, first follow the [VeOmni install instructions](../../docs/start/install.md#optional-engine-backends) (vllm 0.20.2 needs torch 2.11, which conflicts with veomni's pinned `[gpu]` extra — the doc covers the workaround), then run the VeOmni counterpart of the full-weight recipe:

```bash
bash examples/flowgrpo_trainer/run_qwen_image_ocr_veomni.sh
```

`run_qwen_image_ocr_veomni.sh` is intentionally a line-for-line mirror of `run_qwen_image_ocr.sh` and differs only in the engine-selection overrides:

| Override | FSDP2 (default) | VeOmni |
|---|---|---|
| `diffusion/model_engine=` | _(unset; uses `dp_diffusion`)_ | `veomni_diffusion` |
| `actor_rollout_ref.actor.strategy=` | `fsdp2` | `veomni` |
| Actor engine config block | `actor.fsdp_config.*` | `actor.veomni_config.*` |
| Sequence-parallel field | `fsdp_config.ulysses_sequence_parallel_size` | `veomni_config.ulysses_parallel_size` |
| Ref engine config block | `ref.fsdp_config.*` | `ref.veomni_config.*` |

The two backends are not configured by the same Hydra keys, so do not mix `fsdp_config` and `veomni_config` overrides in a single run — `diffusion/model_engine=...` selects the schema, and overrides for the other engine will be rejected as unknown keys.

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

> **Note:** Reward curves may differ from the references above mainly due to rollout-side stochasticity: diffusion rollouts sample random latents/noise, and the example scripts do not fix the data seed, so prompt ordering can vary between runs.
