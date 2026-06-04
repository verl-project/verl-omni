# DPO Training

This directory contains examples for **direct-preference** diffusion training
(DPO and related losses). Two workflows are supported:

1. **Qwen-Image online DPO** — rollout and reward run each training step;
   preference pairs are formed from live samples.
2. **SD3.5 offline DPO** — win/lose pairs and precomputed tensors are prepared
   ahead of time; training reads them from parquet without rollout or reward
   workers.

For implementation details on adding or extending direct-preference algorithms,
see
[`docs/contributing/integrating_a_new_direct_preference_algorithm_for_diffusion_model.md`](../../docs/contributing/integrating_a_new_direct_preference_algorithm_for_diffusion_model.md).

## Qwen-Image Online DPO

Online DPO does not consume pre-ranked win/lose rows from parquet. At each
training step it:

- samples multiple candidate images per prompt with vLLM-Omni rollout;
- scores images through the configured reward function;
- forms one adjacent `[chosen, rejected]` pair per prompt from the highest-
  and lowest-scoring candidates;
- runs the diffusion DPO loss on those pairs.

### Dataset

Use the same OCR prompt parquet as FlowGRPO Qwen-Image training. Prepare the
data following [Prepare the dataset](../flowgrpo_trainer/README.md#prepare-the-dataset)
in `examples/flowgrpo_trainer/README.md` (raw OCR from
[flow_grpo/dataset/ocr](https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr),
then `examples/flowgrpo_trainer/data_process/qwenimage_ocr.py` to write
`$WORKSPACE/data/ocr/qwen_image/train.parquet` and `test.parquet`).

### Run

```bash
bash examples/dpo_trainer/run_qwen_image_online_dpo_lora.sh \
  data.train_files=$WORKSPACE/data/ocr/qwen_image/train.parquet \
  data.val_files=$WORKSPACE/data/ocr/qwen_image/test.parquet
```

### Notes

- Pairing is fixed to top-vs-bottom reward per prompt. Set
  `actor_rollout_ref.rollout.n` to at least `2` so each prompt has enough
  candidates. Recommend to set it to `8` or `16` for better performance.
- The example sets `true_cfg_scale=1.0`, so CFG is no applied.


### Performance

> Online DPO experiment conducted on *NVIDIA H800* GPUs with the same OCR reward and prompt parquet as FlowGRPO Qwen-Image training.

The experiment settings and throughputs are shown in the table below. Online DPO forms one `[chosen, rejected]` pair per prompt after rollout, so **training samples per step** counts actor-update pairs (`train_batch_size × 2`). **Throughput** follows trainer metrics: `perf/total_num_images / (perf/time_per_step × n_gpus)` (64 images per step on 4 GPUs).

| Script | Model | Algorithm | Hybrid Engine | # Cards | Reward Fn | # GPUs for Actor | # GPUs for Rollout | # GPUs for Async Reward | Batch Size | `rollout.n` | lr   | # Val Samples | Training Samples per Step | `ppo_micro_batch_size_per_gpu` | Throughput (Samples / GPU / Seconds) | Time per Step (Seconds) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `run_qwen_image_online_dpo_lora.sh` | Qwen-Image | Online DPO | True | 4 | qwenvl-ocr-vllm | 4 | 4 | 0 (sync) | 32 | 8 | 3e-4 | 1k (full set) | 32×2=64 | 8 | 0.040 | 408 |

- Colocated actor, vLLM-Omni rollout, and sync OCR reward on 4 GPUs; `rollout.n=16` samples candidates, then top/bottom pairing keeps 64 actor-update images per step (`perf/total_num_images=64`).
- Validation uses `trainer.val_before_train=True` on the full OCR test parquet (same as FlowGRPO).

> **Note:** Reward curves may differ between runs because online DPO depends on stochastic diffusion rollouts and the example scripts do not fix the data seed.


## SD3.5 Offline DPO

Offline DPO uses a frozen reference pipeline to generate several candidates per
prompt, score them, and write one pre-ranked win/lose pair per parquet row.
Training loads `OfflineDPODataset` via `data.custom_cls`, expands each row into
adjacent `[win, lose]` samples with a shared `uid`, and stacks precomputed
`latents_clean` plus SD3 prompt embeddings before the DPO loss. Rollout and
reward workers are disabled; validation generation is off by default.

### Pair data schema

Each parquet row contains:

- `prompt`: chat-style prompt messages.
- `negative_prompt`: optional negative prompt messages.
- `img_win` / `img_lose`: paths to the highest- and lowest-scoring images.
- `img_win_latents` / `img_lose_latents`: precomputed diffusion latents for SD3.
- `prompt_embeds`, `prompt_embeds_mask`, `pooled_prompt_embeds`: precomputed
  text-encoder outputs.
- `win_score` / `lose_score`: reward scores used to order the pair.
- `extra_info.raw_prompt`: plain prompt text for traceability.

### Data preparation

```bash
python3 examples/dpo_trainer/data_process/prepare_offline_dpo.py \
  --input_file dataset/my_prompts/train_prompts.txt \
  --output_file data/offline_dpo_sd3/train.parquet \
  --image_dir data/offline_dpo_sd3/images/train \
  --model_path stabilityai/stable-diffusion-3.5-medium \
  --num_images_per_prompt 8 \
  --height 256 \
  --width 256 \
  --num_inference_steps 25 \
  --guidance_scale 4.0
```

`--launch_reward_server` starts a `vllm serve` subprocess and waits for
`/v1/models` before scoring. If a reward server is already running, omit
`--launch_reward_server` and pass `--reward_router_address host:port`. Override
`--reward_server_command` for custom vLLM flags; the template supports
`{model}`, `{host}`, and `{port}`.

`prepare_offline_dpo.py` can call any reward function with the standard VeRL-Omni
custom reward signature, for example:

```bash
  --reward_function_path verl_omni/utils/reward_score/unified_reward.py \
  --reward_function_name compute_score_unified_reward \
  --launch_reward_server \
  --reward_model_name CodeGoat24/UnifiedReward-2.0-qwen3vl-8b
```

### Train

```bash
bash examples/dpo_trainer/run_sd35_medium_offline_dpo_lora.sh \
  data.train_files=data/offline_dpo_sd3/train.parquet \
  data.val_files=data/offline_dpo_sd3/test.parquet
```

