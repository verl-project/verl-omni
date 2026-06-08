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


| Script | Model | Algorithm | Hybrid Engine | # Cards | Reward Fn | # GPUs for Actor | # GPUs for Rollout | # GPUs for Async Reward | Batch Size | `rollout.n` | lr   | # Val Samples | Training Samples per Step | `ppo_micro_batch_size_per_gpu` | Throughput (Samples / GPU / Seconds) | Time per Step (Seconds) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `run_qwen_image_online_dpo_lora.sh` | Qwen-Image | Online DPO | True | 4 | qwenvl-ocr-vllm | 4 | 4 | 0 (sync) | 32 | 16 | 3e-4 | 1k (full set) | 32×2=64 | 8 | 0.040 | 408 |

- Colocated actor, vLLM-Omni rollout, and sync OCR reward on 4 GPUs; `rollout.n=16` samples candidates, then top/bottom pairing keeps 64 actor-update images per step (`perf/total_num_images=64`).
- Validation uses `trainer.val_before_train=True` on the full OCR test parquet (same as FlowGRPO).
- Unlike policy-gradient trainers (e.g. FlowGRPO), where actor updates use `train_batch_size × rollout.n` images per step, online DPO keeps one `[chosen, rejected]` pair per prompt (`train_batch_size × 2`), so throughput numbers are not directly comparable—use the **Training Samples per Step** column.

> **Note:** Reward curves may differ between runs because online DPO depends on stochastic diffusion rollouts and the example scripts do not fix the data seed.


## SD3.5 Offline DPO

This workflow trains Stable Diffusion 3.5 with offline DPO. The data preparation
step first generates several candidate images per prompt with a frozen reference
pipeline, scores the candidates, and writes one pre-ranked win/lose pair per
prompt. Training consumes those pairs directly and does not run online rollout,
training-time reward scoring, or online pair selection.

### Pair data

The resulting parquet rows contain:

- `prompt`: chat-style prompt messages.
- `negative_prompt`: optional negative prompt messages.
- `img_win`: path to the highest-scoring generated image.
- `img_lose`: path to the lowest-scoring generated image.
- `img_win_latents` and `img_lose_latents`: precomputed SD3 VAE latents.
- `prompt_embeds`, `prompt_embeds_mask`, and `pooled_prompt_embeds`: precomputed
  SD3 text-encoder outputs.
- `win_score` and `lose_score`: reward scores used to order the pair.
- `extra_info.raw_prompt`: plain prompt text for traceability.

Generate offline pairs from prompt files and choose the parquet output paths
explicitly:

```bash
python3 examples/dpo_trainer/data_process/prepare_offline_dpo.py \
  --input_file dataset/my_prompts/train_prompts.txt \
  --output_file data/offline_dpo/train.parquet \
  --image_dir data/offline_dpo/images/train \
  --model_path stabilityai/stable-diffusion-3.5-medium \
  --num_images_per_prompt 4 \
  --height 256 \
  --width 256 \
  --num_inference_steps 25 \
  --guidance_scale 4.0 \
  --reward_function_path verl_omni/utils/reward_score/unified_reward.py \
  --reward_function_name compute_score_unified_reward \
  --launch_reward_server \
  --reward_server_host 127.0.0.1 \
  --reward_server_port 8000 \
  --reward_model_name CodeGoat24/UnifiedReward-2.0-qwen3vl-8b

python3 examples/dpo_trainer/data_process/prepare_offline_dpo.py \
  --input_file dataset/my_prompts/eval_prompts.txt \
  --output_file data/offline_dpo/test.parquet \
  --image_dir data/offline_dpo/images/test \
  --split test \
  --model_path stabilityai/stable-diffusion-3.5-medium \
  --num_images_per_prompt 4 \
  --height 256 \
  --width 256 \
  --num_inference_steps 25 \
  --guidance_scale 4.0 \
  --reward_function_path verl_omni/utils/reward_score/unified_reward.py \
  --reward_function_name compute_score_unified_reward \
  --launch_reward_server \
  --reward_server_host 127.0.0.1 \
  --reward_server_port 8000 \
  --reward_model_name CodeGoat24/UnifiedReward-2.0-qwen3vl-8b
```

`--launch_reward_server` starts a `vllm serve` subprocess with the reward model
and waits for `/v1/models` before scoring. If you already have an
OpenAI-compatible reward server running, omit `--launch_reward_server` and pass
`--reward_router_address host:port` instead. For custom vLLM flags, override
`--reward_server_command`; the template can use `{model}`, `{host}` and
`{port}`.

This writes:

- `data/offline_dpo/train.parquet`
- `data/offline_dpo/test.parquet`
- generated images under the requested `--image_dir`

### Training

Train on the offline pairs with:

```bash
bash examples/dpo_trainer/run_sd35_medium_offline_dpo_lora.sh \
  data.train_files=data/offline_dpo/train.parquet \
  data.val_files=data/offline_dpo/test.parquet
```

During training, `run_sd35_medium_offline_dpo_lora.sh` sets
`algorithm.sample_source=offline` and loads `OfflineDPODataset` via
`data.custom_cls`. The dataset expands each row into adjacent `[win, lose]`
samples with a shared `uid`. Collate stacks the precomputed latents (from
`img_win_latents` / `img_lose_latents` in parquet, exposed as `latents_clean` in
the actor batch) plus SD3 prompt embeddings before calling the DPO loss, so
training does not load the SD3 VAE or text encoders during actor updates. Offline
DPO also disables rollout and reward workers, so validation generation is
disabled by default.

### Reward template

`prepare_offline_dpo.py` can call any reward function with the standard VeRL-Omni
custom reward signature. The example commands above use
`verl_omni/utils/reward_score/unified_reward.py` and can either launch a local
OpenAI-compatible vLLM reward server or connect to an existing one through
`--reward_router_address`.

