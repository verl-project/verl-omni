# DPO Training

Last updated: 08/14/2026

This directory contains examples for **direct-preference** training (DPO and
related losses). Three workflows are supported:

1. **Qwen-Image online DPO** — rollout and reward run each training step;
   preference pairs are formed from live samples.
2. **Qwen3-Omni offline DPO** — multimodal preference pairs are prepared ahead
   of time; training updates a thinker-only LoRA adapter.
3. **SD3.5 offline DPO** — win/lose pairs and precomputed tensors are prepared
   ahead of time; training reads them from parquet without rollout or reward
   workers.

For implementation details on adding or extending direct-preference algorithms,
see
[How to Integrate a New Direct-Preference Algorithm for Diffusion Model](../../docs/contributing/integrating_a_new_direct_preference_algorithm_for_diffusion_model.md).

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
data following [Prepare the dataset](https://verl-omni.readthedocs.io/en/latest/examples/flowgrpo_trainer.html#prepare-the-dataset)
in [Examples - FlowGRPO Trainer](https://verl-omni.readthedocs.io/en/latest/examples/flowgrpo_trainer.html) (raw OCR from
[flow_grpo/dataset/ocr](https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr),
then `examples/flowgrpo_trainer/data_process/qwenimage_ocr.py` to write
`$WORKSPACE/data/ocr/qwen_image/train.parquet` and `test.parquet`).

### Run

#### NVIDIA GPU

```bash
bash examples/dpo_trainer/qwen_image/run_qwen_image_online_dpo_lora.sh \
  data.train_files=$WORKSPACE/data/ocr/qwen_image/train.parquet \
  data.val_files=$WORKSPACE/data/ocr/qwen_image/test.parquet
```

#### NPU

For Huawei Ascend NPUs, use the NPU-optimized script:

```bash
bash examples/dpo_trainer/qwen_image/run_qwen_image_online_dpo_lora_npu.sh \
  data.train_files=$WORKSPACE/data/ocr/qwen_image/train.parquet \
  data.val_files=$WORKSPACE/data/ocr/qwen_image/test.parquet
```

This script uses a 16-NPU global distribution strategy with:
- `actor_rollout_ref.model.attn_backend='_native_npu'`
- `actor_rollout_ref.rollout.tensor_model_parallel_size=2`
- `reward.reward_model.rollout.tensor_model_parallel_size=4`
- `trainer.n_gpus_per_node=16`

### Notes

- Pairing is fixed to top-vs-bottom reward per prompt. Set
  `actor_rollout_ref.rollout.n` to at least `2` so each prompt has enough
  candidates. Recommend to set it to `8` or `16` for better performance.
- The example sets `true_cfg_scale=1.0`, so CFG is no applied.


### Performance

> All experiments were conducted on *NVIDIA H800* GPUs; NPU experiments use *16× Ascend NPUs*. The OCR reward was used for all experiments.

| Script | Model | Algorithm | Hybrid Engine | # Cards | Reward Fn | # Cards for Actor | # Cards for Rollout | # Cards for Async Reward | Batch Size | `rollout.n` | lr   | # Val Samples | Training Samples per Step | `ppo_micro_batch_size_per_gpu` | Throughput (Samples / Card / Seconds) | Time per Step (Seconds) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `examples/dpo_trainer/qwen_image/run_qwen_image_online_dpo_lora.sh` | Qwen-Image | Online DPO | True | 4 (NVIDIA) | qwenvl-ocr-vllm | 4 | 4 | 0 (sync) | 32 | 16 | 3e-4 | 1k (full set) | 32×2=64 | 8 | 0.040 | 408 |
| `examples/dpo_trainer/qwen_image/run_qwen_image_online_dpo_lora_npu.sh` | Qwen-Image | Online DPO | True | 16 (NPU) | qwenvl-ocr-vllm | 16 | 16 | 0 (sync) | 32 | 16 | 3e-4 | 1k (full set) | 32×2=64 | 4 | 0.003 | 1188 |

- Colocated actor, vLLM-Omni rollout, and sync OCR reward on 4 NVIDIA GPUs (or 16 NPUs for NPU script); `rollout.n=16` samples candidates, then top/bottom pairing keeps 64 actor-update images per step.
- Validation uses the full OCR test parquet.
- Unlike policy-gradient trainers (e.g. FlowGRPO), where actor updates use `train_batch_size × rollout.n` images per step, online DPO keeps one `[chosen, rejected]` pair per prompt (`train_batch_size × 2`), so throughput numbers are not directly comparable—use the **Training Samples per Step** column.

> **Note:** Reward curves may differ between runs because online DPO depends on stochastic diffusion rollouts and the example scripts do not fix the data seed.


## Qwen3-Omni Offline DPO

This workflow trains Qwen3-Omni on offline image/video/audio preference pairs
from Omni-Preference. Training does not run rollout or online reward scoring; it
loads `[chosen, rejected]` answer pairs from parquet and optimizes a LoRA adapter
with the omni DPO loss.

### Dataset

Prepare Omni-Preference parquet files by following
[`data_process/omni_preference_dpo_dataset.md`](data_process/omni_preference_dpo_dataset.md).
The training script expects:

```text
${DATA_DIR}/image/train.parquet
${DATA_DIR}/image/test.parquet
${DATA_DIR}/video/train.parquet
${DATA_DIR}/video/test.parquet
${DATA_DIR}/audio/train.parquet
${DATA_DIR}/audio/test.parquet
```

Each row is one preference pair with a multimodal prompt, `chosen`, `rejected`,
`win_score`, `lose_score`, media paths, and modality metadata.

### Training

Run the LoRA DPO example:

```bash
DATA_DIR=/path/to/Omni-Preference/parquet_dpo \
MODEL_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct \
bash examples/dpo_trainer/qwen3_omni/qwen3_omni/run_qwen3_omni_omni_preference_lora.sh
```

Common overrides:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TOTAL_TRAINING_STEPS=200 \
TRAIN_BATCH_SIZE=32 \
VAL_BATCH_SIZE=32 \
bash examples/dpo_trainer/qwen3_omni/qwen3_omni/run_qwen3_omni_omni_preference_lora.sh
```


Key settings:

- `algorithm.sample_source=offline`: read preference pairs from parquet; no
  rollout or reward worker is used.
- `algorithm.paired_preference=true`: treat adjacent chosen/rejected rows as one
  DPO pair after collation.
- `data.balance_max_samples_by_modality=true`: split `val_max_samples` evenly
  across image/video/audio validation rows.
- `data.val_max_samples`: total validation sample cap. With the default three
  modalities, `96` means `32` per modality.
- `ModalityGroupedBatchSampler`: keeps batches single-modality, which is required
  by the offline MLLM DPO collator.
- `actor_rollout_ref.model.lora_rank`, `lora_alpha`, `target_modules`,
  `target_parameters`: LoRA on thinker attention (`q/k/v/o_proj`) and MoE
  (`gate_up_proj`, `down_proj`) modules. Rank defaults to 32, alpha to 64.
- `actor_rollout_ref.model.exclude_modules`: freezes talker, code2wav, visual,
  and audio tower modules in the example.
- `actor_rollout_ref.actor.omni_loss.*`: DPO loss options such as `beta`,
  `label_smoothing`, `loss_type`, and whether to average log-probs.
- `trainer.save_freq` / `trainer.test_freq`: checkpoint and validation interval
  in training steps.

### Performance

> Measured on 4× NVIDIA H800 GPUs. Offline DPO reads preference pairs directly,
> so no reward model is used during training.

| Script | Model | Algorithm | # Cards | Reward Model | Training Samples per Step | `ppo_micro_batch_size_per_gpu` | Throughput (Samples / Card / Seconds) | Time per Step (Seconds) | Val Accuracy | Val Reward Margin | W&B Report |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `examples/dpo_trainer/qwen3_omni/qwen3_omni/run_qwen3_omni_omni_preference_lora.sh` | Qwen3-Omni-30B-A3B-Instruct | Offline DPO + LoRA | 4 | None | 32×2=64 | 2 | 1.35 | 14.07 | 0.90742 | 0.843 | [W&B report](https://api.wandb.ai/links/didan/iumxl2zr) |

### Validation

Training-time validation reports offline DPO metrics on held-out parquet rows.
For model-quality comparison, use the MiniCPM-o judge script after checkpoints
are saved.

#### 1. Start the MiniCPM-o judge server

Keep the judge on a separate GPU from generation:

```bash
CUDA_VISIBLE_DEVICES=1 \
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} \
HF_MODULES_CACHE=${HF_MODULES_CACHE:-$HOME/.cache/huggingface/modules} \
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS} \
VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0} \
vllm serve openbmb/MiniCPM-o-4_5 \
  --host 127.0.0.1 \
  --port 8001 \
  --dtype bfloat16 \
  --trust-remote-code \
  --enforce-eager
```

#### 2. Run staged Transformers + LoRA evaluation

`vlm_as_judge.py` uses Hugging Face Transformers. Qwen3-Omni
generation is currently run on a single visible CUDA device. The default
`--device-map cuda-offload-non-thinker` keeps the active thinker path on
`cuda:0` (within `CUDA_VISIBLE_DEVICES`) and offloads unused `talker` /
`code2wav` modules to CPU.

Evaluation is staged so expensive generation can be resumed and inspected:

1. `reference`: load the original Qwen3-Omni weights only and cache generated
   texts.
2. `trained`: load Qwen3-Omni + the LoRA adapter and cache generated texts.
3. `judge`: read the paired cached texts and send them to the MiniCPM-o judge.

`--stage all` runs the three stages in that order. The cache files default to
`<output>.reference.jsonl` and `<output>.trained.jsonl`; the final judge output
is `<output>.jsonl`. All stages iterate samples in dataset order and use
`(data_file, index, uid)` as the stable join key.

The repository root [`eval_vlm_as_judge.sh`](qwen3_omni/eval_vlm_as_judge.sh) is the runnable example.
It keeps reference and trained generation as resumable cache stages, then runs
the judge stage over the cached outputs. Adjust the path variables, checkpoint
steps, modalities, and judge address for your environment:

```bash
CKPT_ROOT=checkpoints/omni-preference-dpo/qwen3-omni-offline-dpo-lora
DATA_DIR=/path/to/Omni-Preference/parquet_dpo
MODEL_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct/
OUT_DIR=outputs/qwen3_omni_judge_eval
MAX_SAMPLES=60
CUDA_DEVICES=1
STEPS=(50 100 150 200)
MODALITIES=(image video audio)

mkdir -p "${OUT_DIR}"

for step in "${STEPS[@]}"; do
  .venv/bin/python examples/dpo_trainer/qwen3_omni/vlm_as_judge.py \
    --data-dir "${DATA_DIR}" \
    --modalities "${MODALITIES[@]}" \
    --output-jsonl "${OUT_DIR}/global_step_${step}.jsonl" \
    --summary-json "${OUT_DIR}/global_step_${step}.summary.json" \
    --reference-jsonl "${OUT_DIR}/reference.jsonl" \
    --trained-jsonl "${OUT_DIR}/global_step_${step}.trained.jsonl" \
    --stage judge \
    --max-samples "${MAX_SAMPLES}" \
    --judge-max-tokens 4096 \
    --judge-router-address 127.0.0.1:8001
done
```

`--adapter-path` may be any of:

- a PEFT directory (`adapter_config.json` + `adapter_model.safetensors|.bin`)
- `.../global_step_N` (uses `actor/lora_adapter` when present)
- `.../global_step_N/actor` (FSDP dir with `fsdp_config.json` + `lora_train_meta.json`)

If the PEFT export is missing, the script runs `export_fsdp_lora_adapter` into
`actor/lora_adapter/` automatically.

#### Notes

- The reference stage does not attach PEFT; it uses the base Qwen3-Omni weights
  directly.
- The trained stage unfuses Qwen3-Omni MoE experts so exported PEFT keys match
  verl training, then attaches the adapter with `PeftModel.from_pretrained`.
- If one stage fails, rerun only the missing stage. 
- Requires `transformers`, `peft`, `accelerate`, and `qwen-omni-utils`.
  FlashAttention 2 is optional via `--attn-implementation flash_attention_2`.

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
bash examples/dpo_trainer/sd35/run_sd35_medium_offline_dpo_lora.sh \
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

`examples/dpo_trainer/data_process/prepare_offline_dpo.py` can call any reward function with the standard VeRL-Omni
custom reward signature. The example commands above use
`verl_omni/utils/reward_score/unified_reward.py` and can either launch a local
OpenAI-compatible vLLM reward server or connect to an existing one through
`--reward_router_address`.

