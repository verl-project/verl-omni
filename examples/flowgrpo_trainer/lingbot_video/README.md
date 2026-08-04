# LingBot Dense T2V FlowGRPO

This example supports only `robbyant/lingbot-video-dense-1.3b` text-to-video
LoRA training.  It intentionally excludes the MoE checkpoint, image input,
the refiner, and SGLang Diffusion.

Install the optional model package in every actor and rollout environment:

```bash
uv pip install -e '.[vllm-omni,lingbot-video,train,dev]'
```

## Rewrite plain prompts into structured captions

LingBot-Video's DiT consumes structured JSON captions, not casual prompts, so
plain-text prompts are first run through the official two-stage rewriter (step1
EXPAND = base VLM; step2 MAP = base VLM + rewriter LoRA). One JSONL record is
written per prompt, keyed by `caption` (the structured rewrite) with the
original prompt kept as the reward ground truth.

`rewrite_prompts.py` is a single driver with two interchangeable backends
(`--backend`); `run_rewrite.sh` is the launcher and is resumable — records
already in the output (matched by `prompt_raw`) are skipped.

```bash
# Recommended (high throughput): one tensor-parallel vLLM server serves both
# stages (base id for EXPAND, `rewriter` LoRA id for MAP). Start it, then drive it.
bash examples/flowgrpo_trainer/lingbot_video/serve_rewriter.sh   # TP=8, in its own shell
bash examples/flowgrpo_trainer/lingbot_video/run_rewrite.sh      # concurrency 256
CONCURRENCY=1024 bash examples/flowgrpo_trainer/lingbot_video/run_rewrite.sh

# Reference/fallback: load the 27B VLM in-process, one model per GPU, sharded.
BACKEND=transformers GPUS=0,1,2,3,4,5,6,7 \
  bash examples/flowgrpo_trainer/lingbot_video/run_rewrite.sh
```

Both paths use tqdm; the vLLM path feeds a bounded producer/consumer queue so a
fixed pool of `CONCURRENCY` prompts stays in flight and the server
continuous-batches them. Then convert the JSONL to the dataset schema:

```bash
python examples/flowgrpo_trainer/lingbot_video/prepare_structured_captions.py \
  --train-jsonl captions/train.jsonl --val-jsonl captions/val.jsonl \
  --output-dir ~/data/lingbot_video
```

Set a video reward function already available in this repository/environment,
then launch `run_lingbot_dense_t2v_lora.sh`. The recipe uses the official
480×832, 81-frame, guidance-3, shift-3 baseline. Training uses a lighter
10-step rollout by default and keeps 40-step validation for quality checks.
`trainer.validation_data_max_samples` only caps how many validation samples are
saved/logged; use a smaller validation parquet if you want to generate fewer
validation videos.
