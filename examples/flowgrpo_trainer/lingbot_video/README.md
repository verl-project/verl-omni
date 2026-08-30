# LingBot Dense T2V FlowGRPO

This example supports only `robbyant/lingbot-video-dense-1.3b` text-to-video
LoRA training.  It intentionally excludes the MoE checkpoint, image input,
the refiner, and SGLang Diffusion.

Install the optional model package in every actor and rollout environment:

```bash
uv pip install -e '.[vllm-omni,lingbot-video,train,dev]'
```

## Rewrite plain prompts into structured captions

Use `rewrite_prompts.py` or `run_rewrite.sh` to convert plain prompts into
LingBot structured-caption JSONL. The launcher is resumable by `prompt_raw`.

```bash
# vLLM backend.
bash examples/flowgrpo_trainer/lingbot_video/serve_rewriter.sh   # TP=8, in its own shell
bash examples/flowgrpo_trainer/lingbot_video/run_rewrite.sh      # concurrency 256
CONCURRENCY=1024 bash examples/flowgrpo_trainer/lingbot_video/run_rewrite.sh

# transformers backend.
BACKEND=transformers GPUS=0,1,2,3,4,5,6,7 \
  bash examples/flowgrpo_trainer/lingbot_video/run_rewrite.sh
```

Then convert JSONL to parquet:

```bash
python examples/flowgrpo_trainer/lingbot_video/prepare_structured_captions.py \
  --train-jsonl captions/train.jsonl --val-jsonl captions/val.jsonl \
  --output-dir ~/data/lingbot_video
```

Launch `run_lingbot_dense_t2v_lora.sh` after preparing train/val parquet.
