# MiniMax H3 image: DiffusionNFT OCR LoRA

DiffusionNFT LoRA training for the MiniMax H3 image model (6-step PDD sampler) with a dense-text OCR reward.

## Reward

`score = ocr * guard * gate` (`ocr_quality_reward.py`)

- `ocr`: Qwen3-VL-8B transcription score (`genrm_ocr`, temperature 0, 160 response tokens).
- `guard`: 0 if a PP-OCR text box touches the frame, else `min(1, 0.08 / text_area)`. Without it the policy enlarges
  the text until it fills or leaves the canvas.
- `gate`: `min(1, HPSv2.1(image, full prompt) / 0.1971)`, where 0.1971 is the untrained model's mean on the validation
  prompts. Without it the policy washes out the scene behind the text.

The full generation prompt must be in `extra_info["prompt"]` of every row.

## Run

```bash
export MODEL_PATH=/path/to/MiniMax-H3-Image
export ACTOR_PATH=/path/to/converted/transformer
export DATA_DIR=/path/to/dense_text            # train_t20.parquet / test_t20.parquet
export PPOCR_MODEL_DIR=/path/to/ppocr_onnx     # PP-OCRv6_det_medium.onnx, PP-OCRv6_rec_medium.onnx
export HPSV2_CLIP_PATH=/path/to/open_clip_pytorch_model.bin
export HPSV2_CHECKPOINT=/path/to/HPS_v2.1_compressed.pt
bash examples/diffusionnft_trainer/minimax_h3_image/run_minimax_h3_image_ocr_lora.sh
```

One node with 4 GPUs: GPUs 0-2 run the trainer, rollout and OCR reward server; GPU 3 runs the HPSv2.1 scorer.
512x512 is generated directly (no upscale canvas). LoRA rank 66, lr 1e-4, 96 images per step, 2 updates per step.
The frozen base is replicated on every rank (`VERL_OMNI_FSDP_REPLICATE_FROZEN=1`) and micro-batches are packed into one
DiT call (`VERL_OMNI_H3_IMAGE_PACK_MICRO_BATCH=1`); a step takes about 75 s on GB300.

## Result (50 steps, 64 held-out prompts, seed 42)

| step | reward | raw OCR | PP-OCR (held-out) | HPSv2.1 | text area | border-touch | contrast | edge |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.439 | 0.497 | 0.415 | 0.197 | 0.054 | 0.000 | 52.8 | 18.0 |
| 10 | 0.560 | 0.730 | 0.623 | 0.202 | 0.088 | 0.016 | 53.0 | 18.0 |
| 20 | 0.613 | 0.767 | 0.640 | 0.202 | 0.081 | 0.016 | 52.6 | 17.4 |
| 30 | 0.629 | 0.785 | 0.659 | 0.202 | 0.087 | 0.000 | 53.5 | 16.8 |
| 40 | 0.619 | 0.825 | 0.718 | 0.202 | 0.095 | 0.031 | 54.2 | 15.3 |
| 50 | 0.695 | 0.799 | 0.665 | 0.198 | 0.079 | 0.000 | 51.7 | 14.3 |

PP-OCR is a second OCR engine that is not part of the reward. Contrast and edge are grayscale std and mean edge
magnitude of the validation images.

## Known limits

- Validated at 512x512 only, one seed.
- Edge strength still drifts down (18.0 -> 14.3); HPSv2.1 does not see it.
- lr 3e-4 washes out the scene even with the gate; keep lr at 1e-4.
- Requires vllm-omni with MiniMax H3 image support (vllm-omni 0.33).
