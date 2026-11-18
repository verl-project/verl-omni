# MiniMax H3 image: DiffusionNFT OCR LoRA

Last updated: 11/16/2026

DiffusionNFT LoRA training for the MiniMax H3 image model (6-step PDD sampler) on a dense-text rendering task.

## Reward

`reward.aggregation=weighted_sum` over three terms in `ocr_preference_reward.py`:

| term | weight | value |
|---|---|---|
| `term_pickscore` | 0.5 | PickScore (logit / 26) of the image against the full generation prompt |
| `term_ocr_guarded` | 0.2 | Qwen3-VL-8B OCR score (`genrm_ocr`, temperature 0, 160 response tokens) x layout guard |
| `term_hpsv2` | 0.3 | HPSv2.1 of the image against the full generation prompt |

Layout guard: 0 if a PP-OCR text box touches the frame (within 2 px), else `min(1, 0.08 / text_area)`.
It is required: with a plain OCR term the policy enlarges the text until it fills the canvas
(text area 0.054 -> 0.175, 20% of validation images touching the frame at step 50), and PickScore / HPSv2.1 do not
react to that. A failed PP-OCR request raises `RuntimeError` so the OCR term contributes 0 and is counted in
`reward/ocr/errors` (or fails the step if `required=true`).

## Setup

```bash
pip install rapidocr onnxruntime open_clip_torch
python examples/diffusionnft_trainer/minimax_h3_image/prepare_dense_text_data.py --output-dir $DATA_DIR
python verl_omni/pipelines/minimax_h3_image_diffusion_nft/checkpoint.py convert \
    --native-root $MODEL_PATH --output $ACTOR_PATH
```

- `rapidocr` must support `OCRVersion.PPOCRV6`; `PPOCR_MODEL_DIR` holds `PP-OCRv6_det_medium.onnx` and
  `PP-OCRv6_rec_medium.onnx`.
- `PICKSCORE_PATH`: `yuvalkirstain/PickScore_v1`. `CLIP_H14_PATH`: `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
  (with `open_clip_pytorch_model.bin`, the default `HPSV2_PRETRAINED_PATH`). `CUSTOM_REWARD_MODEL_PATH`:
  `HPS_v2.1_compressed.pt` from `xswu/HPSv2`. HPSv2.1 is scored by `compute_score_hpsv2`.
- Rollout needs vLLM-Omni with MiniMax H3 image support (vLLM-Omni 0.33, #878).
- Actor training needs `diffusers>=0.43.0` (where `MiniMaxH3Transformer3DModel` first shipped; the repository's
  `diffusers>=0.41.0` floor is too old for this model).

The data rows carry the target text in `reward_model.ground_truth`. The reward looks the full prompt up in
`TRAIN_FILES` / `VAL_FILES` by `(extra_info.tier, extra_info.split, extra_info.index)`.

## Run

```bash
export MODEL_PATH ACTOR_PATH DATA_DIR PPOCR_MODEL_DIR PICKSCORE_PATH CLIP_H14_PATH CUSTOM_REWARD_MODEL_PATH
bash examples/diffusionnft_trainer/minimax_h3_image/run_minimax_h3_image_ocr_lora.sh
```

One node with 4 GPUs: GPUs 0-2 run the trainer, rollout and OCR reward model; GPU 3 runs the PickScore / HPSv2.1
scorer (`PREFERENCE_SCORER_GPU`). Each of the 3 reward-loop workers starts one shared scorer process there (about
7 GB each, serving both `term_pickscore` and `term_hpsv2`) and its own pool of 8 PP-OCR processes. 512x512 is generated directly, without the upscaled canvas.
LoRA rank 66, lr 1e-4, 96 images per step, 2 updates per step. Each micro-batch runs as one packed DiT call
(`VERL_OMNI_H3_IMAGE_PACK_MICRO_BATCH=1`, requires gradient checkpointing off), and rollout normalization/LoRA
kernels stay enabled by default (`VERL_OMNI_H3_IMAGE_ROLLOUT_KERNELS=1`, set `0` to fall back to stock Diffusers).
Set `MINIMAX_H3_IMAGE_DEBUG_DIR` to dump per-step diagnostic tensors on evidence runs. Steps took 60-85 s on GB300.

## Result

50 steps, seed 42, 64 held-out prompts (a quarter of them with scenes not seen in training). Guarded OCR is the OCR
term without its weight; the `PP-OCR` column reports raw diagnostic text accuracy from the same PP-OCRv6 engine (whose
box geometry acts as the layout guard, while its text accuracy is not part of the reward); contrast and edge are the
grayscale standard deviation and mean edge magnitude of the validation images.

| step | reward | guarded OCR | raw OCR | PP-OCR | PickScore | HPSv2.1 | text area | border-touch | contrast | edge |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.530 | 0.458 | 0.497 | 0.415 | 0.756 | 0.197 | 0.054 | 0.000 | 52.8 | 18.0 |
| 10 | 0.559 | 0.579 | 0.696 | 0.587 | 0.763 | 0.203 | 0.077 | 0.016 | 53.5 | 18.4 |
| 20 | 0.582 | 0.678 | 0.849 | 0.735 | 0.767 | 0.203 | 0.096 | 0.016 | 51.8 | 18.1 |
| 30 | 0.586 | 0.704 | 0.839 | 0.732 | 0.766 | 0.202 | 0.092 | 0.000 | 51.2 | 16.7 |
| 40 | 0.584 | 0.712 | 0.764 | 0.644 | 0.759 | 0.200 | 0.076 | 0.000 | 50.0 | 16.1 |
| 50 | 0.603 | 0.794 | 0.848 | 0.719 | 0.765 | 0.201 | 0.078 | 0.000 | 49.6 | 16.0 |

With OCR x guard alone (no preference terms), lr 3e-4 washed the scene out (contrast 52.8 -> 37.0) and collapsed
after about 45 steps; lr 1e-4 avoided both.

## Known limits

- One seed, 512x512 only.
- Edge strength drifts down (18.0 -> 16.0); neither preference term reacts to it.
- The preference terms barely move (PickScore 0.756 -> 0.765); under these weights they act as an anchor and the
  guarded OCR term drives the update.
