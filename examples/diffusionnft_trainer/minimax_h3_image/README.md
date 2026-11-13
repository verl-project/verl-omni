# MiniMax H3 image: DiffusionNFT OCR LoRA

DiffusionNFT LoRA training for the MiniMax H3 image model (6-step PDD sampler) on a dense-text rendering task.

## Reward

`reward.aggregation=weighted_sum` over three terms in `ocr_preference_reward.py`:

| term | weight | value |
|---|---|---|
| `term_pickscore` | 0.5 | PickScore (logit / 26) of the image against the full generation prompt |
| `term_ocr_guarded` | 0.2 | Qwen3-VL-8B OCR score (`genrm_ocr`, temperature 0, 160 response tokens) x layout guard |
| `term_hpsv2` | 0.3 | HPSv2.1 of the image against the full generation prompt |

Layout guard: 0 if a PP-OCR text box touches the frame (within 2 px), else `min(1, 0.08 / text_area)`.
A failed PP-OCR request leaves the guard at 1 and is counted in `reward/ocr/guard_error`.

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

The data rows carry the target text in `reward_model.ground_truth` and `extra_info.split` / `extra_info.index`; the
reward looks the full prompt up in `TRAIN_FILES` / `VAL_FILES` by that key.

## Single-node Ascend 950DT

Use Python 3.12, PyTorch 2.10.0, torch_npu 2.10.0.post7 and CANN 9.2.0 with the CANN, ATB and driver libraries on the runtime search path. Tested source revisions are vLLM `4ba787ec`, vLLM-Ascend `760d7ecb`, vLLM-Omni `eb8b991f` and verl `fefb0802`.

## Diffusers image packing on Ascend

The image-only Actor passes an empty audio index tensor to Diffusers. With
torch_npu 2.10.0.post7 and CANN 9.2.0 on Ascend 950DT, the last out-of-place
`index_copy` can corrupt the text and image rows already packed for the first
transformer block. The supplied patch skips that last copy on an NPU when the
audio indices are empty.

Apply the patch to the active training environment's Diffusers 0.41.0 installation.
Run these commands from the verl-omni repository root, with the environment
activated. The first command locates the actual package parent, including for
editable installs. On a previously patched installation, the reverse check
below should succeed; continue directly to the regression tests in that case.

```bash
diffusers_parent=$(python -c 'import diffusers; from pathlib import Path; print(Path(diffusers.__file__).resolve().parents[1])')
packing_patch="$PWD/examples/diffusionnft_trainer/minimax_h3_image/patches/diffusers-0.41.0-npu-empty-audio.patch"
patch \
    -d "$diffusers_parent" \
    -p1 \
    --forward \
    --dry-run \
    -i "$packing_patch"
patch \
    -d "$diffusers_parent" \
    -p1 \
    --forward \
    -i "$packing_patch"
patch \
    -d "$diffusers_parent" \
    -p1 \
    --reverse \
    --dry-run \
    -i "$packing_patch"
```

The regression suite checks the text/image rows at the first block's input,
zero-initialized LoRA on all six PDD heads, and gradients through the NPU
forward path. It uses one Ascend NPU and the H3-capable vLLM-Omni runtime.

```bash
python -m pytest \
    -q \
    tests/pipelines/test_minimax_h3_image_rollout_kernels_npu.py \
    tests/pipelines/test_minimax_h3_image_packing_npu.py
```

Activate the environment, export the asset paths described above and launch from the repository root:

```bash
bash examples/diffusionnft_trainer/minimax_h3_image/run_minimax_h3_image_ocr_lora_npu.sh
```

The launcher assigns NPUs 0–3 to a four-rank FSDP2 Actor and two colocated TP2 rollout replicas. Three Qwen3-VL OCR replicas use NPUs 4–6 and the preference scorers use NPU 7. The Actor uses parameter/optimizer CPU offload, gradient checkpointing and microbatch size 1. The default starts fresh with a 200-step limit, validation every 5 steps and checkpoint saving disabled. Set `OUTPUT_DIR` to a fresh directory. Pass trainer overrides as trailing arguments when checkpoints are needed.

At step 50 the measured validation reward was 0.60182 on Ascend and 0.603 in the published GB300 reference. The values compare the training recipe across platform-specific parallel configurations. [GPU reference](https://github.com/verl-project/verl-omni/pull/890)

The common H3 image converter, adapters and dense-text recipe derive from aoshen524's commits 46ebe8a, b5dd965 and 5216577. [Source](https://github.com/verl-project/verl-omni/pull/890/files)
