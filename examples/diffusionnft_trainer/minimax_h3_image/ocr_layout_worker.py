# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PP-OCR text-box geometry server used by the layout guard.

One JSON request per stdin line (``{"path": image file}``), one JSON reply per stdout line:
``{"area": sum of line-box areas / image area, "border": 1 if any box is within OCR_GUARD_BORDER_PX of the
frame, "n": boxes, "line_h": median box height in px}``.

Requires ``rapidocr`` and ``onnxruntime`` plus the PP-OCRv6 medium det/rec ONNX files in ``PPOCR_MODEL_DIR``.
"""

import json
import os
import sys

BORDER = float(os.environ.get("OCR_GUARD_BORDER_PX", "2"))


def build():
    from rapidocr import RapidOCR
    from rapidocr.utils.typings import ModelType, OCRVersion

    model_dir = os.environ["PPOCR_MODEL_DIR"]
    return RapidOCR(
        params={
            "Det.model_path": os.path.join(model_dir, "PP-OCRv6_det_medium.onnx"),
            "Rec.model_path": os.path.join(model_dir, "PP-OCRv6_rec_medium.onnx"),
            "Global.model_root_dir": model_dir,
            "Global.log_level": "warning",
            "Global.use_cls": False,
            "Det.ocr_version": OCRVersion.PPOCRV6,
            "Det.model_type": ModelType.MEDIUM,
            "Rec.ocr_version": OCRVersion.PPOCRV6,
            "Rec.model_type": ModelType.MEDIUM,
            "EngineConfig.onnxruntime.intra_op_num_threads": int(os.environ.get("PPOCR_THREADS", "2")),
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    )


def polygon_area(poly):
    area = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area += float(x1) * float(y2) - float(x2) * float(y1)
    return abs(area) / 2.0


def measure(ocr, path):
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("RGB")
    width, height = image.size
    result = ocr(np.asarray(image)[:, :, ::-1].copy())
    boxes = result.boxes if result.boxes is not None else []
    area, border, heights = 0.0, 0, []
    for poly in boxes:
        xs = [float(p[0]) for p in poly]
        ys = [float(p[1]) for p in poly]
        area += polygon_area(poly)
        heights.append(max(ys) - min(ys))
        if min(xs) <= BORDER or min(ys) <= BORDER or max(xs) >= width - 1 - BORDER or max(ys) >= height - 1 - BORDER:
            border = 1
    heights.sort()
    return {
        "area": area / float(width * height),
        "border": border,
        "n": len(heights),
        "line_h": heights[len(heights) // 2] if heights else 0.0,
    }


def main():
    real_stdout = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    ocr = build()
    real_stdout.write(json.dumps({"ready": True}) + "\n")
    real_stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            reply = measure(ocr, json.loads(line)["path"])
        except Exception as error:
            reply = {"area": 0.0, "border": 0, "n": 0, "line_h": 0.0, "error": f"{type(error).__name__}: {error}"[:200]}
        real_stdout.write(json.dumps(reply) + "\n")
        real_stdout.flush()


if __name__ == "__main__":
    main()
