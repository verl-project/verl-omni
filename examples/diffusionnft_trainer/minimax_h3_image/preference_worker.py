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
"""PickScore and HPSv2.1 scorer process for the preference reward terms.

One JSON request per stdin line (``{"path": image file, "prompt": full generation prompt}``), one JSON reply per
stdout line (``{"pickscore": logit / 26, "hpsv2": score}``). Runs on the GPU given by ``CUDA_VISIBLE_DEVICES`` of
this process. HPSv2.1 is ``compute_score_hpsv2`` and PickScore is ``_PickScoreInferencer.score``.

Requires ``PICKSCORE_PATH`` (``yuvalkirstain/PickScore_v1``), ``CLIP_H14_PATH``
(``laion/CLIP-ViT-H-14-laion2B-s32B-b79K``) and the ``compute_score_hpsv2`` inputs ``HPSV2_PRETRAINED_PATH``
(``open_clip_pytorch_model.bin`` of that CLIP) and ``CUSTOM_REWARD_MODEL_PATH`` (``HPS_v2.1_compressed.pt``).
"""

import json
import os
import sys


def main():
    real_stdout = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)

    import torch
    from PIL import Image

    from verl_omni.utils.reward_score.hpsv2_reward import compute_score_hpsv2
    from verl_omni.utils.reward_score.pickscore_reward import _PickScoreInferencer

    if not torch.cuda.is_available():
        raise RuntimeError("preference_worker.py needs a CUDA device")
    os.environ["REWARD_DEVICE"] = "cuda"
    pickscore = _PickScoreInferencer(
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        model_path=os.environ["PICKSCORE_PATH"],
        processor_path=os.environ["CLIP_H14_PATH"],
    )
    blank = Image.new("RGB", (64, 64))
    compute_score_hpsv2(blank, ground_truth="")
    real_stdout.write(json.dumps({"ready": True}) + "\n")
    real_stdout.flush()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            image = Image.open(request["path"]).convert("RGB")
            prompt = request["prompt"]
            with torch.no_grad():
                pick = pickscore.score([prompt], [image])
            reply = {
                "hpsv2": compute_score_hpsv2(image, ground_truth=prompt)["score"],
                "pickscore": float(pick[0].item()),
            }
        except Exception as error:
            reply = {"error": repr(error)}
        real_stdout.write(json.dumps(reply) + "\n")
        real_stdout.flush()


if __name__ == "__main__":
    main()
