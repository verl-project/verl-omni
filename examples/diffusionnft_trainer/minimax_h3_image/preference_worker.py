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
stdout line (``{"pickscore": logit / 26, "hpsv2": cosine}``). Runs on the GPU given by ``CUDA_VISIBLE_DEVICES`` of
this process.

Requires ``PICKSCORE_PATH`` (``yuvalkirstain/PickScore_v1``), ``CLIP_H14_PATH``
(``laion/CLIP-ViT-H-14-laion2B-s32B-b79K``, holding ``open_clip_pytorch_model.bin``) and ``HPSV2_CHECKPOINT``
(``HPS_v2.1_compressed.pt``).
"""

import json
import os
import sys


def main():
    real_stdout = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)

    import torch
    from PIL import Image

    from verl_omni.utils.reward_score.hpsv2_reward import _HPSv2Scorer
    from verl_omni.utils.reward_score.pickscore_reward import _pairwise_pickscore, _PickScoreInferencer

    if not torch.cuda.is_available():
        raise RuntimeError("preference_worker.py needs a CUDA device")
    device = torch.device("cuda")
    clip_h = os.environ["CLIP_H14_PATH"]
    hpsv2 = _HPSv2Scorer(os.path.join(clip_h, "open_clip_pytorch_model.bin"), os.environ["HPSV2_CHECKPOINT"], device)
    pickscore = _PickScoreInferencer(
        device=device, dtype=torch.bfloat16, model_path=os.environ["PICKSCORE_PATH"], processor_path=clip_h
    )
    real_stdout.write(json.dumps({"ready": True}) + "\n")
    real_stdout.flush()
    for line in sys.stdin:
        request = json.loads(line)
        try:
            image = Image.open(request["path"]).convert("RGB")
            prompt = request["prompt"]
            with torch.no_grad():
                hps = hpsv2.infer([image], [prompt])
                pick = pickscore.infer([prompt], [image])
            reply = {
                "hpsv2": torch.nn.functional.cosine_similarity(
                    hps["text_embeddings"][0:1].float(), hps["image_embeddings"][0:1].float(), dim=-1
                ).item(),
                "pickscore": _pairwise_pickscore(
                    pick["text_embeddings"][0:1], pick["image_embeddings"][0:1], pick["logit_scale"], score_divisor=26.0
                ).item(),
            }
        except Exception as error:
            reply = {"error": repr(error)}
        real_stdout.write(json.dumps(reply) + "\n")
        real_stdout.flush()


if __name__ == "__main__":
    main()
