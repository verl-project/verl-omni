"""HPSv2.1 scorer process used by the quality gate.

One JSON request per stdin line (``{"path": image file, "prompt": full generation prompt}``), one JSON reply per
stdout line (``{"hpsv2": cosine score}``). Runs on the GPU given by ``CUDA_VISIBLE_DEVICES`` of this process.

Requires ``HPSV2_CLIP_PATH`` (OpenCLIP ViT-H-14 laion2B ``open_clip_pytorch_model.bin``) and
``HPSV2_CHECKPOINT`` (``HPS_v2.1_compressed.pt``).
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer = _HPSv2Scorer(os.environ["HPSV2_CLIP_PATH"], os.environ["HPSV2_CHECKPOINT"], device)
    real_stdout.write(json.dumps({"ready": True}) + "\n")
    real_stdout.flush()
    for line in sys.stdin:
        request = json.loads(line)
        try:
            image = Image.open(request["path"]).convert("RGB")
            with torch.no_grad():
                out = scorer.infer([image], [request["prompt"]])
            score = torch.nn.functional.cosine_similarity(
                out["text_embeddings"][0].unsqueeze(0).float(), out["image_embeddings"][0].unsqueeze(0).float(), dim=-1
            ).item()
            reply = {"hpsv2": score}
        except Exception as error:  # noqa: BLE001
            reply = {"error": repr(error)}
        real_stdout.write(json.dumps(reply) + "\n")
        real_stdout.flush()


if __name__ == "__main__":
    main()
