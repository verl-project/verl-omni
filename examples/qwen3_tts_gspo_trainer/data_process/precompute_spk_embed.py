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
"""Precompute the fixed clone voice's speaker x-vector ONCE, via the model's canonical
``extract_speaker_embedding`` (ECAPA over a 24 kHz mel). The same vector feeds both the vLLM-Omni
talker rollout (``voice_clone_prompt.ref_spk_embedding``, ``x_vector_only_mode=True``) and the verl
actor (speaker @ pos 6), so generation and the teacher-forced recompute condition on an identical
speaker. Saved as a JSON float list (consumed via
``actor_rollout_ref.model.override_config.tts_spk_embed_path``).

    python precompute_spk_embed.py --ref <clone_voice>.wav --out spk_embed.json
"""

import argparse
import json

import librosa
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    ap.add_argument("--ref", required=True, help="fixed clone reference wav")
    ap.add_argument("--out", required=True, help="output JSON path for the x-vector")
    args = ap.parse_args()

    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    m = Qwen3TTSModel.from_pretrained(args.model, device_map="cuda:0", dtype=torch.bfloat16)
    inner = m.model
    wav, sr = librosa.load(args.ref, sr=None, mono=True)
    with torch.no_grad():
        xvec = inner.extract_speaker_embedding(np.asarray(wav, dtype=np.float32), int(sr))
    xvec = xvec.detach().reshape(-1).float().cpu().tolist()
    with open(args.out, "w") as f:
        json.dump(xvec, f)
    print(f"saved {len(xvec)}-dim x-vector -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
