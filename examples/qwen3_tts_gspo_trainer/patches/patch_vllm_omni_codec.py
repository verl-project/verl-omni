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
"""Patch vllm-omni 0.24.0 so a single-stage AR talker emitting "codec" output works. Idempotent.

Two breakages:
1. The per-step AR postprocess (which surfaces the generated code tensors) only fires for
   engine_output_type "audio", so a "codec" stage wedges during decode.
2. OutputModality does not know "codec". Alias it to "latent", whose CONCAT_DIM0 accumulation
   stacks the per-step (frames, 16) code tensors on dim0; the "audio" alias would use
   CONCAT_LAST and corrupt the code layout.

Usage: python patch_vllm_omni_codec.py
"""

import pathlib

import vllm_omni

base = pathlib.Path(vllm_omni.__file__).parent

f = base / "worker" / "gpu_ar_model_runner.py"
s = f.read_text()
old = 'if engine_output_type == "audio" and not downstream_req_ids:'
new = 'if engine_output_type in ("audio", "codec", "latent") and not downstream_req_ids:'
if new in s:
    print("AR postprocess already patched")
else:
    assert old in s, "vllm-omni AR postprocess site changed"
    f.write_text(s.replace(old, new))
    print("patched AR postprocess (codec/latent)")

f = base / "engine" / "output_modality.py"
s = f.read_text()
if '"codec": "latent",' in s:
    print("codec modality alias already patched")
else:
    old = "_MODALITY_ALIASES: dict[str, str] = {"
    assert old in s, "vllm-omni modality alias table changed"
    f.write_text(s.replace(old, old + '\n    "codec": "latent",'))
    print("patched codec modality alias (latent)")
print("PATCH_DONE")
