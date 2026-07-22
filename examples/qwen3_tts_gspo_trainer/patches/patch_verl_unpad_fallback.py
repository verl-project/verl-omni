# Copyright 2026 Gulp AI Inc and/or its affiliates
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
"""Patch verl's _get_attention_functions to fall back to transformers' reference unpad/pad
helpers when flash_attn is not installed. Idempotent; only needed on envs without flash-attn.

verl hard-imports from flash_attn.bert_padding on CUDA with no fallback
(verl/utils/attention_utils.py). This recipe runs attn_implementation sdpa with
use_remove_padding false, so the flash-attn kernels are never needed, only these padding
helpers. Do NOT install a fake flash_attn package instead: vllm probes for it and would pick a
broken attention backend. This routes the import to transformers.modeling_flash_attention_utils.

Usage: python patch_verl_unpad_fallback.py
"""

import pathlib
import sys

import verl

F = pathlib.Path(verl.__file__).parent / "utils" / "attention_utils.py"
s = F.read_text()
OLD = "        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input"
NEW = (
    "        try:\n"
    "            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input\n"
    "        except ImportError:\n"
    "            from einops import rearrange\n"
    "            from transformers.modeling_flash_attention_utils import (\n"
    "                _index_first_axis as index_first_axis,\n"
    "                _pad_input as pad_input,\n"
    "                _unpad_input as unpad_input,\n"
    "            )"
)

if "except ImportError:" in s and "_unpad_input as unpad_input" in s:
    print("verl unpad fallback already patched")
elif OLD in s:
    F.write_text(s.replace(OLD, NEW))
    print(f"patched verl unpad fallback (transformers) in {F}")
else:
    print("WARN: verl flash_attn import line not found; layout changed", file=sys.stderr)
    sys.exit(1)
print("PATCH_DONE")
