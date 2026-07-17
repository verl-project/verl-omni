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
"""Patch qwen-tts 0.1.1 modeling code to run on transformers 5.x. Idempotent.

qwen-tts pins transformers==4.57.3, but vllm 0.24 dropped transformers 4 support, so this repo
installs qwen-tts with --no-deps and fixes the four transformers-5 breakages in place:
check_model_inputs became a bare decorator, config attribute access became strict,
ROPE_INIT_FUNCTIONS lost its "default" entry, and the mask helpers renamed input_embeds to
inputs_embeds and dropped cache_position.

Usage: python patch_qwen_tts_tf5.py [qwen_tts_pkg_dir]
"""

import importlib.util
import pathlib
import re
import sys

if len(sys.argv) > 1:
    pkg_dir = pathlib.Path(sys.argv[1])
else:
    # find_spec locates the package without importing it (importing would execute the very
    # modeling code this script exists to fix).
    spec = importlib.util.find_spec("qwen_tts")
    assert spec is not None and spec.origin, "qwen_tts is not installed"
    pkg_dir = pathlib.Path(spec.origin).parent

# All modeling files that use transformers internals (talker + speech-tokenizer decoder).
FILES = [
    pkg_dir / "core/models/modeling_qwen3_tts.py",
    pkg_dir / "core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py",
]

_ROPE_HELPER = (
    "def _qtts_default_rope_init(config, device=None, **kw):\n"
    "    import torch\n"
    "    base = getattr(config, 'rope_theta', 10000.0) or 10000.0\n"
    "    dim = getattr(config, 'head_dim', None) or (config.hidden_size // config.num_attention_heads)\n"
    "    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64)"
    ".to(device=device, dtype=torch.float) / dim))\n"
    "    return inv_freq, 1.0\n\n\n"
)


def patch_file(path):
    if not path.exists():
        print("skip (missing)", path.name)
        return
    s = orig = path.read_text()
    # 1) tf5 made check_model_inputs a bare decorator.
    s = s.replace("@check_model_inputs()", "@check_model_inputs")
    # 2) tf5 strict config attribute access (no default pad_token_id).
    s = s.replace("config.pad_token_id", 'getattr(config, "pad_token_id", None)')
    # 3) tf5 dropped "default" from ROPE_INIT_FUNCTIONS; route default/None to a canonical init.
    if "ROPE_INIT_FUNCTIONS[self.rope_type]" in s:
        if "_qtts_default_rope_init" not in s:
            # Insert after the module logger line; inserting before the first class can split
            # decorators from their class.
            anchor = s.find("\nlogger = logging.get_logger(__name__)\n")
            if anchor != -1:
                i = anchor + len("\nlogger = logging.get_logger(__name__)\n") + 1
            else:
                m = re.search(r"^(?:@.*\n)*class ", s, re.M)
                i = m.start() if m else 0
            s = s[:i] + "\n" + _ROPE_HELPER + s[i:]
        s = s.replace(
            "ROPE_INIT_FUNCTIONS[self.rope_type]",
            "(ROPE_INIT_FUNCTIONS.get(self.rope_type) or _qtts_default_rope_init)",
        )
    # 4) tf5 mask helpers take inputs_embeds (renamed) and no longer accept cache_position.
    s = s.replace('"input_embeds": inputs_embeds,', '"inputs_embeds": inputs_embeds,')
    s = s.replace(
        '"attention_mask": attention_mask,\n                "cache_position": cache_position,',
        '"attention_mask": attention_mask,',
    )
    s = s.replace(
        "            input_embeds=inputs_embeds,\n"
        "            attention_mask=attention_mask,\n"
        "            cache_position=cache_position,\n",
        "            inputs_embeds=inputs_embeds,\n            attention_mask=attention_mask,\n",
    )
    if s != orig:
        path.write_text(s)
        print("patched", path.name)
    else:
        print("nochange", path.name)


for f in FILES:
    patch_file(f)
print("PATCH_DONE")
