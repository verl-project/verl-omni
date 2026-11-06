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
"""Pin-compat patch contracts exercised on CPU."""

from types import SimpleNamespace

from verl_omni.utils.vllm_omni.patch import patch_qwen3_tts_default_rope_read


def test_default_rope_registry_wraps_bare_rope_theta_reads():
    """The pinned qwen-tts's default-rope hook reads a bare rope_theta; the
    registry wrapper must fall back to the config's rope_parameters value and
    keep the registered initializer's own math."""

    def bare_reader(config, device=None, seq_len=None, layer_type=None):
        del device, seq_len, layer_type
        return config.rope_theta

    patch_qwen3_tts_default_rope_read()

    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    prior = ROPE_INIT_FUNCTIONS.get("default")
    try:
        ROPE_INIT_FUNCTIONS["default"] = bare_reader
        wrapped = ROPE_INIT_FUNCTIONS["default"]

        moved = SimpleNamespace(rope_parameters={"rope_theta": 321.0, "rope_type": "default"})
        assert wrapped(moved) == 321.0
        assert moved.rope_theta == 321.0
        assert wrapped(SimpleNamespace(rope_theta=7.0)) == 7.0
    finally:
        if prior is None:
            del ROPE_INIT_FUNCTIONS["default"]
        else:
            ROPE_INIT_FUNCTIONS["default"] = prior
