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
"""Patches every test process needs before any test module imports."""

from verl_omni.utils.vllm_omni.patch import patch_qwen3_tts_default_rope_read

# The pinned qwen-tts's default-rope hook reads a bare rope_theta that
# Transformers 5.16+ configs no longer declare; wrap registrations before any
# test imports qwen_tts or builds a default-rope model.
patch_qwen3_tts_default_rope_read()
