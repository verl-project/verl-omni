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
"""
Patches for upstream veRL to support Qwen3-Omni Thinker RL training.

These patches should eventually be upstreamed to veRL. They are applied
here so verl-omni can work without waiting for upstream merges.
"""

from verl_omni.patches.qwen3_omni import apply_all

apply_all()
