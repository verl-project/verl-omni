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

"""CPU unit tests for rollout seed config resolution.

End-to-end rollout seed behavior is covered by GPU tests.
"""

from verl_omni.agent_loop.utils import _build_rollout_seed


def test_build_rollout_seed_resolution():
    assert _build_rollout_seed(42, global_steps=1) == 42
    assert _build_rollout_seed(42, global_steps=3) == 44
    assert _build_rollout_seed(None, global_steps=1) is None
