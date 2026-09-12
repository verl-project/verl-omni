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

from typing import Optional

from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension


class PoolingRewardModelWorkerExtension(vLLMColocateWorkerExtension):
    """Keep verl's generation-only model patches out of pooling workers."""

    def monkey_patch_model(self, vocab_size: int, banned_token_ids: Optional[list[int]] = None):
        del vocab_size, banned_token_ids
