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
from verl.workers.rollout.base import _ROLLOUT_REGISTRY

# verl's vLLM ServerAdapter, unchanged: the omni_delta_sharded backend streams
# its flushes over the stock named_tensors bucketed wire (see
# verl_omni/workers/checkpoint_engine.py), so the rollout side needs no omni
# adapter subclass at any pin.
_ROLLOUT_REGISTRY[("vllm_omni", "async")] = "verl.workers.rollout.vllm_rollout.ServerAdapter"
