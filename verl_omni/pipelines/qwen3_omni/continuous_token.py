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
"""Register the Continuous Token builder family for Qwen3-Omni checkpoints.

Qwen3-Omni's chat template is ChatML: every turn ends with ``<|im_end|>\\n`` while
generation stops at ``<|im_end|>`` — the boundary protocol the Qwen3-VL builder
implements (it re-inserts the trailing newline when merging tool/user turns). verl
keeps no ``qwen3_omni_moe`` entry in its model-type registry, so agent-loop runs
would resolve to the generic VL builder, which drops the newline and then fails
verl's tokenization sanity check because merged multi-turn streams must reproduce
the chat template at every boundary. The registration is an import side effect
consumed by ``verl.utils.tokenizer.continuous_token_wiring``; ``setdefault`` keeps
any future verl-native registration authoritative.
"""

from verl.utils.tokenizer.continuous_token_wiring import _MODEL_TYPE_TO_FAMILY, ContinuousTokenModelFamily

_MODEL_TYPE_TO_FAMILY.setdefault("qwen3_omni_moe", ContinuousTokenModelFamily.QWEN3_VL)
