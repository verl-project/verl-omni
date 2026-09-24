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
"""Reward manager that keeps MiniCPM-o's answer tags in the decoded string."""

from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager

from verl_omni.models.transformers.minicpm_o import patch_minicpm_answer_tags

# TODO (mike): refactor later — the demotion belongs with the checkpoint's tokenizer setup, not a manager subclass.


@register("minicpm_naive")
class MiniCPMNaiveRewardManager(NaiveRewardManager):
    """``NaiveRewardManager`` that demotes MiniCPM-o's special answer tags first.

    The checkpoint registers ``<answer>``/``</answer>`` as special tokens while the
    base class decodes with ``skip_special_tokens=True``, so the tags never reach the
    choice reward and every score is 0. The V1 omni trainer resolves managers by name
    from verl's registry (it runs verl's stock reward loop worker, which builds its own
    tokenizer), so opting in via ``reward.reward_manager.name=minicpm_naive`` is the
    one place the demotion can land. No-op for tokenizers without the tags.
    """

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        patch_minicpm_answer_tags(tokenizer)
        super().__init__(config, tokenizer, compute_score, reward_router_address, reward_model_tokenizer)
