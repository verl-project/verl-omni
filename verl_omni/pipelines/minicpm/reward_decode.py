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
"""Keep ``<answer>`` tags in the reward decode for checkpoints that mark them special.

MiniCPM-o 4.5 registers ``<answer>``/``</answer>`` as special tokens; verl's
reward loop builds its own tokenizer (never through the adapter's
``configure_tokenizer``) and decodes with ``skip_special_tokens=True``, so
the tags are stripped and every choice-reward score is 0 while the answers
are correct. The stock ``NaiveRewardManager`` is wrapped once to demote the
tags on the tokenizer it decodes with — a no-op for tokenizers without
special answer tags, so non-MiniCPM runs are unaffected and recipes keep
``reward.reward_manager.name=naive``.

Installed at ``verl_omni`` package import (this module rides the
``pipelines.minicpm`` import): the trainer resolves the manager name
eagerly in the task-runner during ``_setup``, before any reward worker
imports the custom reward function.
"""

from verl_omni.models.transformers.minicpm_o import keep_answer_tags_when_decoding

_PATCHED_ATTR = "_minicpm_keeps_answer_tags"


def install_reward_decode_patch() -> None:
    from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager

    if getattr(NaiveRewardManager, _PATCHED_ATTR, False):
        return
    original_init = NaiveRewardManager.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        keep_answer_tags_when_decoding(self.tokenizer)

    NaiveRewardManager.__init__ = __init__
    setattr(NaiveRewardManager, _PATCHED_ATTR, True)


install_reward_decode_patch()
