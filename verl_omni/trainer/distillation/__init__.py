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
"""Omni distillation helpers (hidden-state nitrobrew and multimodal SDPO/OPSD self-distillation)."""

from verl_omni.trainer.distillation.self_distillation import (
    EMATeacherState,
    RepromptedTeacherSample,
    TeacherContextDecision,
    align_teacher_sequence_outputs,
    build_multimodal_teacher_prompt,
    build_neutral_teacher_outputs,
    extract_sequence_rewards,
    format_privileged_context,
    reprompt_multimodal_messages,
    select_teacher_contexts,
    summarize_self_distillation_metrics,
)

__all__ = [
    "EMATeacherState",
    "RepromptedTeacherSample",
    "TeacherContextDecision",
    "align_teacher_sequence_outputs",
    "build_multimodal_teacher_prompt",
    "build_neutral_teacher_outputs",
    "extract_sequence_rewards",
    "format_privileged_context",
    "reprompt_multimodal_messages",
    "select_teacher_contexts",
    "summarize_self_distillation_metrics",
]
