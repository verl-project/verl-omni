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

import re
import unicodedata
from typing import Any

_ANSWER_PATTERN = re.compile(r"<answer\s*>(.*?)</answer\s*>", flags=re.IGNORECASE | re.DOTALL)
_SURROUNDING_PUNCTUATION = " \t\r\n\"'`“”‘’.,;:!?！？。，；："


def _normalize_answer(value: Any) -> str:
    value = unicodedata.normalize("NFKC", str(value))
    value = " ".join(value.split())
    return value.strip(_SURROUNDING_PUNCTUATION).casefold()


def _ground_truth_fields(ground_truth: Any) -> tuple[str, list[str], list[int]]:
    if isinstance(ground_truth, str):
        return ground_truth, [ground_truth], [0]
    if not isinstance(ground_truth, dict):
        raise TypeError(f"AudioMCQ ground truth must be a string or dictionary, got {type(ground_truth)}")

    answer = str(ground_truth["answer"])
    choices = [str(choice) for choice in ground_truth["choices"]]
    normalized_answer = _normalize_answer(answer)
    matching_indices = [index for index, choice in enumerate(choices) if _normalize_answer(choice) == normalized_answer]
    if not matching_indices:
        raise ValueError("AudioMCQ answer is absent from choices")

    if "correct_choice_indices" in ground_truth:
        correct_choice_indices = [int(index) for index in ground_truth["correct_choice_indices"]]
        if set(correct_choice_indices) != set(matching_indices):
            raise ValueError("AudioMCQ correct_choice_indices disagree with answer-equivalent choices")
    elif "correct_choice_index" in ground_truth:
        correct_choice_index = int(ground_truth["correct_choice_index"])
        if correct_choice_index not in matching_indices:
            raise ValueError("AudioMCQ answer and correct_choice_index disagree")
        correct_choice_indices = matching_indices
    else:
        correct_choice_indices = matching_indices
    if not correct_choice_indices or any(not 0 <= index < len(choices) for index in correct_choice_indices):
        raise ValueError(f"correct_choice_indices {correct_choice_indices} are outside {len(choices)} choices")
    return answer, choices, correct_choice_indices


def compute_score(solution_str: str, ground_truth: Any, **kwargs) -> dict[str, float]:
    """Score an AudioMCQ response using exact answer text or an exact choice label."""
    answer, _, correct_choice_indices = _ground_truth_fields(ground_truth)
    matches = _ANSWER_PATTERN.findall(solution_str)
    if not matches:
        return {"score": 0.0, "content_correct": 0.0, "format_valid": 0.0}

    normalized_matches = [_normalize_answer(match) for match in matches]
    conflicting_tags = len(set(normalized_matches)) > 1
    format_valid = len(matches) == 1 and bool(normalized_matches[-1])
    if conflicting_tags or not normalized_matches[-1]:
        return {"score": 0.0, "content_correct": 0.0, "format_valid": float(format_valid)}

    predicted = normalized_matches[-1]
    normalized_answer = _normalize_answer(answer)
    labels = [chr(ord("a") + index) for index in correct_choice_indices]
    label_forms = {form for label in labels for form in (label, f"{label})", f"({label})")}
    content_correct = predicted == normalized_answer or predicted in label_forms

    return {
        "score": float(content_correct),
        "content_correct": float(content_correct),
        "format_valid": float(format_valid),
    }
