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
"""CPU tests for MiniCPM offline DPO sample transforms."""

from __future__ import annotations

import torch

from verl_omni.utils.dataset.minicpm_transform import IGNORE_INDEX, process_minicpm_sample


class _CharTokenizer:
    unk_token_id = -1

    def convert_tokens_to_ids(self, token):
        return {"<image>": 1001, "<video>": 1002, "<audio>": 1003}.get(token, self.unk_token_id)

    def __call__(
        self, text, add_special_tokens=False, return_offsets_mapping=False, return_tensors=None, padding=False
    ):
        del add_special_tokens, padding
        input_ids = [ord(ch) for ch in text]
        result = {"input_ids": input_ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        if return_tensors == "pt":
            result["input_ids"] = torch.tensor([input_ids], dtype=torch.long)
        return result


class _MiniCPMProcessor:
    def __init__(self):
        self.tokenizer = _CharTokenizer()

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        del tokenize, kwargs
        return "".join(f"{message['role']}:{message['content']}\n" for message in messages)

    def __call__(self, text, **kwargs):
        del kwargs
        return self.tokenizer(text, return_tensors="pt")


def test_minicpm_transform_labels_only_final_assistant_answer():
    sample = {
        "conversations": [
            ["user", ("text", "first question")],
            ["assistant", ("text", "old answer")],
            ["user", ("text", "final question")],
            ["assistant", ("text", "new answer")],
        ]
    }

    output = process_minicpm_sample(sample, processor=_MiniCPMProcessor())[0]
    input_text = "".join(chr(token_id) for token_id in output["input_ids"].tolist())
    labelled_chars = "".join(
        chr(token_id)
        for token_id, label in zip(output["input_ids"].tolist(), output["labels"].tolist(), strict=True)
        if label != IGNORE_INDEX
    )

    assert "old answer" in input_text
    assert labelled_chars == "new answer"
    assert output["attention_mask"].shape == output["input_ids"].shape
    assert output["position_ids"].shape == output["input_ids"].shape
