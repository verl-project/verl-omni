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

import numpy as np
import pytest
import torch

import verl_omni.utils.dataset.minicpm_transform as minicpm_transform
from verl_omni.utils.dataset.minicpm_transform import (
    _MINICPM_AUDIO_SLOT,
    _MINICPM_IMAGE_SLOT,
    IGNORE_INDEX,
    _call_processor,
    _expand_processor_slots,
    _inject_processor_media_slots,
    process_minicpm_sample,
)


class _CharTokenizer:
    unk_token_id = -1

    def convert_tokens_to_ids(self, token):
        return {"<image>": 1001, "<video>": 1002, "<audio>": 1003}.get(token, self.unk_token_id)

    def encode(self, text):
        return [ord(ch) for ch in text]

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        del tokenize, kwargs
        return "".join(f"{message['role']}:{message['content']}\n" for message in messages)

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

    def __call__(self, text, **kwargs):
        del kwargs
        return self.tokenizer(text, return_tensors="pt")


def test_minicpm_transform_uses_tokenizer_apply_chat_template():
    sample = {
        "conversations": [
            ["user", ("text", "question")],
            ["assistant", ("text", "answer")],
        ]
    }

    output = process_minicpm_sample(sample, processor=_MiniCPMProcessor())[0]
    input_text = "".join(chr(token_id) for token_id in output["input_ids"].tolist())
    assert "user:question" in input_text
    assert "assistant:answer" in input_text


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
    assert output["position_ids"].dtype == torch.int64
    assert torch.equal(output["position_ids"], torch.arange(output["input_ids"].shape[-1]))
    # image_bound stays present (empty) as verl's minicpm list-layout sentinel;
    # the adapter re-derives media containers and bounds from the ids.
    assert "image_bound" in output
    assert output["image_bound"].numel() == 0
    assert "pixel_values" not in output
    assert "tgt_sizes" not in output
    assert "audio_bounds" not in output


def test_inject_processor_media_slots_rewrites_image_and_audio_markers():
    text = "<|im_start|>user\n<image>What is shown?\n<|im_start|>assistant\nanswer"
    injected = _inject_processor_media_slots(
        text,
        image_count=1,
        audio_count=0,
        video_count=0,
        use_audio_in_video=False,
    )
    assert injected.count(_MINICPM_IMAGE_SLOT) == 1
    assert "<image>What" not in injected

    audio_text = "<|im_start|>user\n<audio>What sound?\n<|im_start|>assistant\nanswer"
    audio_injected = _inject_processor_media_slots(
        audio_text,
        image_count=0,
        audio_count=1,
        video_count=0,
        use_audio_in_video=False,
    )
    assert audio_injected.count(_MINICPM_AUDIO_SLOT) == 1
    assert "<audio>" not in audio_injected.replace(_MINICPM_AUDIO_SLOT, "")

    dropped = _inject_processor_media_slots(
        "<|im_start|>user\nWhat sound?\n<|im_start|>assistant\nanswer",
        image_count=0,
        audio_count=1,
        video_count=0,
        use_audio_in_video=False,
    )
    assert dropped.endswith(_MINICPM_AUDIO_SLOT)
    assert dropped.count(_MINICPM_AUDIO_SLOT) == 1


class _RecordingMiniCPMProcessor(_MiniCPMProcessor):
    def __init__(self):
        super().__init__()
        self.last_kwargs: dict = {}

    def __call__(self, text, **kwargs):
        self.last_kwargs = kwargs
        return super().__call__(text, **kwargs)


def test_call_processor_aliases_sample_rate_to_sampling_rate():
    processor = _RecordingMiniCPMProcessor()
    _call_processor(processor, text="hi", images=[], audios=[], sample_rate=8000)
    assert processor.last_kwargs["sampling_rate"] == 8000


def test_call_processor_does_not_drop_media_via_tokenizer_fallback():
    class _RejectingProcessor:
        tokenizer = _CharTokenizer()

        def __call__(self, *args, **kwargs):
            del args, kwargs
            raise TypeError("unsupported MiniCPM processor signature")

    with pytest.raises(TypeError, match="Refusing a text-only tokenizer fallback"):
        _call_processor(_RejectingProcessor(), text="hi", images=["/tmp/x.png"], audios=[])


def test_call_processor_text_only_can_use_tokenizer_fallback():
    class _RejectingProcessor:
        tokenizer = _CharTokenizer()

        def __call__(self, *args, **kwargs):
            del args, kwargs
            raise TypeError("unsupported MiniCPM processor signature")

    output = _call_processor(_RejectingProcessor(), text="hi", images=[], audios=[])
    assert output["input_ids"].tolist() == [[ord("h"), ord("i")]]


def test_process_minicpm_sample_rejects_video_rows():
    sample = {
        "conversations": [
            ["user", ("video", None), ("text", "what happens?")],
            ["assistant", ("text", "answer")],
        ],
        "videos": ["/tmp/clip.mp4"],
    }
    with pytest.raises(ValueError, match="does not support video rows"):
        process_minicpm_sample(sample, processor=_MiniCPMProcessor())


def test_process_minicpm_sample_requires_conversations():
    with pytest.raises(ValueError, match="non-empty 'conversations'"):
        process_minicpm_sample({"images": []}, processor=_MiniCPMProcessor())


def test_process_minicpm_sample_accepts_falsy_use_audio_in_video():
    sample = {
        "conversations": [
            ["user", ("text", "question")],
            ["assistant", ("text", "answer")],
        ]
    }
    output = process_minicpm_sample(sample, processor=_MiniCPMProcessor(), use_audio_in_video=False)[0]
    assert output["input_ids"].numel() > 0


def test_call_processor_prefers_explicit_sampling_rate_over_sample_rate():
    processor = _RecordingMiniCPMProcessor()
    _call_processor(
        processor,
        text="hi",
        images=[],
        audios=[],
        sample_rate=8000,
        sampling_rate=16000,
    )
    assert processor.last_kwargs["sampling_rate"] == 16000


class _MockImageProcessor:
    def get_slice_image_placeholder(self, image_size, image_id, max_slice_nums, use_image_id):
        del image_size, image_id, max_slice_nums, use_image_id
        return "<image_start>" + ("U" * 8) + "<image_end>"


class _ExpandingMiniCPMProcessor:
    def __init__(self):
        self.tokenizer = _CharTokenizer()
        self.image_processor = _MockImageProcessor()

    def get_audio_placeholder(self, audio_lens, chunk_input=True, chunk_length=1):
        del chunk_length
        if chunk_input:
            return "<audio_start>" + ("U" * 10) + "<audio_end>" + "<audio_start>" + ("U" * 5) + "<audio_end>"
        return "<audio_start>" + ("U" * 15) + "<audio_end>"

    def __call__(self, text, audios=None, images=None, **kwargs):
        del kwargs, images
        audios = audios or []
        image_sizes = [(1, 1)] if _MINICPM_IMAGE_SLOT in text else []
        expanded = _expand_processor_slots(
            self,
            text,
            image_sizes=image_sizes,
            audios=audios,
            stream_input=False,
        )
        return self.tokenizer(expanded, return_tensors="pt")


def test_minicpm_transform_labels_skip_expanded_audio_placeholder(monkeypatch):
    monkeypatch.setattr(
        minicpm_transform,
        "_fetch_minicpm_audios",
        lambda sample, **kwargs: [np.zeros(1600, dtype=np.float32)],
    )
    sample = {
        "conversations": [
            ["user", ("audio", None), ("text", " listen")],
            ["assistant", ("text", "preferred")],
        ],
        "audios": ["/tmp/dummy.wav"],
    }

    processor = _ExpandingMiniCPMProcessor()
    output = process_minicpm_sample(sample, processor=processor)[0]
    labelled_chars = "".join(
        chr(token_id)
        for token_id, label in zip(output["input_ids"].tolist(), output["labels"].tolist(), strict=True)
        if label != IGNORE_INDEX
    )
    assert labelled_chars == "preferred"
    assert "U" not in labelled_chars


def test_minicpm_labels_follow_slot_appended_after_template_drops_it(monkeypatch):
    """Labels use the string the processor tokenizes, including a slot restored at the end."""

    monkeypatch.setattr(
        minicpm_transform,
        "_fetch_minicpm_audios",
        lambda sample, **kwargs: [np.zeros(1600, dtype=np.float32)],
    )

    class _DroppingTokenizer(_CharTokenizer):
        def apply_chat_template(self, messages, tokenize=False, **kwargs):
            rendered = super().apply_chat_template(messages, tokenize=tokenize, **kwargs)
            return rendered.replace(_MINICPM_AUDIO_SLOT, "").replace(_MINICPM_IMAGE_SLOT, "")

    class _DroppingProcessor(_ExpandingMiniCPMProcessor):
        def __init__(self):
            super().__init__()
            self.tokenizer = _DroppingTokenizer()
            self.seen_text = None

        def __call__(self, text, audios=None, images=None, **kwargs):
            self.seen_text = text
            return super().__call__(text, audios=audios, images=images, **kwargs)

    sample = {
        "conversations": [
            ["user", ("audio", None), ("text", "what is this")],
            ["assistant", ("text", "preferred")],
        ],
        "audios": ["/tmp/dummy.wav"],
    }
    processor = _DroppingProcessor()
    output = process_minicpm_sample(sample, processor=processor)[0]
    labelled_chars = "".join(
        chr(token_id)
        for token_id, label in zip(output["input_ids"].tolist(), output["labels"].tolist(), strict=True)
        if label != IGNORE_INDEX
    )

    assert processor.seen_text.endswith(_MINICPM_AUDIO_SLOT)
    assert _MINICPM_AUDIO_SLOT not in processor.seen_text[: -len(_MINICPM_AUDIO_SLOT)]
    assert labelled_chars == "preferred"
    assert "U" not in labelled_chars
