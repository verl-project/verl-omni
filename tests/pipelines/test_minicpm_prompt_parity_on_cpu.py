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
"""CPU tests for MiniCPM-o prompt parity (slot collapse, bounds, packing)."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch

from verl_omni.pipelines.minicpm.prompt_parity import (
    MINICPM_AUDIO_SLOT,
    MINICPM_ENGINE_AUDIO_SLOT,
    MINICPM_ENGINE_IMAGE_SLOT,
    MINICPM_IMAGE_SLOT,
    MiniCPMMediaTokens,
    bind_minicpm_processor,
    flatten_block_content_to_slots,
)
from verl_omni.pipelines.minicpm.thinker_training_adapter import (
    MiniCPMThinkerAdapter,
    _apply_media_bounds,
    _is_packed_batch,
    _merge_packed_media,
    split_minicpm_forward_kwargs,
)

_SPECIAL_TOKENS = {
    "<image>": 100,
    "</image>": 101,
    "<unk>": 102,
    "<slice>": 103,
    "</slice>": 104,
    "<|audio_start|>": 105,
    "<|audio_end|>": 106,
    "<image_id>": 107,
    "</image_id>": 108,
    "<|im_start|>": 200,
    "<|im_end|>": 201,
}
_SPECIAL_RE = re.compile("(" + "|".join(re.escape(token) for token in _SPECIAL_TOKENS) + ")")


class _StubTokenizer:
    """Char-level tokenizer with 1:1 special-token round-trip."""

    im_start_id = _SPECIAL_TOKENS["<image>"]
    im_end_id = _SPECIAL_TOKENS["</image>"]
    slice_start_id = _SPECIAL_TOKENS["<slice>"]
    slice_end_id = _SPECIAL_TOKENS["</slice>"]
    audio_start_id = _SPECIAL_TOKENS["<|audio_start|>"]
    audio_end_id = _SPECIAL_TOKENS["<|audio_end|>"]
    unk_token = "<unk>"

    def convert_tokens_to_ids(self, token):
        return _SPECIAL_TOKENS.get(token, -1)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = []
        for part in _SPECIAL_RE.split(text):
            if not part:
                continue
            if part in _SPECIAL_TOKENS:
                ids.append(_SPECIAL_TOKENS[part])
            else:
                ids.extend(ord(char) + 1000 for char in part)
        return ids

    def decode(self, ids, skip_special_tokens=False):
        reverse = {token_id: token for token, token_id in _SPECIAL_TOKENS.items()}
        parts = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id in reverse:
                if not skip_special_tokens:
                    parts.append(reverse[token_id])
            else:
                parts.append(chr(token_id - 1000))
        return "".join(parts)

    chat_template = "<stub template>"

    def apply_chat_template(self, messages, **kwargs):
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)


class _StubProcessor:
    def __init__(self):
        self.tokenizer = _StubTokenizer()
        self.calls = []
        self.template_calls = []

    def __call__(self, text=None, images=None, audios=None, **kwargs):
        self.calls.append({"text": text, "images": images, "audios": audios, **kwargs})
        return {"input_ids": torch.tensor([self.tokenizer.encode(text if isinstance(text, str) else text[0])])}

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append({"messages": messages, **kwargs})
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)


def _tokens():
    return MiniCPMMediaTokens(_StubTokenizer())


def _expanded_prompt(num_unk=3, with_image_id=True):
    prefix = "<image_id>0</image_id>" if with_image_id else ""
    return (
        "<|im_start|>user\n"
        + prefix
        + "<image>"
        + "<unk>" * num_unk
        + "</image>"
        + "<|audio_start|>"
        + "<unk>" * 2
        + "<|audio_end|>"
        + "<|audio_start|>"
        + "<unk>" * 1
        + "<|audio_end|>"
        + "What is playing?<|im_end|>"
    )


def test_collapse_expanded_media_to_hf_and_engine_slots():
    tokens = _tokens()
    text = _expanded_prompt(num_unk=3)
    hf = tokens.collapse_to_slots(text, engine_slots=False)
    assert hf == "<|im_start|>user\n" + MINICPM_IMAGE_SLOT + MINICPM_AUDIO_SLOT + "What is playing?<|im_end|>"
    engine = tokens.collapse_to_slots(text, engine_slots=True)
    assert engine == (
        "<|im_start|>user\n" + MINICPM_ENGINE_IMAGE_SLOT + MINICPM_ENGINE_AUDIO_SLOT + "What is playing?<|im_end|>"
    )


def test_collapse_handles_slice_grid_and_missing_image_id():
    tokens = _tokens()
    text = "<image><unk></image><slice><unk></slice>\n<slice><unk></slice>rest"
    assert tokens.collapse_to_slots(text, engine_slots=False) == MINICPM_IMAGE_SLOT + "rest"


def test_dedup_pad_tokens_round_trips_to_engine_slots():
    processor = bind_minicpm_processor(_StubProcessor())
    tokenizer = processor.tokenizer
    ids = tokenizer.encode(_expanded_prompt(num_unk=3))
    decoded = tokenizer.decode(processor.dedup_pad_tokens(ids))
    assert MINICPM_ENGINE_IMAGE_SLOT in decoded
    assert MINICPM_ENGINE_AUDIO_SLOT in decoded
    assert "<unk>" not in decoded
    # The media span is gone: only the compact slot remains.
    assert decoded.count("(<image>") == 1 and decoded.count("(<audio>") == 1


def test_dedup_pad_tokens_text_only_prompt_is_untouched():
    processor = bind_minicpm_processor(_StubProcessor())
    ids = processor.tokenizer.encode("<|im_start|>user\nplain question<|im_end|>")
    assert processor.dedup_pad_tokens(ids) == ids


def test_dedup_pad_tokens_raises_when_media_tokens_do_not_form_spans():
    processor = bind_minicpm_processor(_StubProcessor())
    # A lone </image> with no <image> start: orphan end marker at unit position.
    ids = processor.tokenizer.encode("broken </image> prompt")
    with pytest.raises(ValueError, match="only valid.*inside an expanded media span"):
        processor.dedup_pad_tokens(ids)
    # Unterminated <image> block.
    ids = processor.tokenizer.encode("<image>" + "<unk>" * 3)
    with pytest.raises(ValueError, match="unterminated <image> block"):
        processor.dedup_pad_tokens(ids)


def test_dedup_pad_tokens_preserves_surrounding_ids_verbatim():
    processor = bind_minicpm_processor(_StubProcessor())
    tokenizer = processor.tokenizer
    prefix = tokenizer.encode("<|im_start|>user\nlook at this: ")
    suffix = tokenizer.encode(" and listen<|im_end|>")
    expanded = tokenizer.encode(
        "<image_id>0</image_id><image><unk><unk><unk></image>"
        + "<|audio_start|><unk><unk><|audio_end|><|audio_start|><unk><|audio_end|>"
    )
    result = processor.dedup_pad_tokens(prefix + expanded + suffix)
    # ids outside media spans pass through byte-for-byte — no decode/re-encode.
    assert result[: len(prefix)] == prefix
    assert result[-len(suffix) :] == suffix
    middle = tokenizer.decode(result[len(prefix) : len(result) - len(suffix)])
    assert middle == MINICPM_ENGINE_IMAGE_SLOT + MINICPM_ENGINE_AUDIO_SLOT


def test_dedup_pad_tokens_collapses_slice_grid_with_newlines():
    processor = bind_minicpm_processor(_StubProcessor())
    tokenizer = processor.tokenizer
    ids = tokenizer.encode("q <image><unk></image><slice><unk></slice>\n<slice><unk></slice> tail")
    result = processor.dedup_pad_tokens(ids)
    decoded = tokenizer.decode(result)
    assert decoded == "q " + MINICPM_ENGINE_IMAGE_SLOT + " tail"


def test_processor_call_adapts_audio_kwarg_and_appends_missing_slots():
    processor = bind_minicpm_processor(_StubProcessor())
    processor(text=["Which song?"], images=["img.png"], audio=[b"wav"])
    call = processor.calls[-1]
    assert call["audios"] == [b"wav"]  # verl's audio= became the remote audios=
    assert call["text"] == ["Which song?" + MINICPM_IMAGE_SLOT + MINICPM_AUDIO_SLOT]


def test_processor_call_collapses_expanded_text_before_remote_call():
    processor = bind_minicpm_processor(_StubProcessor())
    processor(text=[_expanded_prompt(num_unk=2)], images=["img.png"], audios=[b"wav"])
    call = processor.calls[-1]
    assert call["text"] == [
        "<|im_start|>user\n" + MINICPM_IMAGE_SLOT + MINICPM_AUDIO_SLOT + "What is playing?<|im_end|>"
    ]


def test_processor_call_rejects_more_slots_than_media():
    processor = bind_minicpm_processor(_StubProcessor())
    with pytest.raises(ValueError, match="image slots but only"):
        processor(text=[MINICPM_IMAGE_SLOT + MINICPM_IMAGE_SLOT + "?"], images=["img.png"])


def _packed_data():
    tokenizer = _StubTokenizer()
    sample_a = tokenizer.encode("A" + "<image><unk><unk></image>" + "a")
    sample_b = tokenizer.encode("B" + "<|audio_start|><unk><unk><unk><|audio_end|>" + "b")
    input_ids = torch.tensor([sample_a + sample_b], dtype=torch.long)
    position_ids = torch.tensor([list(range(len(sample_a))) + list(range(len(sample_b)))], dtype=torch.long)
    data = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "pixel_values": [[torch.zeros(2, 2)], []],
        "tgt_sizes": [torch.tensor([[1, 1]], dtype=torch.int32), torch.zeros(0, 2, dtype=torch.int32)],
        "audio_features": torch.zeros(1, 80, 10),
        "audio_feature_lens": [[], [torch.tensor([3])]],
        "image_bound": [[] for _ in range(2)],
        "audio_bounds": [[] for _ in range(2)],
    }
    return data, len(sample_a)


def test_is_packed_batch_detects_position_resets():
    data, _ = _packed_data()
    assert _is_packed_batch(data)
    padded = {**data, "input_ids": torch.zeros(2, 5, dtype=torch.long), "position_ids": torch.arange(5).repeat(2, 1)}
    assert not _is_packed_batch(padded)


def test_apply_media_bounds_packed_and_count_parity():
    data, sample_a_len = _packed_data()
    _apply_media_bounds(data, SimpleNamespace(processor=_StubProcessor()))
    image_bounds, audio_bounds = data["image_bound"][0], data["audio_bounds"][0]
    # Image span sits inside sample A: (<image>)(<unk>)(<unk>) at 1-based positions 2..3.
    assert image_bounds == [[2, 4]]
    audio_start = sample_a_len + 2  # 'B' then <|audio_start|>, +1 for the span start
    assert audio_bounds == [[audio_start, audio_start + 3]]


def test_apply_media_bounds_raises_on_count_mismatch():
    data, _ = _packed_data()
    data["pixel_values"] = [[torch.zeros(2, 2), torch.zeros(2, 2)], []]  # two slices, one span in ids
    with pytest.raises(ValueError, match="parity failure"):
        _apply_media_bounds(data, SimpleNamespace(processor=_StubProcessor()))


def test_merge_packed_media_folds_samples_into_one_row():
    data, sample_a_len = _packed_data()
    _apply_media_bounds(data, SimpleNamespace(processor=_StubProcessor()))
    _merge_packed_media(data)
    # One pseudo-row (remote embedders iterate rows, bs == 1 under packing)...
    assert len(data["pixel_values"]) == 1
    assert len(data["image_bound"]) == 1 and len(data["audio_bounds"]) == 1
    # ...holding every slice/span of every sample, not flattened past the row.
    pixel_row = data["pixel_values"][0]
    assert len(pixel_row) == 1 and pixel_row[0].shape == (2, 2)
    assert data["image_bound"] == [[[2, 4]]]
    audio_start = sample_a_len + 2
    assert data["audio_bounds"] == [[[audio_start, audio_start + 3]]]
    assert data["audio_feature_lens"] == [[3]]
    assert data["tgt_sizes"][0].shape == (1, 2)
    # Idempotent: merging the already-merged form changes nothing.
    again = {key: value for key, value in data.items()}
    _merge_packed_media(again)
    assert again["pixel_values"][0][0] is data["pixel_values"][0][0] and len(again["pixel_values"]) == 1
    assert again["image_bound"] == data["image_bound"]
    assert again["audio_bounds"] == data["audio_bounds"]


def test_prepare_model_inputs_packed_drops_attention_mask():
    data, _ = _packed_data()
    model_inputs = {**{key: value for key, value in data.items() if key not in ("image_bound", "audio_bounds")}}
    model_inputs["attention_mask"] = torch.ones_like(model_inputs["input_ids"])
    packed = MiniCPMThinkerAdapter.prepare_model_inputs(
        model_inputs, micro_batch=None, model_config=SimpleNamespace(processor=_StubProcessor())
    )
    assert "attention_mask" not in packed
    assert len(packed["data"]["image_bound"]) == 1


def test_configure_model_rejects_non_45_checkpoints():
    module = torch.nn.Module()
    module.config = SimpleNamespace(version="2.6")
    with pytest.raises(ValueError, match="MiniCPM-o 4.5 checkpoints only"):
        MiniCPMThinkerAdapter.configure_model(module, SimpleNamespace())


def _block_messages():
    return [
        {"role": "system", "content": "Answer with <answer>X</answer>."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/tmp/frame.png"},
                {"type": "audio", "audio": "/tmp/clip.wav"},
                {"type": "text", "text": "Which song is playing?"},
            ],
        },
    ]


def test_flatten_block_content_renders_slots_in_order():
    flattened = flatten_block_content_to_slots(_block_messages())
    assert flattened[0]["content"] == "Answer with <answer>X</answer>."
    assert flattened[1]["content"] == MINICPM_IMAGE_SLOT + MINICPM_AUDIO_SLOT + "Which song is playing?"


def test_flatten_block_content_does_not_mutate_input():
    messages = _block_messages()
    flatten_block_content_to_slots(messages)
    assert isinstance(messages[1]["content"], list)
    assert messages[1]["content"][0] == {"type": "image", "image": "/tmp/frame.png"}


def test_flatten_block_content_rejects_unsupported_blocks():
    with pytest.raises(ValueError, match="cannot render content block type 'video'"):
        flatten_block_content_to_slots([{"role": "user", "content": [{"type": "video", "video": "/tmp/x.mp4"}]}])
    with pytest.raises(ValueError, match="content blocks must be dicts"):
        flatten_block_content_to_slots([{"role": "user", "content": ["raw string"]}])
    # string content passes through untouched
    assert flatten_block_content_to_slots([{"role": "user", "content": "plain"}])[0]["content"] == "plain"


def test_wrapper_apply_chat_template_flattens_and_delegates():
    processor = bind_minicpm_processor(_StubProcessor())
    messages = _block_messages()
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sent = processor.template_calls[-1]["messages"]
    assert sent[1]["content"] == MINICPM_IMAGE_SLOT + MINICPM_AUDIO_SLOT + "Which song is playing?"
    assert isinstance(messages[1]["content"], list)  # caller's messages untouched
    assert rendered == (
        "<system>Answer with <answer>X</answer>.</system>"
        "<user>" + MINICPM_IMAGE_SLOT + MINICPM_AUDIO_SLOT + "Which song is playing?</user>"
    )


def test_wrapper_apply_chat_template_string_content_untouched():
    processor = bind_minicpm_processor(_StubProcessor())
    messages = [{"role": "user", "content": "plain question"}]
    processor.apply_chat_template(messages)
    assert processor.template_calls[-1]["messages"][0]["content"] == "plain question"


def _flatten_ints(nested):
    out = []
    for item in nested:
        if isinstance(item, list | tuple):
            out.extend(_flatten_ints(item))
        elif hasattr(item, "tolist"):
            out.extend(_flatten_ints(item.tolist()))
        else:
            out.append(int(item))
    return out


def test_prepare_model_inputs_discards_stale_processor_bounds():
    # verl's extract path hands the processor's per-sample bounds to the model
    # call even after rmpad flattened the ids to (1, total); trusting them
    # trips the packed guard. They must be dropped and re-derived instead.
    # (Bounds nesting is asserted in the merge tests; here it is flattened so
    # the assertion targets the values, which is what this fix owns.)
    data, sample_a_len = _packed_data()
    model_inputs = {key: value for key, value in data.items() if key not in ("image_bound", "audio_bounds")}
    model_inputs["image_bound"] = [torch.tensor([[7, 9]]), torch.zeros(0, 2, dtype=torch.long)]  # stale
    model_inputs["audio_bounds"] = [torch.zeros(0, 2, dtype=torch.long), torch.tensor([[99, 103]])]  # stale
    model_inputs["attention_mask"] = torch.ones_like(model_inputs["input_ids"])
    packed = MiniCPMThinkerAdapter.prepare_model_inputs(
        model_inputs, micro_batch=None, model_config=SimpleNamespace(processor=_StubProcessor())
    )
    assert "attention_mask" not in packed  # packed layout confirmed: no tripwire fired
    assert _flatten_ints(packed["data"]["image_bound"]) == [2, 4]  # stale 7/9 gone
    audio_start = sample_a_len + 2
    assert _flatten_ints(packed["data"]["audio_bounds"]) == [audio_start, audio_start + 3]  # 99/103 gone


def test_prepare_model_inputs_padded_batch_also_rederives_bounds():
    tokenizer = _StubTokenizer()
    row = tokenizer.encode("A" + "<image><unk><unk></image>" + "a")
    model_inputs = {
        "input_ids": torch.tensor([row, row], dtype=torch.long),
        "position_ids": torch.arange(len(row)).repeat(2, 1),
        "attention_mask": torch.ones(2, len(row), dtype=torch.long),
        "pixel_values": [[torch.zeros(2, 2)], [torch.zeros(2, 2)]],
        "tgt_sizes": [torch.tensor([[1, 1]], dtype=torch.int32), torch.zeros(0, 2, dtype=torch.int32)],
        "audio_features": [],
        "audio_feature_lens": [[], []],
        "image_bound": [torch.tensor([[99, 103]])],  # stale: wrong rows, wrong span
    }
    prepared = MiniCPMThinkerAdapter.prepare_model_inputs(
        model_inputs, micro_batch=None, model_config=SimpleNamespace(processor=_StubProcessor())
    )
    assert prepared["data"]["image_bound"] == [[[2, 4]], [[2, 4]]]


def test_split_classifies_image_sizes_as_data_not_llm_kwargs():
    data, llm_kwargs = split_minicpm_forward_kwargs(
        {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "position_ids": torch.arange(4).unsqueeze(0),
            "image_sizes": [[(8, 8)]],
            "attention_mask": torch.ones(1, 4),
        }
    )
    assert data["image_sizes"] == [[(8, 8)]]
    assert "image_sizes" not in llm_kwargs


def test_prepare_model_inputs_drops_image_sizes_entirely():
    data, _ = _packed_data()
    model_inputs = {key: value for key, value in data.items() if key not in ("image_bound", "audio_bounds")}
    model_inputs["image_sizes"] = [[(8, 8)], [(8, 8)]]
    prepared = MiniCPMThinkerAdapter.prepare_model_inputs(
        model_inputs, micro_batch=None, model_config=SimpleNamespace(processor=_StubProcessor())
    )
    assert "image_sizes" not in prepared
    assert "image_sizes" not in prepared["data"]


def test_processor_call_normalizes_empty_media_to_none():
    processor = bind_minicpm_processor(_StubProcessor())
    processor(text=["text-only prompt"], images=[], audio=[])
    call = processor.calls[-1]
    assert call["images"] is None and call["audios"] is None
    assert call["text"] == ["text-only prompt"]  # no slots appended for empty media


def test_processor_call_keeps_nonempty_media():
    processor = bind_minicpm_processor(_StubProcessor())
    processor(text=["<image>./</image> listen"], images=["img.png"], audio=[b"wav"])
    call = processor.calls[-1]
    assert call["images"] == ["img.png"] and call["audios"] == [b"wav"]


def test_bind_upgrades_processor_in_place():
    processor = _StubProcessor()
    bound = bind_minicpm_processor(processor)
    assert bound is processor
    assert type(processor).__name__ == "MiniCPMOParityProcessor"
    # Native inheritance, no proxying: isinstance and plain attributes work.
    assert isinstance(processor, _StubProcessor)
    assert processor.calls == []
    assert processor.tokenizer is processor.tokenizer


def test_bound_processor_survives_dill_roundtrip():
    dill = pytest.importorskip("dill")
    processor = bind_minicpm_processor(_StubProcessor())
    restored = dill.loads(dill.dumps(processor))
    # datasets ships the dataset (holding the processor) into .filter workers;
    # the upgraded class must unpickle with its behaviors intact.
    restored(text=["Which song?"], images=["img.png"], audio=[b"wav"])
    call = restored.calls[-1]
    assert call["audios"] == [b"wav"]
    assert call["text"] == ["Which song?" + MINICPM_IMAGE_SLOT + MINICPM_AUDIO_SLOT]


def test_processor_call_fixes_tensor_nulling_convert_to_tensors():
    # Mirror of the remote MiniCPMOBatchFeature bug: the converter returns
    # only in the not-is_tensor branch, so already-tensor leaves become None.
    from transformers.feature_extraction_utils import BatchFeature

    class _RemoteBuggyFeature(BatchFeature):
        def convert_to_tensors(self, tensor_type=None):
            if tensor_type is None:
                return self

            def converter(value):
                if not torch.is_tensor(value):
                    return torch.tensor(value)

            for key, value in self.items():
                self[key] = converter(value)
            return self

    class _BuggyStubProcessor(_StubProcessor):
        def __call__(self, text=None, images=None, audios=None, **kwargs):
            self.calls.append({"text": text, "images": images, "audios": audios, **kwargs})
            return _RemoteBuggyFeature({"input_ids": torch.tensor([[1, 2]]), "raw_lens": [3]})

    processor = bind_minicpm_processor(_BuggyStubProcessor())
    feature = processor(text=["hi"])

    # Unpatched, the buggy converter nulls the tensor leaf.
    assert _RemoteBuggyFeature({"input_ids": torch.tensor([[1, 2]])}).convert_to_tensors("pt")["input_ids"] is None

    converted = feature.convert_to_tensors("pt")
    assert torch.is_tensor(converted["input_ids"])  # tensor values survive
    assert converted["input_ids"].tolist() == [[1, 2]]
    assert torch.is_tensor(converted["raw_lens"])  # non-tensors still convert (contract kept)
    assert dict(converted)["input_ids"].tolist() == [[1, 2]]  # verl's dict(feature.convert_to_tensors(...)) path


def test_upgrade_batch_feature_passes_through_stock_and_non_features():
    from transformers.feature_extraction_utils import BatchFeature

    from verl_omni.pipelines.minicpm.prompt_parity import _upgrade_batch_feature

    plain = {"input_ids": torch.tensor([1])}
    assert _upgrade_batch_feature(plain) is plain
    stock = BatchFeature({"input_ids": torch.tensor([1])})
    assert _upgrade_batch_feature(stock) is stock  # no override -> untouched


def test_apply_media_bounds_refuses_media_ids_without_features():
    data, _ = _packed_data()
    stripped = {key: value for key, value in data.items() if key not in ("image_bound", "audio_bounds")}
    stripped["pixel_values"] = [[], []]
    stripped["audio_features"] = []
    stripped["audio_feature_lens"] = [[], []]
    with pytest.raises(ValueError, match="no media features survived"):
        _apply_media_bounds(stripped, SimpleNamespace(processor=_StubProcessor()))

    # Text-only ids with no features stay silent.
    tokenizer = _StubTokenizer()
    text_only = {
        "input_ids": torch.tensor([tokenizer.encode("plain question")], dtype=torch.long),
        "position_ids": torch.arange(11).unsqueeze(0),
        "pixel_values": [[], []],
        "audio_features": [],
        "audio_feature_lens": [[], []],
    }
    _apply_media_bounds(text_only, SimpleNamespace(processor=_StubProcessor()))  # no raise


class _FakeProcessorMixin:
    """Mimics transformers >= 5's generic ProcessorMixin: no tokenizer fallback."""

    def apply_chat_template(self, messages, **kwargs):
        raise ValueError("Cannot use apply_chat_template because this processor does not have a chat template")


class _RemoteStyleProcessor(_FakeProcessorMixin):
    """The remote MiniCPMOProcessor shape: does NOT override apply_chat_template."""

    def __init__(self):
        self.tokenizer = _StubTokenizer()
        self.calls = []

    def __call__(self, text=None, images=None, audios=None, **kwargs):
        self.calls.append({"text": text, "images": images, "audios": audios, **kwargs})
        return {"input_ids": torch.tensor([1])}


def test_apply_chat_template_renders_via_tokenizer_when_processor_has_none():
    processor = bind_minicpm_processor(_RemoteStyleProcessor())
    rendered = processor.apply_chat_template(_block_messages(), tokenize=False, add_generation_prompt=True)
    # Rendered by the tokenizer the flattened slots survive, instead of the
    # generic base raising "no chat template" (which silently emptied the
    # dataset through filter_overlong_prompts).
    assert f"<user>{MINICPM_IMAGE_SLOT}{MINICPM_AUDIO_SLOT}Which song is playing?</user>" in rendered
    assert "<system>Answer with <answer>X</answer>.</system>" in rendered


def test_apply_chat_template_delegates_when_processor_class_overrides():
    processor = bind_minicpm_processor(_StubProcessor())
    processor.apply_chat_template(_block_messages())
    sent = processor.template_calls[-1]["messages"]
    assert sent[1]["content"] == MINICPM_IMAGE_SLOT + MINICPM_AUDIO_SLOT + "Which song is playing?"


def test_apply_chat_template_raises_loudly_when_nothing_can_render():
    processor = bind_minicpm_processor(_RemoteStyleProcessor())
    processor.tokenizer = None  # binding needs one; the apply path must not
    with pytest.raises(ValueError, match="No renderable chat template"):
        processor.apply_chat_template(_block_messages())

    templated = bind_minicpm_processor(_RemoteStyleProcessor())
    templated.tokenizer.chat_template = None  # tokenizer present but template-less
    with pytest.raises(ValueError, match="No renderable chat template"):
        templated.apply_chat_template(_block_messages())


def test_convert_to_tensors_keeps_ragged_media_features():
    from transformers.feature_extraction_utils import BatchFeature

    from verl_omni.pipelines.minicpm.prompt_parity import _safe_convert_to_tensors

    feature = BatchFeature(
        {
            # Per-sample patch packs with varying shapes/counts — what the
            # MiniCPM processor actually emits and what the stock converter
            # refuses ("only one element tensors can be converted...").
            "pixel_values": [
                [torch.zeros(3, 4, 4), torch.zeros(3, 2, 2)],
                [torch.zeros(3, 8, 8), torch.zeros(3, 5, 5)],
            ],
            "input_ids": [[1, 2, 3]],
            "raw_lens": [5, 7],
        }
    )
    with pytest.raises(ValueError, match="pixel_values"):  # the stock behavior we must survive
        BatchFeature.convert_to_tensors(feature, "pt")

    converted = _safe_convert_to_tensors(feature, "pt")
    assert torch.is_tensor(converted["input_ids"]) and converted["input_ids"].tolist() == [[1, 2, 3]]
    assert torch.is_tensor(converted["raw_lens"])  # uniform leaves still convert (stock contract)
    pixel_values = converted["pixel_values"]
    assert isinstance(pixel_values, list) and len(pixel_values) == 2  # ragged structure kept
    assert pixel_values[0][0].shape == (3, 4, 4) and pixel_values[1][1].shape == (3, 5, 5)
    assert torch.is_tensor(pixel_values[0][0])


def test_convert_to_tensors_never_stacks_uniform_slice_lists():
    # transformers' as_tensor wrapper stacks a list of same-shape tensors
    # (plain torch.as_tensor refuses). The processor emits batch-style
    # pixel_values=[[slice x S]]; a uniform per-sample list that stacks to
    # one 4-D tensor counts as a single slice downstream and trips the
    # media parity check only for images whose slices are all identical.
    from transformers.feature_extraction_utils import BatchFeature

    from verl_omni.pipelines.minicpm.media_inputs import sample_pixel_slices
    from verl_omni.pipelines.minicpm.prompt_parity import _safe_convert_to_tensors

    uniform = torch.zeros(3, 14, 14504)
    feature = BatchFeature(
        {
            "pixel_values": [
                [uniform.clone() for _ in range(5)],  # 960x720-style: all slices identical
                [torch.zeros(3, 4, 4), torch.zeros(3, 2, 2)],  # ragged sibling sample
            ]
        }
    )

    converted = _safe_convert_to_tensors(feature, "pt")
    pixel_values = converted["pixel_values"]
    # The uniform sample must stay a 5-slice list, not stack to T(5, 3, 14, 14504).
    assert isinstance(pixel_values[0], list) and len(pixel_values[0]) == 5
    assert all(slice_.shape == (3, 14, 14504) for slice_ in pixel_values[0])
    # Slice counting agrees with the id-side span count for both samples.
    assert [len(sample_pixel_slices(sample)) for sample in pixel_values] == [5, 2]


def test_convert_to_tensors_leaves_unconvertible_scalars_alone():
    from transformers.feature_extraction_utils import BatchFeature

    from verl_omni.pipelines.minicpm.prompt_parity import _safe_convert_to_tensors

    feature = BatchFeature({"note": "metadata", "ids": [1, 2]})
    converted = _safe_convert_to_tensors(feature, "pt")
    assert converted["note"] == "metadata"  # non-container leaves stay as-is
    assert torch.is_tensor(converted["ids"])
