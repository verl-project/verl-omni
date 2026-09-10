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
"""Keep MiniCPM-o prompts consistent across the three verl renderings.

A MiniCPM-o RL prompt exists in three forms that must agree:

1. Training ids — the agent-loop builder renders the chat template, then the
   remote processor expands media slots in place::
       <image_id>0</image_id><image><unk>*query_num</image>   (per image)
       <|audio_start|><unk>*tokens<|audio_end|>               (per audio)
2. Engine prompt — vLLM-Omni's token-id path searches the compact slots
   ``(<image>./</image>)`` / ``(<audio>./</audio>)`` (WITH parens, see
   ``MiniCPMO45OmniLLMProcessingInfo`` patterns) and expands them itself.
3. Actor recompute — verl decodes ids and re-feeds the processor; MiniCPM
   features (pixel_values / tgt_sizes / audio mels) are position-independent,
   but the remote processor asserts the text carries one slot per media item.

This module collapses expanded spans back to slots (rendering 2), normalizes
text fed to the remote processor (rendering 3), and derives the
``image_bound`` / ``audio_bounds`` spans that ``MiniCPMO.forward`` scatters
into — mirroring the remote ``get_inputs_ids`` scan.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Slot forms. The HF processor splits on the bare forms; vLLM-Omni's token-id
# prompt updates search the parenthesized forms.
MINICPM_IMAGE_SLOT = "<image>./</image>"
MINICPM_AUDIO_SLOT = "<audio>./</audio>"
MINICPM_ENGINE_IMAGE_SLOT = "(<image>./</image>)"
MINICPM_ENGINE_AUDIO_SLOT = "(<audio>./</audio>)"

# Remote tokenizer attributes (see processing_minicpmo.py get_inputs_ids).
_MEDIA_TOKEN_ID_ATTRS = {
    "im_start": "im_start_id",
    "im_end": "im_end_id",
    "slice_start": "slice_start_id",
    "slice_end": "slice_end_id",
    "audio_start": "audio_start_id",
    "audio_end": "audio_end_id",
}


class MiniCPMMediaTokens:
    """Special-token strings/ids for one MiniCPM-o tokenizer instance."""

    def __init__(self, tokenizer: Any):
        missing = [attr for attr in _MEDIA_TOKEN_ID_ATTRS.values() if getattr(tokenizer, attr, None) is None]
        if missing:
            raise AttributeError(
                "The MiniCPM-o tokenizer wrapper must expose "
                f"{sorted(_MEDIA_TOKEN_ID_ATTRS.values())} (missing {missing}); "
                "got an incompatible tokenizer."
            )
        self.ids = {name: int(getattr(tokenizer, attr)) for name, attr in _MEDIA_TOKEN_ID_ATTRS.items()}
        unk_id = tokenizer.convert_tokens_to_ids("<unk>")
        if not isinstance(unk_id, int) or unk_id < 0:
            raise ValueError("MiniCPM-o tokenizer has no <unk> token; cannot recognize expanded media spans.")
        self.ids["unk"] = int(unk_id)
        self.strings = {name: tokenizer.decode([token_id]) for name, token_id in self.ids.items()}
        self.tokenizer = tokenizer
        self._image_span_re = self._compile_image_span_re()
        self._audio_span_re = self._compile_audio_span_re()
        # Engine slot encodings are fixed strings — encode once, splice many.
        self._engine_image_slot_ids = tokenizer.encode(MINICPM_ENGINE_IMAGE_SLOT, add_special_tokens=False)
        self._engine_audio_slot_ids = tokenizer.encode(MINICPM_ENGINE_AUDIO_SLOT, add_special_tokens=False)

    def _compile_image_span_re(self) -> re.Pattern:
        s = self.strings
        image_block = re.escape(s["im_start"]) + f"(?:{re.escape(s['unk'])})+" + re.escape(s["im_end"])
        grid_cell = re.escape(s["slice_start"]) + f"(?:{re.escape(s['unk'])})+" + re.escape(s["slice_end"])
        # The img-id prefix (<image_id>N</image_id>) may or may not precede the
        # block depending on use_image_id; tolerate both, plus an optional
        # slice grid (max_slice_nums>1 checkpoints).
        id_open, id_close = self._image_id_token_strings()
        prefix = f"(?:{re.escape(id_open)}\\d+{re.escape(id_close)})?" if id_open and id_close else ""
        grid = f"(?:\\s*{grid_cell})*"
        return re.compile(f"{prefix}{image_block}{grid}")

    def _image_id_token_strings(self) -> tuple[str | None, str | None]:
        tokenizer = self.tokenizer
        id_open = tokenizer.convert_tokens_to_ids("<image_id>")
        id_close = tokenizer.convert_tokens_to_ids("</image_id>")
        if isinstance(id_open, int) and id_open >= 0 and isinstance(id_close, int) and id_close >= 0:
            return tokenizer.decode([id_open]), tokenizer.decode([id_close])
        return None, None

    def _compile_audio_span_re(self) -> re.Pattern:
        s = self.strings
        chunk = re.escape(s["audio_start"]) + f"(?:{re.escape(s['unk'])})+" + re.escape(s["audio_end"])
        return re.compile(f"(?:{chunk})+")

    def has_media_tokens(self, input_ids: torch.Tensor | list[int]) -> bool:
        media_ids = set(self.ids.values())
        return any(int(token_id) in media_ids for token_id in input_ids.reshape(-1).tolist())

    def collapse_to_slots(self, text: str, *, engine_slots: bool) -> str:
        """Replace every expanded media span with its compact slot string."""
        image_slot = MINICPM_ENGINE_IMAGE_SLOT if engine_slots else MINICPM_IMAGE_SLOT
        audio_slot = MINICPM_ENGINE_AUDIO_SLOT if engine_slots else MINICPM_AUDIO_SLOT
        text = self._audio_span_re.sub(audio_slot, text)
        return self._image_span_re.sub(image_slot, text)

    def count_slots(self, text: str) -> tuple[int, int]:
        return (
            len(re.findall(re.escape(MINICPM_IMAGE_SLOT), text)),
            len(re.findall(re.escape(MINICPM_AUDIO_SLOT), text)),
        )

    def _image_id_token_ids(self) -> tuple[int | None, int | None]:
        tokenizer = self.tokenizer
        id_open = tokenizer.convert_tokens_to_ids("<image_id>")
        id_close = tokenizer.convert_tokens_to_ids("</image_id>")
        if isinstance(id_open, int) and id_open >= 0 and isinstance(id_close, int) and id_close >= 0:
            return int(id_open), int(id_close)
        return None, None

    def _is_whitespace_token(self, token_id: int) -> bool:
        decoded = self.tokenizer.decode([token_id])
        return bool(decoded) and decoded.strip() == ""

    def _scan_to(self, ids: list[int], start: int, end_id: int, what: str) -> int:
        """Index just past the next ``end_id``; raises when the block is unterminated."""
        index = start
        while index < len(ids) and ids[index] != end_id:
            index += 1
        if index >= len(ids):
            raise ValueError(f"Unbalanced MiniCPM media ids: unterminated {what} block starting at token {start}.")
        return index + 1

    def collapse_ids_to_slots(self, input_ids: list[int]) -> list[int]:
        """Splice engine slot ids directly in id space, leaving all else untouched.

        Rebuilding the engine prompt via decode -> text -> re-encode risks BPE
        boundary drift around the media markers (this tokenizer has known
        encode/decode inconsistencies there), which would silently desync the
        engine context from the training ids. Instead the expanded spans are
        recognized by marker-token ids — an optional ``<image_id>N</image_id>``
        prefix, the ``<image>...</image>`` block plus newline-tolerant
        ``<slice>...</slice>`` grid cells, or consecutive
        ``<|audio_start|>...<|audio_end|>`` chunks — and replaced by the
        pre-encoded parenthesized engine slots. Every id outside a span is
        passed through verbatim; malformed marker leftovers raise instead of
        producing a mismatched prompt.
        """
        ids = [int(token_id) for token_id in input_ids]
        image_id_open, image_id_close = self._image_id_token_ids()
        marker = self.ids
        # Tokens that are only valid inside a span; a span consumes its
        # contents atomically, so meeting one at unit position is malformed.
        orphan_markers = {
            marker["unk"],
            marker["im_end"],
            marker["slice_start"],
            marker["slice_end"],
            marker["audio_end"],
        }
        if image_id_close is not None and image_id_open is not None:
            orphan_markers.add(image_id_close)

        out: list[int] = []
        i = 0
        while i < len(ids):
            token = ids[i]
            if token in orphan_markers:
                raise ValueError(
                    f"Unbalanced MiniCPM media ids: token {token} at position {i} is only valid "
                    "inside an expanded media span; refusing to build a mismatched engine prompt."
                )
            if image_id_open is not None and token == image_id_open:
                if image_id_close is None:
                    raise ValueError("Unbalanced MiniCPM media ids: <image_id> without </image_id>.")
                after = self._scan_to(ids, i + 1, image_id_close, "<image_id>")
                if after < len(ids) and ids[after] == marker["im_start"]:
                    i = after  # prefix consumed; the image block below handles the unit
                    continue
                out.extend(ids[i:after])  # stray id pair with no image block: keep verbatim
                i = after
                continue
            if token == marker["im_start"]:
                i = self._scan_to(ids, i + 1, marker["im_end"], "<image>")
                while True:  # slice grid cells of the same image, newline-tolerant
                    probe = i
                    while (
                        probe < len(ids)
                        and self._is_whitespace_token(ids[probe])
                        and probe + 1 < len(ids)
                        and ids[probe + 1] == marker["slice_start"]
                    ):
                        probe += 1
                    if probe < len(ids) and ids[probe] == marker["slice_start"]:
                        i = self._scan_to(ids, probe + 1, marker["slice_end"], "<slice>")
                    else:
                        break
                out.extend(self._engine_image_slot_ids)
                continue
            if token == marker["audio_start"]:
                i = self._scan_to(ids, i + 1, marker["audio_end"], "<audio>")
                while i < len(ids) and ids[i] == marker["audio_start"]:  # consecutive chunks
                    i = self._scan_to(ids, i + 1, marker["audio_end"], "<audio>")
                out.extend(self._engine_audio_slot_ids)
                continue
            out.append(token)
            i += 1
        return out

    def derive_media_bounds(self, input_ids_1d: torch.Tensor) -> tuple[list[list[int]], list[list[int]]]:
        """Scan expanded ids for media spans, mirroring the remote scan.

        Returns ``(image_bounds, audio_bounds)`` with ``[start, end)`` pairs;
        audio starts one token after ``<|audio_start|>`` so the bound covers
        only the ``<unk>`` run the embeddings replace.
        """
        ids = input_ids_1d.reshape(-1)
        image_start = (ids == self.ids["im_start"]) | (ids == self.ids["slice_start"])
        image_end = (ids == self.ids["im_end"]) | (ids == self.ids["slice_end"])
        image_start_idx = torch.nonzero(image_start).reshape(-1)
        image_end_idx = torch.nonzero(image_end).reshape(-1)
        if len(image_start_idx) != len(image_end_idx):
            raise ValueError(
                f"Unbalanced image span markers: {len(image_start_idx)} starts vs {len(image_end_idx)} ends."
            )
        image_bounds = torch.hstack([(image_start_idx + 1).unsqueeze(-1), image_end_idx.unsqueeze(-1)]).tolist()

        audio_start_idx = torch.nonzero(ids == self.ids["audio_start"]).reshape(-1)
        audio_end_idx = torch.nonzero(ids == self.ids["audio_end"]).reshape(-1)
        if len(audio_start_idx) != len(audio_end_idx):
            raise ValueError(
                f"Unbalanced audio span markers: {len(audio_start_idx)} starts vs {len(audio_end_idx)} ends."
            )
        audio_bounds = torch.hstack([(audio_start_idx + 1).unsqueeze(-1), audio_end_idx.unsqueeze(-1)]).tolist()
        return image_bounds, audio_bounds


@lru_cache(maxsize=8)
def resolve_media_tokens(processor: Any) -> MiniCPMMediaTokens:
    """Resolve (and cache) media tokens from a configured MiniCPM-o processor."""
    tokenizer = getattr(processor, "tokenizer", processor)
    return MiniCPMMediaTokens(tokenizer)


def flatten_block_content_to_slots(messages: list[dict]) -> list[dict]:
    """Render verl's OpenAI-style content blocks into MiniCPM-o slot strings.

    verl's ``RLHFDataset._build_messages`` turns ``<image>``/``<audio>`` markers
    into structured blocks; the MiniCPM-o 4.5 chat template is the plain Qwen3
    text template and string-concatenates ``message.content`` (list content
    raises ``TypeError``). Media blocks become the processor's slot markers in
    content order; messages with string content pass through unchanged and the
    caller's message dicts are never mutated.
    """
    flattened: list[dict] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            flattened.append(message)
            continue
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                raise ValueError(f"MiniCPM-o content blocks must be dicts, got {block!r}.")
            block_type = block.get("type")
            if block_type == "image":
                parts.append(MINICPM_IMAGE_SLOT)
            elif block_type == "audio":
                parts.append(MINICPM_AUDIO_SLOT)
            elif block_type == "text":
                parts.append(block.get("text", ""))
            else:
                raise ValueError(
                    f"MiniCPM-o chat template cannot render content block type {block_type!r}; "
                    "supported: image, audio, text."
                )
        flattened.append({**message, "content": "".join(parts)})
    return flattened


def _parity_apply_chat_template(self, messages, **kwargs):
    # The MiniCPM-o template only renders string content; flatten verl's
    # blocks (image/audio -> slot markers) at this processor boundary — the
    # single choke point for both rollout renders and the dataset's doc2len.
    base = type(self).__bases__[0]
    return base.apply_chat_template(self, flatten_block_content_to_slots(messages), **kwargs)


def _parity_dedup_pad_tokens(self, input_ids: list[int]) -> list[int]:
    # Collapse happens entirely in id space: re-encoding decoded text risks
    # BPE drift that would silently desync the engine prompt from the
    # training ids. Text-only prompts come back byte-identical.
    tokens: MiniCPMMediaTokens = self._parity_tokens
    if not tokens.has_media_tokens(torch.tensor(input_ids)):
        return list(input_ids)
    return tokens.collapse_ids_to_slots(list(input_ids))


def _parity_normalize_text(tokens: MiniCPMMediaTokens, text: str, images: Any, audios: Any) -> str:
    # verl always renders one prompt per processor call (text=[str]).
    normalized = tokens.collapse_to_slots(text, engine_slots=False)
    if images is not None:
        n_images = len(images) if not isinstance(images, str | bytes) else 1
        have_images, _ = tokens.count_slots(normalized)
        if have_images > n_images:
            raise ValueError(f"Text carries {have_images} image slots but only {n_images} images were given.")
        normalized += MINICPM_IMAGE_SLOT * (n_images - have_images)
    if audios is not None:
        n_audios = len(audios) if not isinstance(audios, str | bytes) else 1
        _, have_audios = tokens.count_slots(normalized)
        if have_audios > n_audios:
            raise ValueError(f"Text carries {have_audios} audio slots but only {n_audios} audios were given.")
        normalized += MINICPM_AUDIO_SLOT * (n_audios - have_audios)
    return normalized


def _parity_call(self, text=None, images=None, audio=None, audios=None, **kwargs):
    if audio is not None and audios is None:
        audios = audio
    # The remote processor expresses "no media" as None (its __call__
    # branches on `is not None`; audio_feature_extract indexes audios[0]).
    # verl forwards empty containers, so normalize them at this boundary.
    if isinstance(images, list | tuple) and len(images) == 0:
        images = None
    if isinstance(audios, list | tuple) and len(audios) == 0:
        audios = None
    tokens: MiniCPMMediaTokens = self._parity_tokens
    if isinstance(text, list):
        text = [
            _parity_normalize_text(tokens, item, images, audios) if isinstance(item, str) else item for item in text
        ]
    elif isinstance(text, str):
        text = _parity_normalize_text(tokens, text, images, audios)
    base = type(self).__bases__[0]
    return _upgrade_batch_feature(base.__call__(self, text=text, images=images, audios=audios, **kwargs))


def _safe_convert_to_tensors(self, tensor_type=None):
    # The remote MiniCPMOBatchFeature override drops the `return value` for
    # already-tensor leaves, so a single convert_to_tensors("pt") nulls every
    # tensor feature (pixel_values, audio_features, bounds, lens, tgt_sizes).
    # The stock transformers BatchFeature implementation is correct — delegate
    # to it instead of reimplementing.
    from transformers.feature_extraction_utils import BatchFeature

    return BatchFeature.convert_to_tensors(self, tensor_type)


def _upgrade_batch_feature(feature: Any) -> Any:
    """Fix the remote BatchFeature's tensor-nulling convert_to_tensors in place.

    verl stores processor output via ``dict(feature.convert_to_tensors("pt"))``;
    with the remote bug every already-tensor leaf maps to None, silently
    wiping all media features so every training forward runs text-only while
    the rollout stays multimodal. The same in-place class upgrade as the
    processor itself: ``convert_to_tensors`` is re-pointed at the stock
    (correct) transformers implementation. Non-BatchFeature results and
    classes that no longer override the method pass through untouched.
    """
    from transformers.feature_extraction_utils import BatchFeature

    if not isinstance(feature, BatchFeature):
        return feature
    if type(feature).convert_to_tensors is BatchFeature.convert_to_tensors:
        return feature  # not overridden (e.g. already fixed upstream) — nothing to do
    feature.__class__ = type(
        "MiniCPMOBatchFeatureSafe",
        (type(feature),),
        {"convert_to_tensors": _safe_convert_to_tensors},
    )
    return feature


def bind_minicpm_processor(processor: Any) -> Any:
    """Upgrade a MiniCPM-o processor in place with the RL parity behaviors.

    A runtime subclass of the processor's own class (assigned via
    ``__class__``) rather than a wrapping proxy: attribute access stays
    native — inherited methods, properties, isinstance checks, and pickle/dill
    probes behave exactly as the wrapped processor's do, with no
    ``__getattr__`` delegation to guard or maintain. The overrides are
    module-level functions so the dynamic class stays reference-picklable
    for datasets/dill worker shipping.

    - ``__call__``: adapts verl's ``audio=`` kwarg to the remote ``audios=``,
      normalizes empty media to None, collapses expanded spans found in text,
      and appends canonical slots when media is present but the text lost its
      slots (features are position-independent, so appended slots only
      satisfy the remote one-slot-per-media assert).
    - ``apply_chat_template``: flattens verl's block content to the string
      form the MiniCPM-o template can render.
    - ``dedup_pad_tokens``: the AR strategy's pre-engine hook collapses the
      expanded spans of the agent-loop ids into vLLM's parenthesized slots,
      in id space.
    """
    parity_cls = type(
        "MiniCPMOParityProcessor",
        (type(processor),),
        {
            "_parity_tokens": resolve_media_tokens(processor),
            "__call__": _parity_call,
            "apply_chat_template": _parity_apply_chat_template,
            "dedup_pad_tokens": _parity_dedup_pad_tokens,
        },
    )
    processor.__class__ = parity_cls
    return processor
