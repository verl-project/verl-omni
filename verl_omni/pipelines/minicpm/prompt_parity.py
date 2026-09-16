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
"""Keep the three renderings of a MiniCPM-o prompt consistent.

One RL prompt has to exist as agent-loop ids (media expanded in place by the
remote processor), as vLLM-Omni's engine prompt (compact parenthesized slots it
expands itself), and as the actor's recompute (ids decoded and re-fed to the
processor). This module collapses expanded spans back to slots, normalizes the
text the processor receives, and derives the ``image_bound`` / ``audio_bounds``
spans ``MiniCPMO.forward`` scatters into.
"""

from __future__ import annotations

import numbers
import re
from functools import lru_cache
from typing import Any

import torch

__all__ = [
    "bind_minicpm_processor",
    "resolve_media_tokens",
]

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


@lru_cache(maxsize=8)
def resolve_media_tokens(processor: Any) -> MiniCPMMediaTokens:
    """Resolve (and cache) media tokens from a configured MiniCPM-o processor."""
    tokenizer = getattr(processor, "tokenizer", processor)
    return MiniCPMMediaTokens(tokenizer)


def bind_minicpm_processor(processor: Any) -> Any:
    """Upgrade a MiniCPM-o processor in place with the RL parity behaviors.

    A runtime subclass of the processor's own class (assigned via ``__class__``)
    rather than a proxy, so attribute access, isinstance, and pickle/dill probes
    stay native and the dynamic class stays reference-picklable for dataset workers.

    Args:
        processor: The processor loaded by the training adapter.

    Returns:
        The same processor instance, with ``__call__``, ``apply_chat_template``,
        and ``dedup_pad_tokens`` rebound.
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


class MiniCPMMediaTokens:
    """Special-token strings/ids for one MiniCPM-o tokenizer instance."""

    def __init__(self, tokenizer: Any):
        """Resolve the marker tokens, span regexes, and engine slot ids for ``tokenizer``."""
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
        self._image_id_ids = self._resolve_image_id_ids()
        self._media_ids = frozenset(self.ids.values())
        # Markers only valid inside a span: meeting one at unit position is malformed.
        self._orphan_markers = frozenset(
            {self.ids[name] for name in ("unk", "im_end", "slice_start", "slice_end", "audio_end")}
            | ({self._image_id_ids[1]} if self._image_id_ids[1] is not None else set())
        )
        self._image_span_re = self._compile_image_span_re()
        self._audio_span_re = self._compile_audio_span_re()
        # Engine slot encodings are fixed strings — encode once, splice many.
        self._engine_image_slot_ids = tokenizer.encode(MINICPM_ENGINE_IMAGE_SLOT, add_special_tokens=False)
        self._engine_audio_slot_ids = tokenizer.encode(MINICPM_ENGINE_AUDIO_SLOT, add_special_tokens=False)

    def has_media_tokens(self, input_ids: torch.Tensor | list[int]) -> bool:
        """True when any id is an expanded-media marker or placeholder."""
        return any(int(token_id) in self._media_ids for token_id in input_ids.reshape(-1).tolist())

    def collapse_to_slots(self, text: str, *, engine_slots: bool) -> str:
        """Replace every expanded media span with its compact slot string."""
        image_slot = MINICPM_ENGINE_IMAGE_SLOT if engine_slots else MINICPM_IMAGE_SLOT
        audio_slot = MINICPM_ENGINE_AUDIO_SLOT if engine_slots else MINICPM_AUDIO_SLOT
        text = self._audio_span_re.sub(audio_slot, text)
        return self._image_span_re.sub(image_slot, text)

    def count_slots(self, text: str) -> tuple[int, int]:
        """``(images, audios)`` slot counts in ``text``."""
        return (
            text.count(MINICPM_IMAGE_SLOT),
            text.count(MINICPM_AUDIO_SLOT),
        )

    def collapse_ids_to_slots(self, input_ids: list[int]) -> list[int]:
        """Replace the ids' expanded media spans with vLLM-Omni's parenthesized slots.

        Args:
            input_ids: Agent-loop prompt ids carrying expanded media spans.

        Returns:
            The prompt ids with each span replaced by its engine slot, every other
            id untouched; malformed marker leftovers raise rather than yielding a
            mismatched prompt.
        """
        # Splicing in id space avoids a decode -> text -> re-encode round trip, which
        # risks BPE drift around the media markers and would silently desync the
        # engine context from the training ids.
        ids = [int(token_id) for token_id in input_ids]
        marker = self.ids
        image_id_open = self._image_id_ids[0]

        out: list[int] = []
        i = 0
        while i < len(ids):
            token = ids[i]
            if token in self._orphan_markers:
                raise ValueError(
                    f"Unbalanced MiniCPM media ids: token {token} at position {i} is only valid "
                    "inside an expanded media span; refusing to build a mismatched engine prompt."
                )
            if token == image_id_open:
                i = self._consume_image_id_prefix(ids, i, out)
            elif token == marker["im_start"]:
                i = self._consume_image_unit(ids, i)
                out.extend(self._engine_image_slot_ids)
            elif token == marker["audio_start"]:
                i = self._consume_audio_chunks(ids, i)
                out.extend(self._engine_audio_slot_ids)
            else:
                out.append(token)
                i += 1
        return out

    def derive_media_bounds(self, input_ids_1d: torch.Tensor) -> tuple[list[list[int]], list[list[int]]]:
        """Scan expanded ids for media spans, mirroring the remote scan.

        Args:
            input_ids_1d: One flattened prompt's ids.

        Returns:
            ``(image_bounds, audio_bounds)`` with ``[start, end)`` pairs; audio
            starts one token after ``<|audio_start|>`` so the bound covers only the
            ``<unk>`` run the embeddings replace.
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

    def _resolve_image_id_ids(self) -> tuple[int | None, int | None]:
        """Ids of ``<image_id>`` / ``</image_id>``, or ``(None, None)`` when absent."""
        tokenizer = self.tokenizer
        id_open = tokenizer.convert_tokens_to_ids("<image_id>")
        id_close = tokenizer.convert_tokens_to_ids("</image_id>")
        if isinstance(id_open, int) and id_open >= 0 and isinstance(id_close, int) and id_close >= 0:
            return int(id_open), int(id_close)
        return None, None

    def _compile_image_span_re(self) -> re.Pattern:
        """Regex for an expanded image span: optional id prefix, block, optional slice grid."""
        s = self.strings
        image_block = re.escape(s["im_start"]) + f"(?:{re.escape(s['unk'])})+" + re.escape(s["im_end"])
        grid_cell = re.escape(s["slice_start"]) + f"(?:{re.escape(s['unk'])})+" + re.escape(s["slice_end"])
        # The img-id prefix (<image_id>N</image_id>) may or may not precede the
        # block depending on use_image_id; tolerate both, plus an optional
        # slice grid (max_slice_nums>1 checkpoints).
        id_open, id_close = self._image_id_ids
        if id_open is not None:
            open_str = re.escape(self.tokenizer.decode([id_open]))
            close_str = re.escape(self.tokenizer.decode([id_close]))
            prefix = f"(?:{open_str}\\d+{close_str})?"
        else:
            prefix = ""
        grid = f"(?:\\s*{grid_cell})*"
        return re.compile(f"{prefix}{image_block}{grid}")

    def _compile_audio_span_re(self) -> re.Pattern:
        """Regex for one or more consecutive expanded audio spans."""
        s = self.strings
        chunk = re.escape(s["audio_start"]) + f"(?:{re.escape(s['unk'])})+" + re.escape(s["audio_end"])
        return re.compile(f"(?:{chunk})+")

    def _consume_image_id_prefix(self, ids: list[int], start: int, out: list[int]) -> int:
        """Drop an ``<image_id>N</image_id>`` prefix before an image block; keep it when stray."""
        after = self._scan_to(ids, start + 1, self._image_id_ids[1], "<image_id>")
        if not (after < len(ids) and ids[after] == self.ids["im_start"]):
            out.extend(ids[start:after])  # a stray id pair with no block: verbatim
        return after

    def _consume_image_unit(self, ids: list[int], start: int) -> int:
        """Index past one ``<image>`` block and any ``<slice>`` grid cells belonging to it."""
        marker = self.ids
        i = self._scan_to(ids, start + 1, marker["im_end"], "<image>")
        while True:
            probe = self._skip_whitespace_before_slice(ids, i)
            if probe >= len(ids) or ids[probe] != marker["slice_start"]:
                return i
            i = self._scan_to(ids, probe + 1, marker["slice_end"], "<slice>")

    def _skip_whitespace_before_slice(self, ids: list[int], start: int) -> int:
        """First index from ``start`` that is not whitespace introducing a slice cell."""
        probe = start
        while (
            probe < len(ids)
            and self._is_whitespace_token(ids[probe])
            and probe + 1 < len(ids)
            and ids[probe + 1] == self.ids["slice_start"]
        ):
            probe += 1
        return probe

    def _consume_audio_chunks(self, ids: list[int], start: int) -> int:
        """Index past consecutive ``<|audio_start|>...<|audio_end|>`` chunks."""
        marker = self.ids
        i = self._scan_to(ids, start + 1, marker["audio_end"], "<audio>")
        while i < len(ids) and ids[i] == marker["audio_start"]:
            i = self._scan_to(ids, i + 1, marker["audio_end"], "<audio>")
        return i

    def _is_whitespace_token(self, token_id: int) -> bool:
        """True when the token decodes to whitespace only."""
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


def flatten_block_content_to_slots(messages: list[dict]) -> list[dict]:
    """Render verl's OpenAI-style content blocks into MiniCPM-o slot strings.

    Media blocks become the processor's slot markers in content order; messages
    with string content pass through unchanged and the caller's dicts are never
    mutated.

    Args:
        messages: Chat messages whose content may be a list of block dicts.

    Returns:
        New messages with list content flattened to a slot-marked string.
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


def _parity_call(self, text=None, images=None, audio=None, audios=None, **kwargs):
    """Call the remote processor with verl's media conventions adapted."""
    # verl passes ``audio=``; the remote processor's kwarg is ``audios=``.
    if audio is not None and audios is None:
        audios = audio
    # The remote processor expresses "no media" as None (it branches on
    # `is not None`, and audio extraction indexes audios[0]), while verl forwards
    # empty containers.
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
    # Call the remote processor's own __call__, then repair its BatchFeature.
    base = type(self).__bases__[0]
    return _upgrade_batch_feature(base.__call__(self, text=text, images=images, audios=audios, **kwargs))


def _parity_normalize_text(tokens: MiniCPMMediaTokens, text: str, images: Any, audios: Any) -> str:
    """Collapse the text's expanded spans and append the slots its media count still needs."""
    # verl always renders one prompt per processor call (text=[str]).
    normalized = tokens.collapse_to_slots(text, engine_slots=False)
    have_images, have_audios = tokens.count_slots(normalized)
    for name, given, have, slot in (
        ("image", images, have_images, MINICPM_IMAGE_SLOT),
        ("audio", audios, have_audios, MINICPM_AUDIO_SLOT),
    ):
        if given is None:
            continue
        n_given = 1 if isinstance(given, str | bytes) else len(given)
        if have > n_given:
            raise ValueError(f"Text carries {have} {name} slots but only {n_given} {name}s were given.")
        # Features are position-independent: the appended slots only satisfy the
        # remote processor's one-slot-per-media assert.
        normalized += slot * (n_given - have)
    return normalized


def _parity_apply_chat_template(self, messages, **kwargs):
    """Flatten block content, then render through the processor or tokenizer template."""
    # The MiniCPM-o template renders string content only, so flatten verl's blocks
    # (image/audio -> slot markers) at this boundary — the single choke point for
    # both rollout renders and the dataset's doc2len.
    base = type(self).__bases__[0]
    flattened = flatten_block_content_to_slots(messages)
    # Deterministic dispatch: the processor's own rendering or its processor-level
    # template first, then the tokenizer's. transformers >= 5's generic
    # ProcessorMixin.apply_chat_template never consults the tokenizer, while MiniCPM-o
    # ships the template on the tokenizer only (local checkpoint dirs have no
    # chat_template.json). Every verl call site renders with tokenize=False, where the
    # two are equivalent — so raise loudly when neither can render.
    if "apply_chat_template" in base.__dict__ or getattr(self, "chat_template", None) is not None:
        return base.apply_chat_template(self, flattened, **kwargs)
    if self.tokenizer is not None and getattr(self.tokenizer, "chat_template", None) is not None:
        return self.tokenizer.apply_chat_template(flattened, **kwargs)
    raise ValueError(
        "No renderable chat template for MiniCPM-o: the processor class does not "
        "implement apply_chat_template, the processor has no chat_template, and the "
        "tokenizer has none either. The checkpoint should ship one in "
        "tokenizer_config.json (or a chat_template.json next to it) — without it "
        "every prompt render fails."
    )


def _parity_dedup_pad_tokens(self, input_ids: list[int]) -> list[int]:
    """Collapse the agent-loop ids' expanded media spans into vLLM's engine slots."""
    # Collapse happens entirely in id space: re-encoding decoded text risks
    # BPE drift that would silently desync the engine prompt from the
    # training ids. Text-only prompts come back byte-identical.
    tokens: MiniCPMMediaTokens = self._parity_tokens
    if not tokens.has_media_tokens(torch.tensor(input_ids)):
        return list(input_ids)
    return tokens.collapse_ids_to_slots(list(input_ids))


def _scalar_tree(value: Any) -> bool:
    """True when every recursive leaf is a scalar.

    Stacking scalars is unambiguous, but stacking same-shape tensor leaves
    collapses a sample's slice list into one tensor, so only these may convert
    wholesale.
    """
    if isinstance(value, list | tuple):
        return bool(value) and all(_scalar_tree(item) for item in value)
    return isinstance(value, numbers.Number)


def _safe_convert_to_tensors(self, tensor_type=None):
    """``convert_to_tensors`` that keeps ragged media structure and tensor leaves intact."""
    # Survives three converter defects: the remote override nulls already-tensor
    # leaves, the stock converter raises on ragged structures (MiniCPM's
    # pixel_values is a per-sample list of patch tensors), and transformers'
    # ``as_tensor`` wrapper stacks same-shape tensor lists (see ``_scalar_tree``).
    if tensor_type is None:
        return self

    is_tensor, as_tensor = self._get_is_as_tensor_fns(tensor_type)

    def convert(value):
        if is_tensor(value):
            return value
        if isinstance(value, list | tuple):
            if _scalar_tree(value):
                try:
                    return as_tensor(value)
                except Exception:
                    pass  # ragged scalars: descend, do not raise
            if isinstance(value, tuple):
                return tuple(convert(item) for item in value)
            return [convert(item) for item in value]
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        try:
            return as_tensor(value)  # bare leaves (np arrays, scalars)
        except Exception:
            return value

    for key, value in self.items():
        self[key] = convert(value)
    return self


def _upgrade_batch_feature(feature: Any) -> Any:
    """Re-point the remote BatchFeature's tensor-nulling ``convert_to_tensors``.

    The remote override maps already-tensor leaves to None, which would wipe every
    media feature while the rollout stays multimodal. Non-BatchFeature results and
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
