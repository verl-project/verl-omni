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


class _MiniCPMProcessorParityWrapper:
    """Delegate everything to the remote processor except ``__call__``.

    A wrapper class (not an instance-attribute patch) because ``processor(...)``
    dispatches ``__call__`` on the type; instance assignment never intercepts it.
    """

    def __init__(self, processor: Any, tokens: MiniCPMMediaTokens):
        self._processor = processor
        self._tokens = tokens
        self.tokenizer = processor.tokenizer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._processor, name)

    def dedup_pad_tokens(self, input_ids: list[int]) -> list[int]:
        tokenizer = self.tokenizer
        text = tokenizer.decode(input_ids, skip_special_tokens=False)
        if self._tokens.has_media_tokens(torch.tensor(input_ids)):
            collapsed = self._tokens.collapse_to_slots(text, engine_slots=True)
            n_images, n_audios = self._tokens.count_slots(collapsed)
            if n_images == 0 and n_audios == 0:
                raise ValueError(
                    "Prompt ids contain MiniCPM media tokens but no expanded span matched the "
                    "collapse patterns; refusing to send a mismatched prompt to vLLM-Omni."
                )
            text = collapsed
        return tokenizer.encode(text, add_special_tokens=False)

    def _normalize_text(self, text: str, images: Any, audios: Any) -> str:
        # verl always renders one prompt per processor call (text=[str]).
        normalized = self._tokens.collapse_to_slots(text, engine_slots=False)
        if images is not None:
            n_images = len(images) if not isinstance(images, str | bytes) else 1
            have_images, _ = self._tokens.count_slots(normalized)
            if have_images > n_images:
                raise ValueError(f"Text carries {have_images} image slots but only {n_images} images were given.")
            normalized += MINICPM_IMAGE_SLOT * (n_images - have_images)
        if audios is not None:
            n_audios = len(audios) if not isinstance(audios, str | bytes) else 1
            _, have_audios = self._tokens.count_slots(normalized)
            if have_audios > n_audios:
                raise ValueError(f"Text carries {have_audios} audio slots but only {n_audios} audios were given.")
            normalized += MINICPM_AUDIO_SLOT * (n_audios - have_audios)
        return normalized

    def __call__(self, text=None, images=None, audio=None, audios=None, **kwargs):
        if audio is not None and audios is None:
            audios = audio
        if isinstance(text, list):
            text = [self._normalize_text(item, images, audios) if isinstance(item, str) else item for item in text]
        elif isinstance(text, str):
            text = self._normalize_text(text, images, audios)
        return self._processor(text=text, images=images, audios=audios, **kwargs)


def bind_minicpm_processor(processor: Any) -> Any:
    """Bind the parity helpers a MiniCPM-o processor needs for RL rollout.

    - ``dedup_pad_tokens``: the AR strategy's pre-engine hook collapses the
      expanded spans of the agent-loop ids into vLLM's parenthesized slots.
    - ``__call__`` wrapper: adapts verl's ``audio=`` kwarg to the remote
      ``audios=``, collapses any expanded spans found in text, and appends
      canonical slots when media is present but the decoded text lost its
      slots (features are position-independent, so appended slots only
      satisfy the remote one-slot-per-media assert).
    """
    return _MiniCPMProcessorParityWrapper(processor, resolve_media_tokens(processor))
