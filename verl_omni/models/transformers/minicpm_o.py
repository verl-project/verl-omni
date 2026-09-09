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
"""Transformers-version shims for MiniCPM-o remote-code models.

MiniCPM-o checkpoints ship modeling code written against transformers 4.x
(MiniCPM-o-4_5 targets ~4.10). verl-omni runs transformers >= 5. The two API
breaks below are the only version gaps; keep every MiniCPM-o transformers-version
patch in this file.

1. ``post_init()`` / ``all_tied_weights_keys``
   Transformers 5 requires every ``PreTrainedModel.__init__`` to call ``post_init()``,
   which sets ``all_tied_weights_keys`` before weight loading. MiniCPM-o remote
   ``MiniCPMO`` only declares ``_tied_weights_keys`` and skips ``post_init()``, so
   ``from_pretrained`` fails. ``patch_remote_auto_model_init`` wraps the dynamic
   class so ``__init__`` ends with ``post_init()`` when the attribute is missing.
   Apply this *before* ``from_pretrained``. No-op on transformers < 5.

2. WhisperAttention return value and cache kwarg
   MiniCPM-o's remote ``MiniCPMWhisperEncoderLayer`` still does::

       hidden_states, attn_weights, past_key_values = self.self_attn(..., past_key_value=...)

   Current ``WhisperAttention.forward`` returns ``(hidden_states, attn_weights)``
   and takes ``past_key_values`` (plural). Training still runs the audio encoder
   (including dummy wavs), so the unpack crashes. ``patch_remote_whisper_self_attn``
   wraps each ``apm`` layer's ``self_attn`` after load: pad a 2-tuple to a 3-tuple
   and rename the cache kwarg. Apply this *after* ``from_pretrained``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_POST_INIT_PATCHED_ATTR = "_verl_omni_post_init_patched"
_WHISPER_ATTN_PATCHED_ATTR = "_verl_omni_whisper_attn_return3"
# Back-compat alias for tests that reset the post_init wrap.
_PATCHED_ATTR = _POST_INIT_PATCHED_ATTR


def _needs_transformers5_compat() -> bool:
    try:
        import transformers

        return int(transformers.__version__.split(".", 1)[0]) >= 5
    except Exception:
        return False


def patch_remote_auto_model_init(
    model_path: str,
    *,
    trust_remote_code: bool,
    config: Any = None,
    auto_class_name: str = "AutoModel",
) -> None:
    """Wrap a remote auto-model class so ``post_init()`` runs when missing.

    Why (transformers 4.x remote code vs transformers >= 5):
        Transformers 5 expects every ``PreTrainedModel`` to call ``post_init()`` at
        the end of ``__init__``, which sets ``all_tied_weights_keys``. Remote MiniCPM-o
        (written for ~4.10) only declares ``_tied_weights_keys`` and omits
        ``post_init()``, so ``from_pretrained`` fails during weight loading.

    Call this before ``AutoModel.from_pretrained``.
    """
    if not _needs_transformers5_compat() or not trust_remote_code:
        return

    from transformers import AutoConfig
    from transformers.models.auto.auto_factory import get_class_from_dynamic_module

    resolved_config = config
    if resolved_config is None:
        resolved_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    auto_map = getattr(resolved_config, "auto_map", None)
    if not auto_map or auto_class_name not in auto_map:
        return

    model_cls = get_class_from_dynamic_module(
        auto_map[auto_class_name],
        model_path,
        trust_remote_code=trust_remote_code,
    )
    wrap_model_init_with_post_init(model_cls)


def wrap_model_init_with_post_init(model_cls: type) -> None:
    """Ensure ``model_cls.__init__`` ends with ``post_init()`` when needed."""
    if getattr(model_cls, _POST_INIT_PATCHED_ATTR, False):
        return

    original_init = model_cls.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, "all_tied_weights_keys") and hasattr(self, "post_init"):
            self.post_init()

    model_cls.__init__ = patched_init
    setattr(model_cls, _POST_INIT_PATCHED_ATTR, True)
    logger.debug(
        "Patched %s.__init__ to call post_init() for transformers >= 5 compatibility.",
        model_cls.__name__,
    )


def _pad_whisper_self_attn_output(output, past_key_values=None):
    """Normalize WhisperAttention output to the 3-tuple MiniCPM remote code unpacks."""
    if not isinstance(output, tuple):
        return output, None, past_key_values
    if len(output) == 2:
        hidden_states, attn_weights = output
        return hidden_states, attn_weights, past_key_values
    return output


def wrap_whisper_self_attn_forward(attn_module) -> None:
    """Make ``self_attn`` always return ``(hidden_states, attn_weights, past_key_values)``.

    Why (transformers 4.x remote code vs current WhisperAttention):
        MiniCPM-o's remote ``MiniCPMWhisperEncoderLayer`` still unpacks three values
        and passes ``past_key_value`` (singular). Newer WhisperAttention returns
        ``(hidden_states, attn_weights)`` and takes ``past_key_values``. Training
        still runs the audio encoder, so the unpack fails even without real audio.
    """
    if attn_module is None or getattr(attn_module, _WHISPER_ATTN_PATCHED_ATTR, False):
        return

    original_forward = attn_module.forward

    def _forward(*args, _original=original_forward, **kwargs):
        past_key_values = kwargs.get("past_key_values", kwargs.get("past_key_value"))
        if "past_key_value" in kwargs and "past_key_values" not in kwargs:
            kwargs["past_key_values"] = kwargs.pop("past_key_value")
        return _pad_whisper_self_attn_output(_original(*args, **kwargs), past_key_values)

    attn_module.forward = _forward
    setattr(attn_module, _WHISPER_ATTN_PATCHED_ATTR, True)


def patch_remote_whisper_self_attn(module) -> None:
    """Patch MiniCPM-o ``apm`` Whisper self-attn after remote-code ``from_pretrained``.

    Walks ``module.apm.layers[*].self_attn``. No-op when ``apm`` is missing.
    Call this after the model is loaded.
    """
    apm = getattr(module, "apm", None)
    layers = getattr(apm, "layers", None) if apm is not None else None
    if not layers:
        return
    for layer in layers:
        wrap_whisper_self_attn_forward(getattr(layer, "self_attn", None))
