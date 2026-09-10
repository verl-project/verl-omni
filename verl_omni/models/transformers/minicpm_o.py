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
"""Runtime shims for MiniCPM-o Hugging Face remote-code models.

Keep MiniCPM-o remote-code patches in this file. Call sites should import
helpers from here rather than reimplementing them.

Transformers version (remote code ~4.10 vs transformers >= 5)
-------------------------------------------------------------
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

3. ``get_vision_embedding``
   Remote MiniCPM-o still forwards a dummy image when ``pixel_values`` is empty so
   unused encoder parameters stay in the autograd graph. LoRA ``exclude_modules``
   already skips adapters on ``vpm``, so that dummy is wasted compute and would
   put vision tensors on the backward path. Empty ``pixel_values`` skip the
   encoder; real images still go through the original method under
   ``torch.no_grad()``.

4. ``get_vllm_embedding``
   Remote ``get_vllm_embedding`` does ``vllm_embedding[i].scatter_(...)``. PEFT
   ``enable_input_require_grads`` makes the embed_tokens output a leaf, so that
   view in-place op fails. Clone each row, use out-of-place ``scatter``, then
   ``stack``.
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Any

logger = logging.getLogger(__name__)

_POST_INIT_PATCHED_ATTR = "_verl_omni_post_init_patched"
_WHISPER_ATTN_PATCHED_ATTR = "_verl_omni_whisper_attn_return3"
_VISION_EMB_PATCH_ATTR = "_verl_omni_get_vision_embedding_patched"
_VLLM_EMB_PATCH_ATTR = "_verl_omni_get_vllm_embedding_patched"
_SIGLIP_FA2_PATCHED_ATTR = "_verl_omni_siglip_fa2_aliased"
_AUDIO_DUMMY_PATCH_ATTR = "_verl_omni_get_audio_embedding_patched"
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


def patch_remote_siglip_flash_attn_support(model_path: str, *, trust_remote_code: bool) -> None:
    """Alias the remote SigLIP's transformers-4.x FA2 flag to the >= 5 name.

    Why (remote code written for 4.x vs transformers >= 5 dispatch):
        The remote ``modeling_navit_siglip.py`` declares
        ``_supports_flash_attn_2 = True`` (the 4.x class flag) on
        ``SiglipVisionTransformer`` and implements ``SiglipFlashAttention2``
        natively — the per-layer choice is driven by
        ``config._attn_implementation``, which transformers 5 still honors.
        Transformers 5 renamed the class flag to ``_supports_flash_attn`` and
        its init-time dispatch hard-rejects ``flash_attention_2`` when only
        the old flag is present, crashing ``MiniCPMO``'s vision-tower
        construction before any weights load. The alias copies the remote's
        own declaration only — it never enables support the remote did not
        claim.

    Call this before ``AutoModel.from_pretrained``. No-op on transformers < 5.
    """
    if not _needs_transformers5_compat() or not trust_remote_code:
        return

    from transformers import PreTrainedModel
    from transformers.models.auto.auto_factory import get_class_from_dynamic_module

    vision_cls = get_class_from_dynamic_module(
        "modeling_navit_siglip.SiglipVisionTransformer",
        model_path,
        trust_remote_code=trust_remote_code,
    )
    module = sys.modules[vision_cls.__module__]
    for value in vars(module).values():
        if (
            isinstance(value, type)
            and issubclass(value, PreTrainedModel)
            and getattr(value, "_supports_flash_attn_2", False)
            and "_supports_flash_attn" not in value.__dict__
            and not getattr(value, _SIGLIP_FA2_PATCHED_ATTR, False)
        ):
            value._supports_flash_attn = True
            setattr(value, _SIGLIP_FA2_PATCHED_ATTR, True)
            logger.debug(
                "Aliased %s._supports_flash_attn_2 to the transformers>=5 _supports_flash_attn name.",
                value.__name__,
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


def _has_pixel_slices(pixel_values) -> bool:
    if pixel_values is None or pixel_values == []:
        return False
    if isinstance(pixel_values, (list | tuple)):
        return any(_has_pixel_slices(sample) for sample in pixel_values)
    return True


def patch_minicpm_get_vision_embedding(module) -> None:
    """Skip dummy vpm forwards and stop vision embeddings from entering autograd.

    Why (remote dummy image vs LoRA-excluded ``vpm``):
        Remote ``get_vision_embedding`` still forwards a dummy image during
        training so unused encoder parameters stay in the autograd graph.
        LoRA ``exclude_modules`` leaves ``vpm`` without adapters, so that dummy
        is wasted compute and would put vision tensors on the backward path.
        Empty ``pixel_values`` skip the encoder; real images still go through
        the original method under ``torch.no_grad()``.
    """
    original = getattr(module, "get_vision_embedding", None)
    if original is None or getattr(module, _VISION_EMB_PATCH_ATTR, False):
        return

    def get_vision_embedding(self, data, _original=original):
        del self
        if isinstance(data, dict) and "vision_hidden_states" in data:
            return _original(data)
        pixel_values = data.get("pixel_values") if isinstance(data, dict) else None
        if not _has_pixel_slices(pixel_values):
            return [[] for _ in (pixel_values or [])]
        import torch

        with torch.no_grad():
            return _original(data)

    module.get_vision_embedding = types.MethodType(get_vision_embedding, module)
    setattr(module, _VISION_EMB_PATCH_ATTR, True)


def _embed_tokens_module(module):
    llm = getattr(module, "llm", None)
    if llm is None:
        return None
    model = getattr(llm, "model", llm)
    return getattr(model, "embed_tokens", None)


def patch_minicpm_get_vllm_embedding(module) -> None:
    """Clone text embeddings before vision scatter so LoRA backward is legal.

    Why (remote MiniCPM-o ``scatter_`` vs PEFT input grads):
        Remote ``get_vllm_embedding`` does ``vllm_embedding[i].scatter_(...)``.
        PEFT ``enable_input_require_grads`` makes the embed_tokens output a leaf,
        so that view in-place op fails. Clone each row, use out-of-place
        ``scatter``, then ``stack``.
    """
    if getattr(module, _VLLM_EMB_PATCH_ATTR, False):
        return
    if _embed_tokens_module(module) is None or not hasattr(module, "get_vision_embedding"):
        return

    def get_vllm_embedding(self, data):
        import torch

        vision_hidden_states = self.get_vision_embedding(data)
        vllm_embedding = _embed_tokens_module(self)(data["input_ids"])
        llm_config = getattr(self.llm, "config", None)
        if llm_config is not None and hasattr(llm_config, "scale_emb"):
            vllm_embedding = vllm_embedding * llm_config.scale_emb

        vision_hidden_states = [
            i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i for i in vision_hidden_states
        ]

        rows = []
        batch_size = len(data["input_ids"])
        image_bound = data.get("image_bound") or [[] for _ in range(batch_size)]
        for i in range(batch_size):
            row = vllm_embedding[i].clone()
            cur_vs_hs = vision_hidden_states[i] if i < len(vision_hidden_states) else []
            if len(cur_vs_hs) > 0:
                cur_image_bound = image_bound[i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [torch.arange(int(bound[0]), int(bound[1]), dtype=torch.long) for bound in cur_image_bound]
                    ).to(vllm_embedding.device)
                    src = cur_vs_hs.view(-1, cur_vs_hs.shape[-1]).to(device=row.device, dtype=row.dtype)
                    index = image_indices.view(-1, 1).repeat(1, row.shape[-1])
                    row = row.scatter(0, index, src)
                elif self.training:
                    row = row + cur_vs_hs[0].mean() * 0
            rows.append(row)
        return torch.stack(rows, dim=0), vision_hidden_states

    module.get_vllm_embedding = types.MethodType(get_vllm_embedding, module)
    setattr(module, _VLLM_EMB_PATCH_ATTR, True)


def patch_minicpm_get_audio_embedding(module) -> None:
    """Skip the dummy Whisper forward for audio-free batches in training.

    Why (remote dummy wav vs frozen, FSDP2-ignored ``apm``):
        Remote ``get_audio_embedding`` forwards a dummy ``(1, 80, 100)`` wav
        when training with empty ``audio_features``, solely to keep unused
        encoder parameters in the autograd graph. Here ``apm`` is frozen
        (LoRA-excluded and adapter-ignored under FSDP2), so that is pure
        wasted compute — and the dummy path reads
        ``self.embed_positions.weight`` directly, assuming the subtree is not
        FSDP-managed. Returning ``[]`` instead would break the remote
        ``get_omni_embedding`` contract (``audio_embeddings[0].mean() * 0``
        for the training no-audio branch), so the patched path returns one
        zero tensor shaped like a single pooled audio token: no Whisper
        forward, contract preserved. Non-empty audio batches delegate to the
        original method unchanged.
    """
    apm = getattr(module, "apm", None)
    llm = getattr(module, "llm", None)
    original = getattr(module, "get_audio_embedding", None)
    if apm is None or llm is None or original is None or getattr(module, _AUDIO_DUMMY_PATCH_ATTR, False):
        return

    def get_audio_embedding(self, data, chunk_length=-1, dummy=True, **kwargs):
        # Mirrors the remote get_omni_embedding emptiness test (post-split
        # normalization empties are exactly []); no `or` — features may be a
        # tensor and tensor truthiness is ambiguous.
        features = data.get("audio_features", [])
        if features is None:
            features = []
        if self.training and len(features) == 0:
            import torch

            weight = self.apm.conv1.weight
            hidden = getattr(getattr(self.llm, "config", None), "hidden_size", 1)
            return [torch.zeros(1, hidden, 1, device=weight.device, dtype=weight.dtype)]
        return original(data, chunk_length=chunk_length, dummy=dummy, **kwargs)

    module.get_audio_embedding = types.MethodType(get_audio_embedding, module)
    setattr(module, _AUDIO_DUMMY_PATCH_ATTR, True)
