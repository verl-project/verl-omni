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
"""vLLM general plugins, loaded in every engine process.

Registered under the ``vllm.general_plugins`` entry point group (see
pyproject.toml). vLLM calls each plugin at startup in every process,
including freshly spawned engine cores — the only hook that reliably
crosses process boundaries.

Both fixes this plugin carries have landed upstream: the
``embed_multimodal`` alias in vllm-omni#7384 and the bare-tensor forward
return in vllm-omni#7517 — but the pinned vllm-omni commit predates both.
"""

# TODO (mike): drop this file and its ``vllm.general_plugins`` entry point in
# pyproject.toml once .github/vllm_omni_pin.txt includes vllm-omni#7517 —
# landing on main is NOT enough; a pin bumped past only #7384 would still
# need the forward-return half.

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register() -> None:
    """Alias the new-style ``embed_multimodal`` onto MiniCPM-o's engine class.

    The class implements the old-style ``get_multimodal_embeddings``; vLLM
    0.28's encoder paths call ``embed_multimodal`` directly and the
    ``SupportsMultiModal`` default is a stub returning ``None`` — every
    MiniCPM-o engine process died there. The old-style method already
    satisfies the new contract, so the alias points the new name at it; an
    in-process alias does not survive engine-core spawns, hence this plugin.
    A missing or reshaped minicpmo_4_5 module is logged and skipped so
    unrelated models keep working. Landed upstream in vllm-omni#7384.
    """
    try:
        from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
            MiniCPMO45OmniLLMForConditionalGeneration,
        )
    except Exception as exc:  # vllm-omni builds without minicpmo_4_5
        logger.debug("MiniCPM-o 4.5 engine class unavailable (%s); skipping embed_multimodal alias.", exc)
        return

    model_cls = MiniCPMO45OmniLLMForConditionalGeneration
    if "embed_multimodal" not in model_cls.__dict__:
        # native or already aliased; plugins may load multiple times
        legacy = getattr(model_cls, "get_multimodal_embeddings", None)
        if legacy is not None:
            model_cls.embed_multimodal = legacy
            logger.debug("Aliased %s.embed_multimodal to get_multimodal_embeddings.", model_cls.__name__)
        else:
            logger.warning(
                "%s has neither embed_multimodal nor get_multimodal_embeddings; skipping the "
                "alias — a MiniCPM-o rollout would fail at multimodal profiling.",
                model_cls.__name__,
            )
    _normalize_forward_return(model_cls)


_FORWARD_NORM_ATTR = "_verl_omni_forward_normalized"


def _normalize_forward_return(model_cls) -> None:
    """Return hidden states from the LLM engine class's forward.

    The class's forward returns ``(text_inputs_embeds, hidden_states)``
    — embeddings first — while the AR runner's ``extract_multimodal_outputs``
    consumes tuple element [0] as the hidden states: logits were computed
    from raw input embeddings and every rollout degenerated into
    self-repetition. The wrapper returns only the hidden-states element
    squeezed to ``[num_tokens, hidden]``; a plain tensor return passes
    through untouched. Only this LLM class is wrapped — the 3-stage
    wrapper keeps its tuple for the Talker bridge.

    Landed upstream in vllm-omni#7517 (fixes #7497).
    """
    if getattr(model_cls, _FORWARD_NORM_ATTR, False):
        return

    original_forward = model_cls.forward

    def forward(self, *args, **kwargs):
        output = original_forward(self, *args, **kwargs)
        if isinstance(output, tuple):
            hidden_states = output[1]
            if hidden_states.ndim == 3:
                hidden_states = hidden_states.squeeze(0)
            return hidden_states
        return output

    model_cls.forward = forward
    setattr(model_cls, _FORWARD_NORM_ATTR, True)
    logger.debug("Normalized %s.forward to return hidden states.", model_cls.__name__)
