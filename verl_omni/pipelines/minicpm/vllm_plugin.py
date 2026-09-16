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

Registered as a ``vllm.general_plugins`` entry point (pyproject.toml): vLLM
runs these at startup in every process, spawned engine cores included — the
only hook that reliably crosses process boundaries.
"""

# TODO (mike): both fixes have landed upstream (vllm-omni#7384, #7517); drop this
# file and its pyproject entry point once the pin includes them.

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register() -> None:
    """Alias the new-style ``embed_multimodal`` onto MiniCPM-o's engine class.

    The class implements the old-style ``get_multimodal_embeddings``, while
    vLLM's encoder paths call ``embed_multimodal`` and its default stub returns
    ``None``. An in-process alias does not survive engine-core spawns, hence a
    plugin; a missing minicpmo_4_5 module is skipped so other models still run.
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
    """Return this class's hidden states instead of its embeddings-first tuple.

    The AR runner consumes tuple element [0] as the hidden states, but this
    forward returns ``(text_inputs_embeds, hidden_states)``, so logits came
    from raw input embeddings and every rollout degenerated into
    self-repetition. A plain tensor return passes through untouched; only this
    LLM class is wrapped, since the 3-stage wrapper keeps its tuple for the
    Talker bridge.
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
