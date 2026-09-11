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
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register() -> None:
    """Alias the new-style ``embed_multimodal`` onto MiniCPM-o's engine class.

    Why (vllm-omni's MiniCPM-o 4.5 class vs vLLM >= 0.28):
        ``MiniCPMO45OmniLLMForConditionalGeneration`` implements the old-style
        ``get_multimodal_embeddings``; vLLM 0.28's encoder-cache profiling and
        runtime encoder paths call ``embed_multimodal`` directly, and the
        ``SupportsMultiModal`` default is an abstract stub returning ``None`` —
        every MiniCPM-o engine process died there. The old-style method already
        satisfies the new API's contract, so the alias points the new name at
        it. An in-process alias does not survive engine-core spawns, hence this
        plugin.

    The plugin runs in every vLLM process of the environment, so a missing or
    reshaped minicpmo_4_5 module is logged and skipped — unrelated models keep
    working, and a MiniCPM-o run against an incompatible vllm-omni fails with
    its own error rather than breaking every engine. It also normalizes the
    class's forward return to hidden states (see _normalize_forward_return).
    Upstream both fixes to vLLM-Omni and delete this plugin.
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

    Why (vllm-omni forward tuple vs the AR runner's consumption):
        ``MiniCPMO45OmniLLMForConditionalGeneration.forward`` returns
        ``(text_inputs_embeds, hidden_states)`` — the raw input EMBEDDINGS
        first (minicpmo_4_5_omni_llm.py: "return text_inputs_embeds,
        hidden_states.unsqueeze(0) ..."). The AR runner's
        ``extract_multimodal_outputs`` takes tuple element [0] as the hidden
        states, so logits were computed by applying the LM head to raw input
        embeddings; an input embedding strongly predicts its own token, and
        every rollout degenerated into confident self-repetition loops
        truncated at the response cap — the zero-reward signature of the
        whole bring-up. The wrapper returns only the hidden-states element,
        squeezed to the ``[num_tokens, hidden]`` layout the runner expects;
        a plain tensor return (what a fixed upstream forward produces)
        passes through untouched. Only this LLM class is wrapped — the
        3-stage wrapper class keeps its tuple for the Talker bridge.
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
