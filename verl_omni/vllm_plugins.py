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
    its own error rather than breaking every engine. Upstream the rename to
    vLLM-Omni and delete this plugin.
    """
    try:
        from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
            MiniCPMO45OmniLLMForConditionalGeneration,
        )
    except Exception as exc:  # vllm-omni builds without minicpmo_4_5
        logger.debug("MiniCPM-o 4.5 engine class unavailable (%s); skipping embed_multimodal alias.", exc)
        return

    model_cls = MiniCPMO45OmniLLMForConditionalGeneration
    if "embed_multimodal" in model_cls.__dict__:
        return  # native or already aliased; plugins may load multiple times
    legacy = getattr(model_cls, "get_multimodal_embeddings", None)
    if legacy is None:
        logger.warning(
            "%s has neither embed_multimodal nor get_multimodal_embeddings; skipping the "
            "alias — a MiniCPM-o rollout would fail at multimodal profiling.",
            model_cls.__name__,
        )
        return
    model_cls.embed_multimodal = legacy
    logger.debug("Aliased %s.embed_multimodal to get_multimodal_embeddings.", model_cls.__name__)
