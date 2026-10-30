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

"""Carry Lance reference media separately from its text-only prompt tokens."""

from typing import Any

from verl.experimental.agent_loop.agent_loop import register

from verl_omni.agent_loop.single_turn_agent_loop import DiffusionSingleTurnAgentLoop
from verl_omni.agent_loop.utils import messages_to_text as _messages_to_text


@register("lance_diffusion_single_turn_agent")
class LanceDiffusionSingleTurnAgentLoop(DiffusionSingleTurnAgentLoop):
    """Let Lance own reference decoding, resizing and frame sampling."""

    async def process_multi_modal_info(self, messages: list[dict]) -> dict[str, Any]:
        """Collect reference media from the message content itself.

        Nothing else can: Lance ships no processor, so the dataset layer's
        content items are the only place the reference lives, and it has already
        turned the ``<image>`` marker into an image item by the time this runs.
        """
        media: dict[str, list[Any]] = {"images": [], "videos": []}
        keys = {"image": "images", "video": "videos"}
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                kind = item.get("type")
                if kind == "audio":
                    raise ValueError("Lance conditioning does not support reference audio.")
                if kind in keys:
                    media[keys[kind]].append(item[kind])
        return {key: values for key, values in media.items() if values}

    async def ct_build_initial_tokens(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
        audios: list[Any] | None = None,
    ) -> list[int]:
        """Tokenize the text only; the pipeline prefills the reference itself.

        Handing the reference to the text encoder's processor would expand it
        into prompt tokens - for Lance's checkpoint into a request-sized
        sequence the rollout then rejects - while ``LancePipeline`` reads the
        reference from ``multi_modal_data`` and builds its own vision and VAE
        prefill around the instruction.
        """
        if audios or len(images or []) + len(videos or []) > 1:
            raise ValueError("Lance conditioning accepts one reference image or one reference video, without audio.")
        del tools
        text_only = [{"role": message.get("role"), "content": _messages_to_text([message])} for message in messages]
        return await super().ct_build_initial_tokens(text_only)

    def _assert_mm_supported(self, has_multi_modal: bool) -> None:
        """Allow reference media carried beside the prompt."""
        del has_multi_modal
