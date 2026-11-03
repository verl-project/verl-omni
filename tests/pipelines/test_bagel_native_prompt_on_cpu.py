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
"""Regression coverage for native caption framing and agent-loop token transport."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from verl_omni.agent_loop.single_turn_agent_loop import DiffusionSingleTurnAgentLoop
from verl_omni.pipelines.bagel_flow_grpo.vllm_omni_rollout_adapter import BagelPipelineWithLogProb


class Tokenizer:
    def convert_tokens_to_ids(self, name):
        return {"<|im_start|>": 1, "<|im_end|>": 2}[name]

    def decode(self, ids, skip_special_tokens=False):
        return "".join({1: "<|im_start|>", 2: "<|im_end|>"}[i] if i in (1, 2) else chr(i - 3) for i in ids)


def ids(text):
    return [1, *[ord(c) + 3 for c in text], 2]


def pipeline():
    p = object.__new__(BagelPipelineWithLogProb)
    torch.nn.Module.__init__(p)
    p.tokenizer = Tokenizer()
    return p


@pytest.mark.parametrize(
    "caption",
    [
        "user interface of a dashboard",
        "user-friendly camera",
        "user\npainting a house",
        "a cat",
        " user-friendly camera ",
        "\nuser interface\t",
    ],
)
def test_native_caption_and_request_flag(caption):
    p = pipeline()
    req = SimpleNamespace(
        prompts=[{"prompt_ids": ids(caption)}],
        sampling_params=SimpleNamespace(extra_args={"native_prompt_tokens": True}),
    )
    p._ensure_bagel_prompt_text(req)
    assert req.prompts[0]["prompt"] == caption


def test_chat_format_is_unchanged():
    assert pipeline()._decode_token_prompt(ids("user\na cat")) == "a cat"


@pytest.mark.parametrize("tokens", [None, [], [4, 5, 2], [1, 5, 4]])
def test_malformed_native_framing_fails(tokens):
    with pytest.raises(ValueError, match="framing"):
        pipeline()._decode_token_prompt(tokens, native=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True, None])
async def test_agent_loop_marks_only_native_tokens_without_mutating_sampling(native):
    loop = object.__new__(DiffusionSingleTurnAgentLoop)
    loop.rollout_config = SimpleNamespace(prompt_length=1024)
    loop.extra_tokenizer_map = {}
    loop.process_multi_modal_info = AsyncMock(return_value={})
    loop.ct_build_initial_tokens = AsyncMock(return_value=[7, 8])
    loop._assert_mm_supported = lambda _: None
    loop._get_routing_request_id = lambda _: "test"
    loop._get_mm_processor_kwargs = lambda _: None
    loop.server_manager = SimpleNamespace(
        generate=AsyncMock(
            return_value=SimpleNamespace(
                num_preempted=0,
                diffusion_output=torch.zeros(3, 8, 8, dtype=torch.uint8),
                log_probs=None,
                extra_fields={},
            )
        )
    )
    original = {"seed": 42}
    kwargs = {"raw_prompt": [{"role": "user", "content": "a cat"}]}
    if native:
        kwargs["prompt_token_ids"] = ids("user-friendly camera")
    elif native is None:
        kwargs["prompt_token_ids"] = None
    output = await loop.run(original, **kwargs)
    call = loop.server_manager.generate.call_args.kwargs
    assert original == {"seed": 42}
    if native:
        assert call["sampling_params"]["native_prompt_tokens"] is True
        assert call["prompt_ids"] == kwargs["prompt_token_ids"] == output.prompt_ids
        loop.ct_build_initial_tokens.assert_not_called()
    else:
        assert "native_prompt_tokens" not in call["sampling_params"]
        loop.ct_build_initial_tokens.assert_awaited_once()
