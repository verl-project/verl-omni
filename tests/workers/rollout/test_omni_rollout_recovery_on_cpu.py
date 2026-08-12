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
"""CPU contract pin for the separate-async rollout recovery client.

``PPOTrainerSeparateAsync.get_llm_client`` returns upstream
``FullyAsyncLLMServerClient``; omni correctness under weight-sync aborts hinges
on multimodal invariants upstream does not test: media and processor kwargs are
re-sent on every resubmission, token/log-prob streams are merged, the token
budget shrinks by what was already generated, and ``min/max_global_steps``
record the weight-version span.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from verl.workers.rollout.llm_server import FullyAsyncLLMServerClient, LLMServerClient


def _client(cls, **attrs):
    client = cls.__new__(cls)
    for name, value in attrs.items():
        setattr(client, name, value)
    return client


async def test_continuation_resends_media_merges_tokens_and_shrinks_budget():
    """Pin upstream ``FullyAsyncLLMServerClient`` behavior the omni path relies on."""
    outputs = [
        SimpleNamespace(
            token_ids=[7, 8],
            log_probs=[-0.1, -0.2],
            routed_experts=None,
            num_preempted=0,
            stop_reason="abort",
            extra_fields={"global_steps": 3},
        ),
        SimpleNamespace(
            token_ids=[9],
            log_probs=[-0.3],
            routed_experts=None,
            num_preempted=0,
            stop_reason="completed",
            extra_fields={"global_steps": 5},
        ),
    ]
    client = _client(FullyAsyncLLMServerClient)
    media = {"image_data": ["img"], "video_data": None, "audio_data": ["aud"]}
    # sampling_params is mutated in place, so capture the budget at call time.
    seen_budgets = []
    output_iter = iter(outputs)

    async def _record(*args, **kwargs):
        seen_budgets.append(kwargs["sampling_params"]["max_tokens"])
        return next(output_iter)

    with patch.object(LLMServerClient, "generate", new=AsyncMock(side_effect=_record)) as mock_gen:
        final = await client.generate(
            "req-1",
            prompt_ids=[1, 2, 3],
            sampling_params={"max_tokens": 8},
            mm_processor_kwargs={"fps": 2},
            **media,
        )

    assert mock_gen.call_count == 2
    first, second = mock_gen.call_args_list
    # Continuation resubmits prompt + generated-so-far tokens with the same media.
    assert first.kwargs["prompt_ids"] == [1, 2, 3]
    assert second.kwargs["prompt_ids"] == [1, 2, 3, 7, 8]
    for call in (first, second):
        assert call.kwargs["image_data"] == ["img"]
        assert call.kwargs["audio_data"] == ["aud"]
        assert call.kwargs["mm_processor_kwargs"] == {"fps": 2}
    # The resumed request's budget shrinks by the tokens already generated.
    assert seen_budgets == [8, 6]

    assert final.token_ids == [7, 8, 9]
    assert final.log_probs == [-0.1, -0.2, -0.3]
    assert final.stop_reason == "completed"
    assert final.extra_fields["min_global_steps"] == 3
    assert final.extra_fields["max_global_steps"] == 5
