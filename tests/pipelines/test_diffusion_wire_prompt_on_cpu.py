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
"""Exercise canonical wire prompts at real adapter extraction boundaries."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.rollout_request import OmniRolloutRequest, prompt_ids_from_payload


def _prompt():
    return OmniRolloutRequest.from_generate_kwargs(
        prompt_ids=[1, 2],
        negative_prompt_ids=[3],
        prompt_mask=[1, 1],
        image_data=["image"],
        audio_data=["audio"],
        mm_processor_kwargs={"video_fps": 24, "audio_sample_rate": 32000},
    ).to_diffusion_prompt()


def test_wire_has_one_media_location_and_preserves_empty_processor_kwargs():
    prompt = _prompt()
    assert prompt["prompt_ids"] == [1, 2]
    assert prompt["multi_modal_data"] == {"image": ["image"], "audio": ["audio"]}
    assert prompt["mm_processor_kwargs"]["audio_sample_rate"] == 32000
    assert "prompt_token_ids" not in prompt
    assert "extra_args" not in prompt
    empty = OmniRolloutRequest.from_generate_kwargs(prompt_ids=[], image_data=[], mm_processor_kwargs={})
    assert empty.to_diffusion_prompt() == {
        "prompt_ids": [],
        "multi_modal_data": {"image": []},
        "mm_processor_kwargs": {},
    }


@pytest.mark.parametrize("key", ["prompt_ids", "prompt_token_ids"])
def test_token_transport_boundary(key):
    assert prompt_ids_from_payload({key: []}) == []
    assert prompt_ids_from_payload({key: [1, 2]}) == [1, 2]
    assert prompt_ids_from_payload({}) is None


def test_conflicting_token_spellings_fail():
    with pytest.raises(ValueError, match="Conflicting prompt_ids"):
        prompt_ids_from_payload({"prompt_ids": [1], "prompt_token_ids": [2]})


@pytest.mark.parametrize(
    "architecture,algorithm,method",
    [
        ("QwenImagePipeline", "flow_grpo", "_extract_prompt_ids"),
        ("QwenImagePipeline", "dual_grpo", "_extract_prompt_ids"),
        ("QwenImagePipeline", "mix_grpo", "_extract_prompt_ids"),
        ("BooguImagePipeline", "flow_grpo", "_extract_prompt_ids"),
        ("QwenImagePipeline", "dpo", "_extract_step_prompt_ids"),
        ("QwenImagePipeline", "diffusion_nft", "_extract_step_prompt_ids"),
    ],
)
def test_image_adapters_consume_canonical_ids(architecture, algorithm, method):
    cls = VllmOmniPipelineBase.get_class(architecture, algorithm)
    pipeline = object.__new__(cls)
    payload = _prompt()
    result = getattr(pipeline, method)(payload if method == "_extract_step_prompt_ids" else [payload])
    assert result == ([1, 2], [1, 1], [3], None)


def test_sd3_extra_encoders_and_negatives_round_trip():
    from verl_omni.pipelines.sd3_flow_grpo.vllm_omni_rollout_adapter import _extract_extra_prompt_ids

    prompt = OmniRolloutRequest.from_generate_kwargs(
        prompt_ids=[1],
        extra_prompt_ids={"clip": [2], "t5": [3, 4]},
        negative_extra_prompt_ids={"clip": [5], "t5": [6]},
    ).to_diffusion_prompt()
    assert _extract_extra_prompt_ids([prompt]) == {"clip": [[2]], "t5": [[3, 4]]}
    assert _extract_extra_prompt_ids([prompt], "negative_extra_prompt_ids") == {"clip": [[5]], "t5": [[6]]}
    with pytest.raises(ValueError, match="nested under extra_args"):
        _extract_extra_prompt_ids([{"extra_prompt_ids": {"clip": [2], "t5": [3]}}])


@pytest.mark.parametrize("algorithm", ["flow_grpo", "diffusion_nft"])
def test_minimax_canonical_ids_and_references_survive(algorithm):
    from verl_omni.pipelines.minimax_h3_diffusion_nft.common import MINIMAX_H3_TOKEN_ID_NATIVE_KEY

    cls = VllmOmniPipelineBase.get_class("MiniMaxH3Pipeline", algorithm)
    pipeline = object.__new__(cls)
    request = SimpleNamespace(
        prompts=[_prompt()], sampling_params=SimpleNamespace(extra_args={MINIMAX_H3_TOKEN_ID_NATIVE_KEY: True})
    )
    pipeline._ensure_prompt_text(request)
    torch.testing.assert_close(pipeline._h3_prompt_ids, torch.tensor([1, 2]))
    _, media = pipeline._extract_prompt(request.prompts[0])
    assert media == {"image": ["image"], "audio": ["audio"]}


def test_bagel_keeps_top_level_media_without_copy_up():
    cls = VllmOmniPipelineBase.get_class("OmniBagelForConditionalGeneration", "flow_grpo")
    pipeline = object.__new__(cls)
    pipeline._decode_token_prompt = Mock(side_effect=["positive", "negative"])
    request = SimpleNamespace(prompts=[_prompt()], sampling_params=SimpleNamespace(extra_args={}))
    pipeline._ensure_bagel_prompt_text(request)
    assert request.prompts[0]["prompt"] == "positive"
    assert request.prompts[0]["multi_modal_data"]["image"] == ["image"]
    assert "extra_args" not in request.prompts[0]
    assert request.sampling_params.extra_args["negative_prompt"] == "negative"
    assert pipeline._decode_token_prompt.call_args_list[0].args == ([1, 2],)


def test_ltx_canonical_ids_reach_encoder():
    cls = VllmOmniPipelineBase.get_class("LTX2Pipeline", "flow_grpo")
    pipeline = object.__new__(cls)
    pipeline._encode_token_ids = Mock(return_value=(torch.ones(1, 2, 4), torch.ones(1, 2)))
    request = SimpleNamespace(prompt=_prompt(), sampling_params=SimpleNamespace(max_sequence_length=12))
    pipeline._inject_precomputed_prompt_embeds(request)
    assert pipeline._encode_token_ids.call_args_list[0].args == ([1, 2], [1, 1], 12)
    assert pipeline._encode_token_ids.call_args_list[1].args == ([3], None, 12)
    assert request.prompt["prompt_embeds"].shape == (2, 4)
