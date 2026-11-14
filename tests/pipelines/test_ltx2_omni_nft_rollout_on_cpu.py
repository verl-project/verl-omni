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
"""LTX OmniNFT emits named media while retaining its clean actor tensors."""

from types import SimpleNamespace

import pytest
import torch
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.ltx2_omni_nft.vllm_omni_rollout_adapter import LTX2Pipeline, LTX23OmniNFTPipeline


@pytest.mark.parametrize("as_list", [False, True])
def test_rollout_keeps_replay_and_named_media(monkeypatch, as_list):
    pipeline = object.__new__(LTX23OmniNFTPipeline)
    # object.__new__ skips nn.Module.__init__; monkeypatch teardown's delattr needs that state.
    torch.nn.Module.__init__(pipeline)
    pipeline.distributed_video_decode = False
    pipeline.vocoder = SimpleNamespace(config=SimpleNamespace(output_sampling_rate=24000))
    request = SimpleNamespace(request_id="smoke", sampling_params=SimpleNamespace(extra_args={}))
    monkeypatch.setattr(pipeline, "_prepare_request", lambda value: value)
    video = torch.rand(1, 9, 3, 32, 32)
    audio = torch.rand(1, 2, 9000)
    video_tokens = torch.randn(1, 8, 128)
    audio_tokens = torch.randn(1, 6, 128)
    prompt = SimpleNamespace(
        **{
            key: torch.zeros(1, 2, 4)
            for key in (
                "positive_connector_prompt_embeds",
                "positive_connector_audio_prompt_embeds",
                "positive_connector_attention_mask",
                "negative_connector_prompt_embeds",
                "negative_connector_audio_prompt_embeds",
                "negative_connector_attention_mask",
            )
        }
    )

    def native_forward(self, req):
        self._omni_nft_clean_state = SimpleNamespace(video=video_tokens, audio=audio_tokens)
        self._omni_nft_forward_context = SimpleNamespace(
            prompt_context=prompt,
            original_audio_num_frames=5,
            timesteps=torch.tensor([1000.0, 500.0]),
            request_inputs=SimpleNamespace(frame_rate=24.0),
        )
        self._omni_nft_native_latents = (torch.zeros(1, 128, 2, 1, 1), torch.zeros(1, 8, 5, 16))
        result = DiffusionOutput(output=(video, audio))
        return [result] if as_list else result

    monkeypatch.setattr(LTX2Pipeline, "forward", native_forward)
    result = pipeline.forward(DiffusionRequestBatch(requests=[request]))
    payload = result.output["payload"]
    assert set(payload["video"][0]) == {"video_preview", "audio", "video_latent", "audio_latent"}
    assert payload["video"][0]["video_preview"].dtype == torch.uint8
    torch.testing.assert_close(payload["video"][0]["audio"], audio[0])
    torch.testing.assert_close(result.output["metadata"]["rl"]["video_latents_clean"], video_tokens)
    torch.testing.assert_close(result.output["metadata"]["rl"]["audio_latents_clean"], audio_tokens[:, :5])
    assert pipeline._omni_nft_clean_state is None
    assert pipeline._omni_nft_forward_context is None
    assert pipeline._omni_nft_native_latents is None


def test_warmup_does_not_require_decoded_artifacts(monkeypatch):
    pipeline = object.__new__(LTX23OmniNFTPipeline)
    torch.nn.Module.__init__(pipeline)
    request = SimpleNamespace(request_id="dummy_req_id")
    monkeypatch.setattr(pipeline, "_prepare_request", lambda value: value)
    monkeypatch.setattr(LTX2Pipeline, "forward", lambda self, req: DiffusionOutput(output=(torch.empty(0), None)))
    result = pipeline.forward(DiffusionRequestBatch(requests=[request]))
    assert result.output is None
    assert pipeline._omni_nft_clean_state is None
    assert pipeline._omni_nft_native_latents is None
