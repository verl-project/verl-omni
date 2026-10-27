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

"""Image-conditioned rollout preserves replay masks, native cache positions and CFG branches."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from verl_omni.pipelines.bagel_alphagrpo import vllm_omni_rollout_adapter as rollout


def test_reflection_cache_and_replay_layout(monkeypatch):
    pipeline = object.__new__(rollout.BagelAlphaGRPOPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.new_token_ids = {"bos_token_id": 1, "eos_token_id": 2}
    pipeline.tokenizer = SimpleNamespace(encode=lambda *a, **k: [3, 4], decode=lambda *a, **k: "<think>x</think>")
    pipeline.bagel = SimpleNamespace(config=SimpleNamespace(llm_config=SimpleNamespace(num_hidden_layers=1)))
    monkeypatch.setattr(rollout, "NaiveCache", lambda n: [])
    monkeypatch.setattr(pipeline, "_decode_token_prompt", lambda ids: "")
    monkeypatch.setattr(pipeline, "_prefill", lambda cache, ids, offset=0: cache + [(list(ids), offset)])
    monkeypatch.setattr(
        pipeline,
        "_prefill_image",
        lambda cache, image, offset: (cache + [("vae", offset), ("vit", offset + 1)], torch.ones(7, 8), [4, 3]),
    )
    sampled_offsets = []

    def think(cache, offset, max_tokens, temperature, generator):
        sampled_offsets.append(offset)
        return [9, 2], [-0.7, -0.8]

    monkeypatch.setattr(pipeline, "_sample_thinking", think)
    branches = {}

    def generate(self, request):
        branches.update(deepcopy(vars(request.sampling_params)))
        return object()

    monkeypatch.setattr(rollout.BagelPipelineWithLogProb, "forward", generate)
    monkeypatch.setattr(rollout, "with_rollout_data", lambda output, rl: rl)
    sampling = SimpleNamespace(
        extra_args={"max_think_tokens": 2},
        max_sequence_length=32,
        seed=1,
        past_key_values=None,
        kv_metadata=None,
        cfg_text_past_key_values=None,
        cfg_text_kv_metadata=None,
        cfg_img_past_key_values=None,
        cfg_img_kv_metadata=None,
    )
    request = SimpleNamespace(
        is_dummy_run=lambda: False,
        sampling_params=sampling,
        prompts=[{"prompt_ids": [1, 5, 2], "multi_modal_data": {"image": [Image.new("RGB", (8, 8))]}}],
    )
    replay = pipeline.forward(request)
    assert sampled_offsets == [9]
    assert replay["condition_image_layout"].tolist() == [[4, 4, 3]]
    assert replay["condition_image_embeds"].shape == (1, 7, 8)
    assert replay["thinking_mask"].nonzero()[:, 1].tolist() == [14, 15]
    torch.testing.assert_close(replay["old_text_log_probs"][0, 14:16], torch.tensor([-0.7, -0.8]))
    assert replay["cfg_text_mask"].sum() == 11
    assert replay["cfg_image_mask"].sum() == 7
    assert branches["kv_metadata"]["ropes"] == [12]
    assert branches["cfg_text_kv_metadata"]["ropes"] == [6]
    assert branches["cfg_img_kv_metadata"]["ropes"] == [7]
    assert branches["extra_args"]["cfg_img_scale"] == 2.0
    assert branches["extra_args"]["cfg_renorm_type"] == "text_channel"
    assert sampling.past_key_values is None and sampling.cfg_text_past_key_values is None
    request.sampling_params.extra_args["cfg_img_scale"] = 1.0
    with pytest.raises(ValueError, match="reference"):
        pipeline.forward(request)
