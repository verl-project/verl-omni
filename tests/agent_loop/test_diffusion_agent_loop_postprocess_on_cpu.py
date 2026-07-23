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
"""CPU tests for reward_extra_info aggregation in DiffusionAgentLoopWorker._postprocess.

Sub-rewards do not publish a fixed key set per sample. ``MultiVisualRewardManager``
emits ``reward/{key}/{subkey}`` for every field of a dict result, but on an exception
it emits only ``reward/{key}``. ``_postprocess`` must therefore align a batch whose
samples carry different reward-extra keys.
"""

import numpy as np
import torch
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics

from verl_omni.agent_loop.diffusion_agent_loop import (
    DiffusionAgentLoopWorker,
    _InternalDiffusionAgentLoopOutput,
)

# Mirrors MultiVisualRewardManager: a dict-returning sub-reward publishes the extra
# field, the exception path publishes the bare score only.
_OK = {"reward/ocr": 1.0, "reward/ocr/genrm_response": "matched", "reward/combined": 1.0}
_FAILED = {"reward/ocr": 0.0, "reward/combined": 0.0}


def _make_output(reward_extra_info: dict) -> _InternalDiffusionAgentLoopOutput:
    return _InternalDiffusionAgentLoopOutput(
        prompt_ids=torch.zeros(1, 4, dtype=torch.long),
        response_diffusion_output=torch.zeros(1, 3, 8, 8),
        reward_score=float(reward_extra_info["reward/combined"]),
        num_turns=1,
        metrics=AgentLoopMetrics(),
        extra_fields={"reward_extra_info": dict(reward_extra_info)},
    )


def _postprocess(reward_extra_infos: list[dict]):
    """_postprocess never touches ``self``; call it unbound to avoid building a worker."""
    outputs = [_make_output(info) for info in reward_extra_infos]
    return DiffusionAgentLoopWorker._postprocess(None, outputs)


class TestRewardExtraInfoAggregation:
    def test_later_sample_missing_subkey_does_not_crash(self):
        """A sub-reward failing on any sample but the first must not kill the batch."""
        result = _postprocess([_OK, _FAILED])

        assert result.non_tensor_batch["reward/ocr/genrm_response"].tolist() == ["matched", None]
        np.testing.assert_allclose(result.non_tensor_batch["reward/ocr"].astype(float), [1.0, 0.0])

    def test_first_sample_missing_subkey_keeps_other_samples(self):
        """Keys are collected over the whole batch, not read off sample 0."""
        result = _postprocess([_FAILED, _OK])

        assert "reward/ocr/genrm_response" in result.non_tensor_batch
        assert result.non_tensor_batch["reward/ocr/genrm_response"].tolist() == [None, "matched"]
        assert "reward/ocr/genrm_response" in result.meta_info["reward_extra_keys"]

    def test_homogeneous_keys_keep_numeric_dtype(self):
        """When every sample agrees, arrays stay numeric so downstream metrics still work."""
        result = _postprocess([_FAILED, _FAILED])

        combined = result.non_tensor_batch["reward/combined"]
        assert np.issubdtype(combined.dtype, np.floating)
        np.testing.assert_allclose(combined, [0.0, 0.0])

    def test_all_samples_failed(self):
        """A sub-reward that fails everywhere yields no subkey column at all."""
        result = _postprocess([_FAILED, _FAILED])

        assert "reward/ocr/genrm_response" not in result.meta_info["reward_extra_keys"]

    def test_disjoint_key_sets_are_unioned(self):
        """Different sub-rewards failing on different samples still produce a full batch."""
        left = {"reward/combined": 1.0, "reward/a": 1.0, "reward/a/detail": "x"}
        right = {"reward/combined": 1.0, "reward/b": 1.0, "reward/b/detail": "y"}
        result = _postprocess([left, right])

        assert result.non_tensor_batch["reward/a/detail"].tolist() == ["x", None]
        assert result.non_tensor_batch["reward/b/detail"].tolist() == [None, "y"]
        for key in ("reward/a", "reward/b", "reward/a/detail", "reward/b/detail"):
            assert len(result.non_tensor_batch[key]) == 2
