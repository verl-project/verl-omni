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
"""CPU tests for the safetensors teacher lam_head loader."""

import json
import os

import pytest
import torch
from safetensors.torch import save_file

from verl_omni.utils.lm_head import DEFAULT_LM_HEAD_KEYS, _iter_lm_head_candidates, load_lm_head_weight


def _write_model(tmp_path, weights: dict[str, torch.Tensor], shard_by_key: bool = False):
    """Write weights into tmp_path; single-file or sharded per shard_by_key."""
    if not shard_by_key:
        save_file({k: v.contiguous() for k, v in weights.items()}, os.path.join(tmp_path, "model.safetensors"))
        return
    weight_map = {}
    for i, (k, v) in enumerate(weights.items()):
        shard = f"model-{i:05d}-of-{len(weights):05d}.safetensors"
        save_file({k: v.contiguous()}, os.path.join(tmp_path, shard))
        weight_map[k] = shard
    with open(os.path.join(tmp_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": weight_map}, f)


class TestIterCandidates:
    def test_single_file_picks_thinker_lm_head(self, tmp_path):
        w = torch.randn(5, 3)
        _write_model(tmp_path, {"thinker.lm_head.weight": w})
        got = list(_iter_lm_head_candidates(str(tmp_path), DEFAULT_LM_HEAD_KEYS))
        assert [k for k, _ in got] == ["thinker.lm_head.weight"]

    def test_sharded_index_resolves_shard(self, tmp_path):
        w = torch.randn(5, 3)
        other = torch.randn(4, 4)
        _write_model(tmp_path, {"thinker.lm_head.weight": w, "other.weight": other}, shard_by_key=True)
        got = list(_iter_lm_head_candidates(str(tmp_path), DEFAULT_LM_HEAD_KEYS))
        assert [k for k, _ in got] == ["thinker.lm_head.weight"]
        assert torch.equal(got[0][1], w)

    def test_tied_embed_fallback(self, tmp_path):
        w = torch.randn(6, 2)
        _write_model(tmp_path, {"thinker.model.embed_tokens.weight": w})
        got = list(_iter_lm_head_candidates(str(tmp_path), DEFAULT_LM_HEAD_KEYS))
        assert [k for k, _ in got] == ["thinker.model.embed_tokens.weight"]


class TestLoadLmHeadWeight:
    def test_returns_fp32_cpu(self, tmp_path):
        w = torch.randn(5, 3)
        _write_model(tmp_path, {"thinker.lm_head.weight": w})
        out = load_lm_head_weight(str(tmp_path))
        assert out.dtype == torch.float32
        assert out.device.type == "cpu"
        assert torch.equal(out, w.float())

    def test_missing_key_raises(self, tmp_path):
        _write_model(tmp_path, {"unrelated.weight": torch.randn(2, 2)})
        with pytest.raises(ValueError, match="Could not find lm_head"):
            load_lm_head_weight(str(tmp_path))
