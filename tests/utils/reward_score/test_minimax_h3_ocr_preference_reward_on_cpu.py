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
"""CPU unit tests for examples/diffusionnft_trainer/minimax_h3_image/ocr_preference_reward.py."""

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from verl.utils.import_utils import load_extern_object

REWARD_PATH = str(
    Path(__file__).resolve().parents[3]
    / "examples"
    / "diffusionnft_trainer"
    / "minimax_h3_image"
    / "ocr_preference_reward.py"
)


def _reset_shared_state():
    for key in [k for k in sys.modules if k.startswith("_verl_omni_minimax_h3_ocr_preference_state:")]:
        sys.modules.pop(key, None)


class TestOcrPreferenceReward:
    def test_preference_terms_share_one_scorer_across_separate_loads(self, monkeypatch, tmp_path):
        _reset_shared_state()
        monkeypatch.setenv("OCR_REWARD_TMPDIR", str(tmp_path))
        pick_fn = load_extern_object(REWARD_PATH, "term_pickscore")
        hps_fn = load_extern_object(REWARD_PATH, "term_hpsv2")

        spawns = []
        requests = []

        class _FakeProc:
            returncode = None

        async def fake_spawn(script, env):
            spawns.append(script)
            return _FakeProc()

        async def fake_request(proc, script, payload):
            requests.append((proc, script, payload["prompt"]))
            return {"pickscore": 0.72, "hpsv2": 0.31}

        for mod_fn in (pick_fn, hps_fn):
            ns = mod_fn.__globals__
            monkeypatch.setitem(ns, "_spawn", fake_spawn)
            monkeypatch.setitem(ns, "_request", fake_request)

        image = torch.full((3, 16, 16), 128, dtype=torch.uint8)
        extra_info = {"prompt": "cafe menu", "split": "train", "index": 0, "tier": "t20", "media_kind": "image"}

        import asyncio

        async def run():
            pick = await pick_fn("src", image, "target", extra_info)
            hps = await hps_fn("src", image, "target", extra_info)
            return pick, hps

        pick, hps = asyncio.run(run())
        assert pick["score"] == pytest.approx(0.72)
        assert hps["score"] == pytest.approx(0.31)
        assert spawns == ["preference_worker.py"]
        assert len(requests) == 1

    def test_ppocr_error_raises_instead_of_identity_guard(self, monkeypatch):
        _reset_shared_state()
        ocr_fn = load_extern_object(REWARD_PATH, "term_ocr_guarded")
        guard_value = ocr_fn.__globals__["guard_value"]

        with pytest.raises(RuntimeError, match="PP-OCR layout worker failed: boom"):
            guard_value({"area": 0.0, "border": 0, "line_h": 0.0, "error": "boom"})

        async def fake_ocr(*args, **kwargs):
            return {"score": 1.0}

        async def fake_layout(*args, **kwargs):
            return {"area": 0.0, "border": 0, "line_h": 0.0, "error": "detector crashed"}

        monkeypatch.setitem(ocr_fn.__globals__, "compute_score_ocr", fake_ocr)
        monkeypatch.setitem(ocr_fn.__globals__, "_layout", fake_layout)

        import asyncio

        with pytest.raises(RuntimeError, match="detector crashed"):
            asyncio.run(ocr_fn("src", torch.zeros(3, 16, 16, dtype=torch.uint8), "gt", {}))

    def test_full_prompt_keys_by_tier_split_index_and_rejects_duplicates(self, monkeypatch, tmp_path):
        _reset_shared_state()
        mod_fn = load_extern_object(REWARD_PATH, "full_prompt")

        p_t5 = tmp_path / "train_t5.parquet"
        p_t20 = tmp_path / "train_t20.parquet"
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "prompt": [{"role": "user", "content": "prompt tier t5 idx 0"}],
                        "extra_info": {"split": "train", "index": 0, "tier": "t5"},
                    }
                ]
            ),
            str(p_t5),
        )
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "prompt": [{"role": "user", "content": "prompt tier t20 idx 0"}],
                        "extra_info": {"split": "train", "index": 0, "tier": "t20"},
                    }
                ]
            ),
            str(p_t20),
        )
        monkeypatch.setenv("TRAIN_FILES", f"{p_t5},{p_t20}")
        monkeypatch.delenv("VAL_FILES", raising=False)

        assert mod_fn({"split": "train", "index": 0, "tier": "t5"}) == "prompt tier t5 idx 0"
        assert mod_fn({"split": "train", "index": 0, "tier": "t20"}) == "prompt tier t20 idx 0"

        _reset_shared_state()
        monkeypatch.setenv("TRAIN_FILES", f"{p_t5},{p_t5}")
        with pytest.raises(ValueError, match="Duplicate prompt key"):
            mod_fn({"split": "train", "index": 0, "tier": "t5"})
