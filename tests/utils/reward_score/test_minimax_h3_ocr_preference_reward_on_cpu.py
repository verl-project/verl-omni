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
    def test_worker_backend_matches_launcher(self, monkeypatch):
        scorer_device = load_extern_object(str(Path(REWARD_PATH).with_name("preference_worker.py")), "_scorer_device")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setenv("H3_IMAGE_DEVICE", "")
        assert scorer_device() == torch.device("cuda:0")
        monkeypatch.setenv("H3_IMAGE_DEVICE", "typo")
        with pytest.raises(ValueError, match="Unsupported H3_IMAGE_DEVICE"):
            scorer_device()

    @pytest.mark.parametrize(
        ("device_kind", "hip", "visibility"),
        [
            ("cuda", None, "CUDA_VISIBLE_DEVICES"),
            ("npu", None, "ASCEND_RT_VISIBLE_DEVICES"),
            ("cuda", "7.0", "HIP_VISIBLE_DEVICES"),
        ],
    )
    def test_preference_terms_share_one_scorer_across_separate_loads(
        self, monkeypatch, tmp_path, device_kind, hip, visibility
    ):
        _reset_shared_state()
        monkeypatch.setenv("OCR_REWARD_TMPDIR", str(tmp_path))
        monkeypatch.setenv("H3_IMAGE_DEVICE", device_kind)
        monkeypatch.setattr(torch.version, "hip", hip)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0,1,2")
        monkeypatch.setenv("PREFERENCE_SCORER_GPU", "7")
        monkeypatch.setenv(visibility, "0,1,2,3,4,5,6")
        pick_fn = load_extern_object(REWARD_PATH, "term_pickscore")
        hps_fn = load_extern_object(REWARD_PATH, "term_hpsv2")

        spawns = []
        requests = []

        class _FakeProc:
            returncode = None

        async def fake_spawn(script, env):
            assert env[visibility] == "7"
            if hip:
                assert "CUDA_VISIBLE_DEVICES" not in env
                assert "ROCR_VISIBLE_DEVICES" not in env
            spawns.append(script)
            return _FakeProc()

        async def fake_request(proc, script, payload):
            requests.append((proc, script, payload["prompt"]))
            if payload["prompt"] == "other prompt":
                return {"pickscore": 0.41, "hpsv2": 0.19}
            return {"pickscore": 0.72, "hpsv2": 0.31}

        for mod_fn in (pick_fn, hps_fn):
            ns = mod_fn.__globals__
            monkeypatch.setitem(ns, "_spawn", fake_spawn)
            monkeypatch.setitem(ns, "_request", fake_request)

        image = torch.full((3, 16, 16), 128, dtype=torch.uint8)
        extra_info = {"prompt": "cafe menu", "split": "train", "index": 0, "tier": "t20", "media_kind": "image"}
        other_info = {"prompt": "other prompt", "split": "train", "index": 0, "tier": "t20", "media_kind": "image"}

        import asyncio

        async def run():
            pick = await pick_fn("src", image, "target", extra_info)
            hps = await hps_fn("src", image, "target", extra_info)
            other = await pick_fn("src", image, "target", other_info)
            return pick, hps, other

        pick, hps, other = asyncio.run(run())
        assert pick["score"] == pytest.approx(0.72)
        assert hps["score"] == pytest.approx(0.31)
        assert other["score"] == pytest.approx(0.41)
        assert spawns == ["preference_worker.py"]
        assert [r[2] for r in requests] == ["cafe menu", "other prompt"]

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

    def test_request_timeout_kills_child_and_raises(self):
        import asyncio

        _reset_shared_state()
        fn = load_extern_object(REWARD_PATH, "term_ocr_guarded")
        request_fn = fn.__globals__["_request"]
        fn.__globals__["TIMEOUT_S"] = 0.01

        class _HangingStdout:
            async def readline(self):
                await asyncio.sleep(5)
                return b""

        class _FakeStdin:
            def write(self, data):
                pass

            async def drain(self):
                pass

        class _FakeProc:
            returncode = None
            stdin = _FakeStdin()
            stdout = _HangingStdout()
            killed = False

            def kill(self):
                self.killed = True
                self.returncode = -9

            async def wait(self):
                return self.returncode

        proc = _FakeProc()
        with pytest.raises(RuntimeError, match="gave no reply"):
            asyncio.run(request_fn(proc, "ocr_layout_worker.py", {"path": "x.png"}))
        assert proc.killed is True

    def test_layout_pool_preserves_slot_on_failed_respawn_and_recovers(self, monkeypatch, tmp_path):
        import asyncio

        _reset_shared_state()
        monkeypatch.setenv("OCR_REWARD_TMPDIR", str(tmp_path))
        fn = load_extern_object(REWARD_PATH, "term_ocr_guarded")
        ns = fn.__globals__
        monkeypatch.setitem(ns, "POOL_SIZE", 1)
        monkeypatch.setitem(ns, "TIMEOUT_S", 0.05)

        class _Proc:
            def __init__(self, idx, die_on_request=False):
                self.idx = idx
                self.returncode = None
                self.die_on_request = die_on_request

        spawn_count = 0
        fail_next_spawn = False

        async def fake_spawn(script, env):
            nonlocal spawn_count
            if fail_next_spawn:
                raise RuntimeError("spawn failed")
            spawn_count += 1
            return _Proc(spawn_count, die_on_request=(spawn_count == 1))

        async def fake_request(proc, script, payload):
            if proc.die_on_request:
                proc.returncode = 1
                raise RuntimeError("worker died")
            return {"area": 0.04, "border": 0, "line_h": 12.0}

        monkeypatch.setitem(ns, "_spawn", fake_spawn)
        monkeypatch.setitem(ns, "_request", fake_request)

        image = torch.full((3, 16, 16), 128, dtype=torch.uint8)
        extra_info = {"media_kind": "image"}
        layout_fn = ns["_layout"]
        get_state = ns["_get_state"]

        async def run_phases():
            nonlocal fail_next_spawn
            # Phase 1: first call spawns worker 1, worker 1 dies during _request, respawn fails -> slot becomes None.
            orig_request = fake_request

            async def failing_first_request(proc, script, payload):
                nonlocal fail_next_spawn
                fail_next_spawn = True
                return await orig_request(proc, script, payload)

            monkeypatch.setitem(ns, "_request", failing_first_request)
            with pytest.raises(RuntimeError, match="worker died"):
                await layout_fn(image, extra_info)
            assert get_state().guard_pool.qsize() == 1

            # Phase 2: respawn succeeds on the next call using the preserved slot.
            fail_next_spawn = False
            monkeypatch.setitem(ns, "_request", orig_request)
            res = await layout_fn(image, extra_info)
            assert res == {"area": 0.04, "border": 0, "line_h": 12.0}
            assert spawn_count == 2
            assert get_state().guard_pool.qsize() == 1

            # Phase 3: empty pool times out instead of hanging forever.
            slot = await get_state().guard_pool.get()
            try:
                with pytest.raises(RuntimeError, match="no idle worker"):
                    await layout_fn(image, extra_info)
            finally:
                get_state().guard_pool.put_nowait(slot)

        asyncio.run(run_phases())

    def test_preference_respawns_dead_worker(self, monkeypatch, tmp_path):
        import asyncio

        _reset_shared_state()
        monkeypatch.setenv("OCR_REWARD_TMPDIR", str(tmp_path))
        pick_fn = load_extern_object(REWARD_PATH, "term_pickscore")
        ns = pick_fn.__globals__

        class _Proc:
            def __init__(self, idx):
                self.idx = idx
                self.returncode = None

        spawned = []

        async def fake_spawn(script, env):
            proc = _Proc(len(spawned) + 1)
            spawned.append(proc)
            return proc

        async def fake_request(proc, script, payload):
            if proc.idx == 1:
                proc.returncode = -9
                raise RuntimeError("preference worker exited")
            return {"pickscore": 0.65, "hpsv2": 0.25}

        monkeypatch.setitem(ns, "_spawn", fake_spawn)
        monkeypatch.setitem(ns, "_request", fake_request)

        image = torch.full((3, 16, 16), 128, dtype=torch.uint8)
        extra_info = {"prompt": "cafe menu", "media_kind": "image"}

        async def run():
            with pytest.raises(RuntimeError, match="exited"):
                await pick_fn("src", image, "target", extra_info)
            res = await pick_fn("src", image, "target", extra_info)
            return res

        res = asyncio.run(run())
        assert res["score"] == pytest.approx(0.65)
        assert len(spawned) == 2
