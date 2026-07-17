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
"""CPU unit tests for the tts reward modules (stubbed backends, no models, no GPU)."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from verl_omni.utils.reward_score import tts
from verl_omni.utils.reward_score.tts import asr_reward, emo_reward, spk_sim_reward


def _audio(seconds=10.0, sr=24000):
    return np.ones(int(seconds * sr), dtype=np.float32), sr


@pytest.fixture
def stub_backends(monkeypatch):
    monkeypatch.setattr(asr_reward, "transcribe", lambda wav, sr, *a, **kw: "hello world")
    monkeypatch.setattr(spk_sim_reward, "_spk_embed", lambda wav, sr, *a: np.asarray([1.0, 0.0], dtype=np.float32))
    monkeypatch.setattr(spk_sim_reward, "_ref_cache", {})
    monkeypatch.setattr(emo_reward, "_resample", lambda wav, sr, target_sr: wav)
    monkeypatch.setattr(
        emo_reward,
        "_get_emo",
        lambda *a: SimpleNamespace(generate=lambda *aa, **kk: [{"scores": [0.0, 0.0, 0.0, 0.7, 0.2, 0, 0, 0, 0]}]),
    )


class TestAsrReward:
    def test_glm_form(self, stub_backends):
        r = asr_reward.compute_score(_audio(), "hello world")
        assert r["score"] == 1.0 and r["cer"] == 0.0 and r["stab"] == 0.0 and r["synth_ok"] == 1.0

    def test_score_tracks_cer(self, stub_backends, monkeypatch):
        monkeypatch.setattr(asr_reward, "transcribe", lambda wav, sr, *a, **kw: "hxllo world")
        r = asr_reward.compute_score(_audio(), "hello world")
        assert r["cer"] > 0.0
        assert abs(r["score"] - math.exp(-2.5 * r["cer"])) < 1e-9

    def test_stability_penalties(self, stub_backends, monkeypatch):
        # truncated: 20 words expect 8s at 2.5 wps, a 1s clip is below the 0.5 ratio
        text = " ".join(["word"] * 20)
        monkeypatch.setattr(asr_reward, "transcribe", lambda wav, sr, *a, **kw: text)
        r = asr_reward.compute_score(_audio(seconds=1.0), text)
        assert r["truncated"] == 1.0 and r["stab"] <= -1.0

        monkeypatch.setattr(
            asr_reward, "transcribe", lambda wav, sr, *a, **kw: "please hold on please hold on please hold on"
        )
        r = asr_reward.compute_score(_audio(), "please hold on")
        assert r["repeated"] == 1.0

    def test_synth_failure_keeps_key_set(self, stub_backends):
        ok = asr_reward.compute_score(_audio(), "hello world")
        fail = asr_reward.compute_score(None, "hello world")
        assert set(fail.keys()) == set(ok.keys())
        assert fail["score"] == 0.0 and fail["stab"] == -1.0 and fail["synth_ok"] == 0.0

    def test_cer_and_normalization(self):
        assert asr_reward.normalize_text("Hello, World!") == "hello world"
        assert asr_reward.cer("abc", "abc") == 0.0
        assert asr_reward.cer("abcd", "abxd") == 0.25
        assert asr_reward.cer("", "anything") is None

    def test_has_repetition(self):
        assert asr_reward.has_repetition("the the the cat cat cat", n=3) is False
        assert asr_reward.has_repetition("please hold on please hold on", n=2) is True
        assert asr_reward.has_repetition("a normal sentence with no loops", n=3) is False


class TestSpkSimReward:
    def test_cosine_form(self, stub_backends, tmp_path):
        import soundfile as sf

        ref = tmp_path / "ref.wav"
        sf.write(ref, np.ones(16000, dtype=np.float32), 16000)
        r = spk_sim_reward.compute_score(_audio(), extra_info={"ref_audio": str(ref)})
        assert abs(r["score"] - 1.0) < 1e-6 and abs(r["cos"] - 1.0) < 1e-6

    def test_missing_inputs_score_zero(self, stub_backends):
        assert spk_sim_reward.compute_score(None, extra_info={})["score"] == 0.0
        assert spk_sim_reward.compute_score(_audio(), extra_info={})["score"] == 0.0


class TestEmoReward:
    def test_tagged_and_untagged(self, stub_backends):
        r = emo_reward.compute_score(_audio(), extra_info={"emotion": 3})
        assert r == {"score": 0.7, "p_neutral": 0.2}
        r = emo_reward.compute_score(_audio(), extra_info={})
        assert abs(r["score"] - 0.8) < 1e-9  # 1 - P(neutral)

    def test_unavailable_backend_scores_zero(self, monkeypatch):
        monkeypatch.setattr(emo_reward, "_get_emo", lambda *a: None)
        assert emo_reward.compute_score(_audio(), extra_info={})["score"] == 0.0


class TestComposition:
    def test_weighted_sum_and_dimensions(self, stub_backends, tmp_path):
        import soundfile as sf

        ref = tmp_path / "ref.wav"
        sf.write(ref, np.ones(16000, dtype=np.float32), 16000)
        r = tts.compute_score(_audio(), "hello world", extra_info={"ref_audio": str(ref)}, weights={"sim": 2.0})
        assert r["asr"] == 1.0 and abs(r["sim"] - 1.0) < 1e-6 and abs(r["emo"] - 0.8) < 1e-9
        assert r["stab"] == 0.0
        assert abs(r["score"] - (1.0 + 2.0 * 1.0 + 0.8 + 0.0)) < 1e-6

    def test_synth_failure_keeps_key_set(self, stub_backends):
        ok = tts.compute_score(_audio(), "hello world", extra_info={})
        fail = tts.compute_score(None, "hello world", extra_info={})
        assert set(fail.keys()) == set(ok.keys())
        assert fail["asr"] == fail["sim"] == fail["emo"] == 0.0 and fail["stab"] == -1.0
