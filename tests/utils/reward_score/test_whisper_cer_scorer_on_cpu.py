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
"""CPU tests for the external Hindi Whisper-CER scorer."""

import base64
import importlib.util
import json
import sys
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def scorer():
    module_path = Path(__file__).parents[3] / "examples/grpo_trainer/qwen3_tts/whisper_cer_scorer.py"
    spec = importlib.util.spec_from_file_location("whisper_cer_scorer_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _request(waveform, prompt="सही", metadata=None, sample_rate=16_000):
    waveform = np.asarray(waveform, dtype="<f4")
    return {
        "protocol_version": "1",
        "waveform_f32_base64": base64.b64encode(waveform.tobytes()).decode("ascii"),
        "num_samples": waveform.size,
        "sample_rate": sample_rate,
        "prompt": prompt,
        "metadata": metadata or {},
    }


def _fake_scorer(scorer, transcript="सही"):
    instance = object.__new__(scorer.WhisperCERScorer)
    instance.transcribe = lambda waveform, sample_rate: transcript
    return instance


def test_normalization_and_error_rates_preserve_devanagari(scorer):
    assert scorer.normalize_text(" नमस्ते।  WORLD—हाँ! ") == "नमस्ते world हाँ"
    assert scorer.char_error_rate("abc", "abc") == 0.0
    assert scorer.char_error_rate("ab", "abc") == pytest.approx(1 / 3)


def test_audio_protocol_decodes_float32_waveform(scorer):
    waveform = np.linspace(-0.5, 0.5, 160, dtype=np.float32)

    decoded, sample_rate, prompt, metadata = scorer.decode_request(_request(waveform, metadata={"id": "hi-1"}))

    np.testing.assert_array_equal(decoded, waveform)
    assert sample_rate == 16_000
    assert prompt == "सही"
    assert metadata == {"id": "hi-1"}


def test_audio_protocol_fails_closed_on_length_mismatch(scorer):
    payload = _request(np.ones(160, dtype=np.float32))
    payload["num_samples"] = 159

    with pytest.raises(ValueError, match="does not match"):
        scorer.decode_request(payload)


def test_audio_protocol_requires_metadata(scorer):
    payload = _request(np.ones(160, dtype=np.float32))
    del payload["metadata"]

    with pytest.raises(ValueError, match="metadata must be a JSON object"):
        scorer.decode_request(payload)


def test_linear_cer_reward_uses_deterministic_transcript(scorer):
    waveform = np.full(16_000, 0.1, dtype=np.float32)

    matched = _fake_scorer(scorer, "नमस्ते दुनिया").score(
        waveform,
        16_000,
        "नमस्ते दुनिया",
        {"id": "hi-17"},
    )
    mismatched = _fake_scorer(scorer, "नमस्ते दुनिया").score(
        waveform,
        16_000,
        "नमस्ते",
        {},
    )

    assert matched["score"] == pytest.approx(1.0)
    assert matched["cer"] == 0.0
    assert matched["cer_capped"] == 0.0
    assert matched["sample_id"] == "hi-17"
    assert mismatched["score"] == pytest.approx(1.0 - mismatched["cer_capped"])


def test_http_server_implements_audio_json_contract(scorer):
    class DummyScorer:
        def score(self, waveform, sample_rate, prompt, metadata):
            return {"score": float(waveform.mean()), "sample_rate": sample_rate, "prompt": prompt}

    server = scorer.WhisperCERHTTPServer(("127.0.0.1", 0), DummyScorer())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/score",
            data=json.dumps(_request(np.full(160, 0.25, dtype=np.float32))).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            result = json.loads(response.read())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result == {"score": 0.25, "sample_rate": 16_000, "prompt": "सही"}
