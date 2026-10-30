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
    module_path = Path(__file__).parents[3] / "examples/grpo_trainer/qwen3_tts/scorer/whisper_cer.py"
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
    assert scorer.__all__ == ["WhisperCERScorer"]
    assert scorer.WhisperCERScorer.__doc__
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


def test_audio_protocol_accepts_the_supported_sample_rate_range(scorer):
    waveform = np.ones(160, dtype=np.float32)

    for rate in (8_000, 16_000, 24_000, 384_000):
        decoded, accepted, _, _ = scorer.decode_request(_request(waveform, sample_rate=rate))

        assert accepted == rate
        assert decoded.size == 160


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


class _FakeDeviceType:
    def __init__(self, device_type, index=None):
        self.type = device_type
        self.index = index


class _FakeAccelerator:
    """Models ``torch.accelerator``: ``current_accelerator()`` yields a device whose
    index is unknown, and ``current_device_index()`` is where the index comes from."""

    _UNSET = object()

    def __init__(self, current, *, runtime_available=True, current_index=_UNSET):
        self._current = current
        self._runtime_available = runtime_available
        self._current_index = current_index
        self.check_available_calls = []

    def current_accelerator(self, check_available=False):
        self.check_available_calls.append(check_available)
        if not check_available:
            # The compile-time answer, which ignores whether a device exists.
            return self._current
        return self._current if self._runtime_available else None

    def current_device_index(self):
        # Real torch reports the index here; by default mirror the device shown.
        if self._current_index is self._UNSET:
            return self._current.index if self._current is not None and self._current.index is not None else 0
        if isinstance(self._current_index, BaseException):
            raise self._current_index
        return self._current_index


class _FakeBackend:
    def __init__(self, available):
        self._available = available

    def is_available(self):
        return self._available


class _ExplodingBackend:
    def is_available(self):
        raise RuntimeError("driver missing")


def _fake_torch(accelerator=None, **backends):
    attributes = {name: _FakeBackend(available) for name, available in backends.items()}
    if accelerator is not None:
        attributes["accelerator"] = accelerator
    return type("FakeTorch", (), attributes)()


@pytest.mark.parametrize(
    ("torch_module", "expected"),
    [
        (_fake_torch(accelerator=_FakeAccelerator(_FakeDeviceType("npu"))), "npu:0"),
        (_fake_torch(accelerator=_FakeAccelerator(_FakeDeviceType("npu", 3))), "npu:3"),
        (_fake_torch(accelerator=_FakeAccelerator(_FakeDeviceType("cuda", 1))), "cuda:1"),
        # ``current_accelerator()`` never carries an index in real torch; the index
        # of the device in use has to come from ``current_device_index()``.
        (_fake_torch(accelerator=_FakeAccelerator(_FakeDeviceType("npu"), current_index=3)), "npu:3"),
        (_fake_torch(accelerator=_FakeAccelerator(_FakeDeviceType("cuda"), current_index=2)), "cuda:2"),
        # A build that raises from the accessor must not break device selection.
        (
            _fake_torch(accelerator=_FakeAccelerator(_FakeDeviceType("npu"), current_index=RuntimeError("no index"))),
            "npu:0",
        ),
        (_fake_torch(accelerator=_FakeAccelerator(_FakeDeviceType("cpu")), cuda=False, npu=True), "npu:0"),
        (_fake_torch(cuda=True, npu=True), "cuda:0"),
        (_fake_torch(cuda=False, npu=False, xpu=True), "xpu:0"),
        # MPS is not probed: auto must never pick it on an Apple-silicon host.
        (_fake_torch(cuda=False, npu=False, xpu=False, mps=True), "cpu"),
        (_fake_torch(cuda=False, npu=False), "cpu"),
    ],
)
def test_auto_device_uses_a_non_cuda_accelerator(scorer, torch_module, expected):
    """Probing CUDA alone resolves to CPU on Ascend NPU, which then rejects float16."""
    assert scorer._resolve_auto_device(torch_module) == expected


def test_auto_device_falls_back_to_cpu_when_a_probe_raises(scorer):
    torch_module = _fake_torch(accelerator=_FakeAccelerator(_FakeDeviceType("cpu")), npu=False)
    torch_module.cuda = _ExplodingBackend()

    assert scorer._resolve_auto_device(torch_module) == "cpu"


def test_auto_device_uses_runtime_availability_not_the_compile_time_answer(scorer):
    """A build compiled with CUDA on a host with no usable CUDA device must not
    select ``cuda:0``; the compile-time answer alone is not enough."""
    torch_module = _fake_torch(
        accelerator=_FakeAccelerator(_FakeDeviceType("cuda"), runtime_available=False),
        cuda=False,
        npu=False,
    )

    assert scorer._resolve_auto_device(torch_module) == "cpu"


def test_auto_device_supports_builds_without_the_check_available_kwarg(scorer):
    """Older builds only accept a bare ``current_accelerator()``."""

    class _NoKwargAccelerator:
        def current_accelerator(self):
            return _FakeDeviceType("npu")

    torch_module = _fake_torch(accelerator=_NoKwargAccelerator(), npu=True)

    assert scorer._resolve_auto_device(torch_module) == "npu:0"
