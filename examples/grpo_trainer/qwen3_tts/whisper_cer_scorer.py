#!/usr/bin/env python3
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
"""Serve a fail-closed Whisper CER reward over the audio JSON protocol."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import re
import threading
import unicodedata
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", str(text))
    normalized = re.sub(r"[।॥.,!?;:\"'`´’‘“”()\[\]{}<>/\\|@#%^&*_+=~–—-]", " ", normalized).lower()
    return " ".join(normalized.split())


def _edit_distance(reference, hypothesis) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_item in enumerate(reference, 1):
        current = [row]
        for column, hypothesis_item in enumerate(hypothesis, 1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + int(reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def char_error_rate(hypothesis: str, reference: str) -> float:
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _edit_distance(reference, hypothesis) / len(reference)


def decode_request(payload: Any) -> tuple[np.ndarray, int, str, dict]:
    if not isinstance(payload, dict):
        raise ValueError("Audio scorer request must be a JSON object.")
    if payload.get("protocol_version") != "1":
        raise ValueError("Audio scorer requires protocol_version='1'.")
    encoded = payload.get("waveform_f32_base64")
    if not isinstance(encoded, str):
        raise ValueError("waveform_f32_base64 must be a base64 string.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("waveform_f32_base64 is not valid base64.") from exc
    if len(raw) % np.dtype("<f4").itemsize:
        raise ValueError("Decoded waveform byte length is not float32-aligned.")
    waveform = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)

    num_samples = payload.get("num_samples")
    if isinstance(num_samples, bool) or not isinstance(num_samples, int) or num_samples != waveform.size:
        raise ValueError(f"num_samples does not match the decoded waveform: {num_samples!r} != {waveform.size}.")
    sample_rate = payload.get("sample_rate")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise ValueError(f"sample_rate must be a positive integer, got {sample_rate!r}.")
    if not np.isfinite(waveform).all():
        raise ValueError("Decoded waveform contains NaN or infinity.")

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not normalize_text(prompt):
        raise ValueError("Audio scorer requires a non-empty prompt.")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object.")
    return waveform, sample_rate, prompt, metadata


class WhisperCERScorer:
    def __init__(
        self,
        model: str,
        *,
        revision: str | None,
        device: str,
        dtype: str,
        language: str,
        local_files_only: bool,
    ) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype!r}.")
        if device == "cpu" and dtype != "float32":
            raise ValueError("CPU Whisper inference requires --dtype=float32.")

        self.device = device
        self.dtype = dtype_map[dtype]
        self.language = language
        self.processor = AutoProcessor.from_pretrained(
            model,
            revision=revision,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model,
            revision=revision,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            local_files_only=local_files_only,
        ).to(device)
        self.model.eval()
        self._lock = threading.Lock()

    def transcribe(self, waveform: np.ndarray, sample_rate: int) -> str:
        import torch
        import torchaudio.functional as audio_functional

        samples = torch.from_numpy(waveform)
        if sample_rate != 16_000:
            samples = audio_functional.resample(samples, sample_rate, 16_000)
        features = self.processor(
            samples.numpy(),
            sampling_rate=16_000,
            return_tensors="pt",
        ).input_features.to(device=self.device, dtype=self.dtype)
        with self._lock, torch.inference_mode():
            generated = self.model.generate(
                features,
                language=self.language,
                task="transcribe",
                do_sample=False,
                condition_on_prev_tokens=False,
            )
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    def score(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        target: str,
        metadata: dict,
    ) -> dict:
        transcript = self.transcribe(waveform, sample_rate)
        cer = char_error_rate(transcript, target)
        cer_capped = min(1.0, max(0.0, cer))

        sample_id = metadata.get("id", metadata.get("index", ""))
        return {
            "score": float(1.0 - cer_capped),
            "cer": float(cer),
            "cer_capped": float(cer_capped),
            "asr_text": transcript,
            "normalized_target": normalize_text(target),
            "normalized_hypothesis": normalize_text(transcript),
            "sample_id": str(sample_id),
        }


class WhisperCERHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, scorer) -> None:
        super().__init__(server_address, WhisperCERRequestHandler)
        self.scorer = scorer


class WhisperCERRequestHandler(BaseHTTPRequestHandler):
    server: WhisperCERHTTPServer

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/healthz":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return
        self._send_json(HTTPStatus.OK, {"status": "ok"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/score":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            return
        try:
            content_length = int(self.headers.get("Content-Length", ""))
            if not 0 < content_length <= 8 * 1024 * 1024:
                raise ValueError("Content-Length must be between 1 byte and 8 MiB.")
            payload = json.loads(self.rfile.read(content_length))
            waveform, sample_rate, prompt, metadata = decode_request(payload)
            result = self.server.scorer.score(waveform, sample_rate, prompt, metadata)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:  # The client treats a scorer failure as fatal.
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Whisper scoring failed: {exc}"})
            return
        self._send_json(HTTPStatus.OK, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Whisper checkpoint or Hugging Face model ID")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--language", default="hi")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    scorer = WhisperCERScorer(
        args.model,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        language=args.language,
        local_files_only=not args.allow_download,
    )
    server = WhisperCERHTTPServer((args.host, args.port), scorer)
    print(json.dumps({"status": "ready", "host": args.host, "port": server.server_port}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
