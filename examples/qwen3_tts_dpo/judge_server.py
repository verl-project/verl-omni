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
"""Generic LLM-audio judge for online TTS DPO.

Serves the /rank contract AudioJudgeRewardManager speaks: POST {text, wavs_b64, debias} ->
{scores, chosen, rejected}. An audio-capable LLM compares two clips of the same text and picks the
more natural read. The provider is pluggable: Gemini (default) or any OpenAI-audio-compatible
endpoint (a hosted API or a locally vLLM-served audio model) via --base-url.

    python judge_server.py --provider gemini --model gemini-3.5-flash --port 8901
    python judge_server.py --provider openai --base-url http://localhost:8000/v1 \
        --model Qwen2.5-Omni-7B --api-key-env OPENAI_API_KEY --port 8901
"""

import argparse
import base64
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SYSTEM = (
    "You are an expert judge of text-to-speech. You will hear two clips, A then B, both reading the "
    "same target text. Decide which is the more natural, human-sounding read. Reply with exactly one "
    "token: A, B, or tie."
)


def _user_prompt(text):
    return f'Target text: "{text}"\nClip A is first, clip B is second. Which is better? Answer A, B, or tie.'


def _parse_label(out):
    t = (out or "").strip().lower()
    if "tie" in t:
        return 0.5
    head = t[:6]
    if "a" in head and "b" not in head:
        return 1.0
    if "b" in head and "a" not in head:
        return 0.0
    return 0.5


class Judge:
    """Return P(clip A is better) in {0.0, 0.5, 1.0} for one ordered A-vs-B call."""

    def compare(self, text, wav_a, wav_b):
        raise NotImplementedError


class GeminiJudge(Judge):
    def __init__(self, model):
        from google import genai

        self._model = model
        self._client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    def compare(self, text, wav_a, wav_b):
        from google.genai import types

        contents = [
            SYSTEM,
            _user_prompt(text),
            types.Part.from_bytes(data=wav_a, mime_type="audio/wav"),
            types.Part.from_bytes(data=wav_b, mime_type="audio/wav"),
        ]
        resp = self._client.models.generate_content(model=self._model, contents=contents)
        return _parse_label(resp.text)


class OpenAICompatJudge(Judge):
    """Any OpenAI-audio-compatible chat endpoint (hosted or a locally vLLM-served audio LLM)."""

    def __init__(self, model, base_url, api_key_env):
        from openai import OpenAI

        self._model = model
        self._client = OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env, "EMPTY"))

    def compare(self, text, wav_a, wav_b):
        def audio_part(wav):
            return {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}}

        user_content = [{"type": "text", "text": _user_prompt(text)}, audio_part(wav_a), audio_part(wav_b)]
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_content},
        ]
        resp = self._client.chat.completions.create(model=self._model, messages=messages, temperature=0.0)
        return _parse_label(resp.choices[0].message.content)


def _pref(judge, text, wa, wb, debias):
    """P(a better than b), optionally averaged with the swapped order to cancel position bias."""
    p = judge.compare(text, wa, wb)
    if debias:
        p = 0.5 * (p + (1.0 - judge.compare(text, wb, wa)))
    return p


def _rank(judge, text, wavs, debias):
    """Per-candidate score in [0, 1]. n==2 is one (debiased) call; n>2 is a round-robin average.

    A debiased tie yields equal scores, so the DPO pairing drops that group.
    """
    n = len(wavs)
    if n == 2:
        s = _pref(judge, text, wavs[0], wavs[1], debias)
        scores = [s, 1.0 - s]
    else:
        scores = []
        for i in range(n):
            others = [_pref(judge, text, wavs[i], wavs[j], debias) for j in range(n) if j != i]
            scores.append(sum(others) / max(1, len(others)))
    chosen = max(range(n), key=lambda i: scores[i])
    rejected = min(range(n), key=lambda i: scores[i])
    return {"scores": scores, "chosen": chosen, "rejected": rejected}


def make_handler(judge, sem):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"status": "ok", "model": judge._model})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/rank":
                self._send(404, {"error": "not found"})
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                wavs = [base64.b64decode(w) for w in payload["wavs_b64"]]
                with sem:
                    result = _rank(judge, payload.get("text", ""), wavs, bool(payload.get("debias", True)))
                self._send(200, result)
            except Exception as e:  # noqa: BLE001
                self._send(500, {"error": str(e)})

    return Handler


def build_judge(args):
    if args.provider == "gemini":
        return GeminiJudge(args.model)
    if args.provider == "openai":
        return OpenAICompatJudge(args.model, args.base_url, args.api_key_env)
    raise ValueError(f"unknown provider {args.provider!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["gemini", "openai"], default="gemini")
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint base url")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY", help="env var holding the openai api key")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--max-concurrency", type=int, default=16, help="cap simultaneous LLM calls")
    args = ap.parse_args()

    judge = build_judge(args)
    sem = threading.Semaphore(args.max_concurrency)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(judge, sem))
    print(f"audio judge ({args.provider}:{args.model}) listening on {args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
