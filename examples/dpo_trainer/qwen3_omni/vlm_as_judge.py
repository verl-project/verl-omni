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
"""Compare reference and trained Qwen3-Omni answers with a MiniCPM-o judge.

The script expects:

1. a transformers Qwen3-Omni generation backend. The base model is loaded via
   ``Qwen3OmniMoeForConditionalGeneration``, the PEFT LoRA adapter is attached
   with ``PeftModel.from_pretrained``, and answers are produced with
   ``model.generate()``. Reference answers disable adapters; trained answers
   enable them.
2. a MiniCPM-o OpenAI-compatible judge endpoint, by default
   ``openbmb/MiniCPM-o-4_5``.

It reads held-out Omni-Preference rows, generates one answer with the reference
model and one with the trained LoRA for each prompt, then asks MiniCPM-o to
score and compare the two answers under the original multimodal input.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import logging
import mimetypes
import os
import random
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("qwen3_omni_minicpm_judge")
pd: Any | None = None
torch: Any | None = None

DEFAULT_JUDGE_SERVER_COMMAND = (
    "vllm serve {model} --host {host} --port {port} --dtype bfloat16 --trust-remote-code --enforce-eager"
)
DEFAULT_JUDGE_MODEL = "openbmb/MiniCPM-o-4_5"
DEFAULT_TRAINED_MODEL_NAME = "qwen3-omni-trained"
DEFAULT_IMAGE_MIN_PIXELS = 3136
DEFAULT_IMAGE_MAX_PIXELS = 602112
DEFAULT_VIDEO_MIN_PIXELS = 100352
DEFAULT_VIDEO_MAX_PIXELS = 602112
DEFAULT_VIDEO_MIN_FRAMES = 4
DEFAULT_VIDEO_MAX_FRAMES = 8
DEFAULT_VIDEO_FPS = 2.0
MEDIA_KEYS = (("image", "images"), ("video", "videos"), ("audio", "audios"))
MEDIA_CONTENT_TYPES = {
    "image": ("image_url", "image_url"),
    "video": ("video_url", "video_url"),
    "audio": ("audio_url", "audio_url"),
}
JUDGE_DIMENSIONS = ("fluency", "relevance", "accuracy", "reasoning_quality", "safety")
_MODALITY_PREFIX_RE = re.compile(r"^<(image|video|audio)>\s*", flags=re.IGNORECASE)
_GLOBAL_STEP_MODALITY_RE = re.compile(r"global_step_(?P<step>\d+)(?:_(?P<modality>[A-Za-z0-9_-]+))?$")


@dataclass
class EvalSample:
    data_file: str
    index: int
    uid: str
    modality: str
    prompt_text: str
    media: dict[str, list[str]]
    raw_prompt: Any


@dataclass
class JudgeResult:
    reference_score: float
    trained_score: float
    winner: str
    rationale: str
    raw_response: str
    reference_dimension_scores: dict[str, float] = field(default_factory=dict)
    trained_dimension_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class SummaryStats:
    total: int = 0
    trained_wins: int = 0
    reference_wins: int = 0
    ties: int = 0
    reference_score_sum: float = 0.0
    trained_score_sum: float = 0.0

    def update(self, result: JudgeResult) -> None:
        self.total += 1
        self.reference_score_sum += result.reference_score
        self.trained_score_sum += result.trained_score
        if result.winner == "trained":
            self.trained_wins += 1
        elif result.winner == "reference":
            self.reference_wins += 1
        else:
            self.ties += 1

    def to_dict(self) -> dict[str, float | int]:
        if self.total == 0:
            return {
                "total": 0,
                "trained_wins": 0,
                "reference_wins": 0,
                "ties": 0,
                "trained_win_rate": 0.0,
                "tie_rate": 0.0,
                "reference_mean_score": 0.0,
                "trained_mean_score": 0.0,
                "mean_score_margin": 0.0,
            }
        reference_mean = self.reference_score_sum / self.total
        trained_mean = self.trained_score_sum / self.total
        return {
            "total": self.total,
            "trained_wins": self.trained_wins,
            "reference_wins": self.reference_wins,
            "ties": self.ties,
            "trained_win_rate": self.trained_wins / self.total,
            "tie_rate": self.ties / self.total,
            "reference_mean_score": reference_mean,
            "trained_mean_score": trained_mean,
            "mean_score_margin": trained_mean - reference_mean,
        }


class ManagedServer(AbstractContextManager):
    """Small subprocess manager for optional local model servers."""

    def __init__(
        self,
        *,
        command: str | None,
        router_address: str,
        timeout_s: float,
        name: str,
    ) -> None:
        self.command = command
        self.router_address = router_address
        self.timeout_s = timeout_s
        self.name = name
        self.process: subprocess.Popen | None = None

    def __enter__(self):
        if not self.command:
            wait_for_server(self.router_address, self.timeout_s, name=self.name)
            return self
        logger.info("Launching %s server: %s", self.name, self.command)
        self.process = subprocess.Popen(self.command, shell=True)
        try:
            wait_for_server(self.router_address, self.timeout_s, name=self.name)
        except Exception:
            self._terminate()
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._terminate()
        return False

    def _terminate(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        logger.info("Stopping %s server", self.name)
        if os.name == "nt":
            self.process.send_signal(signal.CTRL_BREAK_EVENT if hasattr(signal, "CTRL_BREAK_EVENT") else signal.SIGTERM)
        else:
            self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-files",
        nargs="+",
        default=[],
        help="Held-out Omni-Preference parquet/json/jsonl files. "
        "Not required when summarizing cached judge jsonl files.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Root directory containing <modality>/test.parquet files. Used when --data-files is omitted.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=[],
        help="Modalities to evaluate under --data-dir, e.g. image video audio.",
    )
    parser.add_argument(
        "--data-file-name",
        default="test.parquet",
        help="Dataset file name under each modality directory when using --data-dir.",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Where per-sample generation and judge results are written. "
        "If this points to an existing directory with judge jsonl files, stages are skipped and cached results "
        "are summarized.",
    )
    parser.add_argument(
        "--reference-jsonl",
        default=None,
        help="Where reference/base generations are cached. Defaults to <output-jsonl>.reference.jsonl.",
    )
    parser.add_argument(
        "--trained-jsonl",
        default=None,
        help="Where trained/LoRA generations are cached. Defaults to <output-jsonl>.trained.jsonl.",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["reference", "trained", "judge", "all"],
        help="Run one stage or the full staged pipeline.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum number of samples to evaluate from each input data file.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for A/B judge order.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    parser.add_argument("--model-path", default=None, help="Base Qwen3-Omni model path or HF repo id.")
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="PEFT LoRA adapter directory, or a verl FSDP / global_step checkpoint. "
        "Non-PEFT checkpoints are exported via export_fsdp_lora_adapter, then loaded "
        "with PeftModel.from_pretrained for trained generation.",
    )
    parser.add_argument("--trained-model-name", default=DEFAULT_TRAINED_MODEL_NAME)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Torch dtype used when loading Qwen3-Omni.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="Attention implementation passed to from_pretrained (e.g. sdpa, flash_attention_2).",
    )
    parser.add_argument(
        "--device-map",
        default="cuda-offload-non-thinker",
        choices=["auto", "cuda-offload-non-thinker", "cuda", "cpu"],
        help="How to place Qwen3-Omni modules. "
        "'cuda-offload-non-thinker' keeps thinker on one visible CUDA device and "
        "offloads talker/code2wav to CPU; "
        "'auto' is plain device_map='auto'; "
        "'cuda'/'cpu' place the whole model on one device.",
    )
    parser.add_argument(
        "--use-audio-in-video",
        action="store_true",
        help="Forward use_audio_in_video=True to process_mm_info / processor / generate.",
    )
    parser.add_argument("--generation-max-tokens", type=int, default=512)
    parser.add_argument("--generation-temperature", type=float, default=0.0)
    parser.add_argument("--generation-top-p", type=float, default=1.0)
    parser.add_argument("--image-min-pixels", type=int, default=DEFAULT_IMAGE_MIN_PIXELS)
    parser.add_argument("--image-max-pixels", type=int, default=DEFAULT_IMAGE_MAX_PIXELS)
    parser.add_argument("--video-min-pixels", type=int, default=DEFAULT_VIDEO_MIN_PIXELS)
    parser.add_argument("--video-max-pixels", type=int, default=DEFAULT_VIDEO_MAX_PIXELS)
    parser.add_argument("--video-min-frames", type=int, default=DEFAULT_VIDEO_MIN_FRAMES)
    parser.add_argument("--video-max-frames", type=int, default=DEFAULT_VIDEO_MAX_FRAMES)
    parser.add_argument("--video-fps", type=float, default=DEFAULT_VIDEO_FPS)

    parser.add_argument("--judge-router-address", default="127.0.0.1:8001")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--launch-judge-server", action="store_true")
    parser.add_argument("--judge-server-host", default="127.0.0.1")
    parser.add_argument("--judge-server-port", type=int, default=8001)
    parser.add_argument(
        "--judge-server-command",
        default=DEFAULT_JUDGE_SERVER_COMMAND,
        help="Command template used with --launch-judge-server.",
    )
    parser.add_argument("--judge-max-tokens", type=int, default=1024)
    parser.add_argument(
        "--judge-retry-max-tokens",
        type=int,
        default=4096,
        help="Retry judge parsing once with this max token budget when the first response cannot be parsed.",
    )
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--server-timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Where to save the stage summary JSON. Defaults to <output-jsonl>.summary.json.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _require_pandas() -> Any:
    global pd
    if pd is None:
        import pandas as pandas_module

        pd = pandas_module
    return pd


def _require_torch() -> Any:
    global torch
    if torch is None:
        import torch as torch_module

        torch = torch_module
    return torch


def _as_python(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") or text.startswith("{"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(_require_pandas().isna(value))
    except (TypeError, ValueError):
        return False


def _normalise_list(value: Any) -> list[Any]:
    value = _as_python(value)
    if _is_missing(value):
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        return [item for item in value if not _is_missing(item)]
    return [value]


def _content_to_text(content: Any) -> str:
    content = _as_python(content)
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if content.get("type") == "text":
            return str(content.get("text", ""))
        if "content" in content:
            return _content_to_text(content["content"])
        if "text" in content:
            return str(content["text"])
        return ""
    if isinstance(content, Sequence):
        return "\n".join(part for item in content if (part := _content_to_text(item)))
    return "" if _is_missing(content) else str(content)


def _append_media_from_content(content: Any, media: dict[str, list[str]]) -> None:
    content = _as_python(content)
    if isinstance(content, dict):
        item_type = content.get("type")
        for modality, media_key in MEDIA_KEYS:
            if item_type == modality and content.get(modality):
                media[media_key].append(str(content[modality]))
        if "content" in content:
            _append_media_from_content(content["content"], media)
        return
    if isinstance(content, Sequence) and not isinstance(content, str):
        for item in content:
            _append_media_from_content(item, media)


def _prompt_to_text_and_media(prompt: Any, row: dict[str, Any]) -> tuple[str, dict[str, list[str]]]:
    prompt = _as_python(prompt)
    media = {
        "images": [str(item) for item in _normalise_list(row.get("images"))],
        "videos": [str(item) for item in _normalise_list(row.get("videos"))],
        "audios": [str(item) for item in _normalise_list(row.get("audios"))],
    }
    if isinstance(prompt, str):
        return prompt, media

    parts = []
    for message in prompt or []:
        message = _as_python(message)
        if not isinstance(message, dict):
            continue
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "user":
            parts.append(_content_to_text(content))
        _append_media_from_content(content, media)
    return "\n".join(part for part in parts if part), media


def _infer_modality(row: dict[str, Any], media: dict[str, list[str]]) -> str:
    for modality, media_key in MEDIA_KEYS:
        if media[media_key]:
            return modality
    data_source = str(row.get("data_source", "")).lower()
    for modality, _media_key in MEDIA_KEYS:
        if modality in data_source:
            return modality
    return "text"


def read_samples(data_files: list[str], max_samples: int) -> list[EvalSample]:
    pd_module = _require_pandas()
    samples: list[EvalSample] = []
    for data_file in data_files:
        path = Path(data_file)
        if path.suffix == ".jsonl":
            dataframe = pd_module.read_json(path, lines=True)
        elif path.suffix == ".json":
            dataframe = pd_module.read_json(path)
        else:
            dataframe = pd_module.read_parquet(path)
        if max_samples > 0:
            dataframe = dataframe.head(max_samples)
        for index, row in enumerate(dataframe.to_dict(orient="records")):
            prompt_text, media = _prompt_to_text_and_media(row.get("prompt", []), row)
            modality = _infer_modality(row, media)
            samples.append(
                EvalSample(
                    data_file=str(path),
                    index=index,
                    uid=str(row.get("uid") or f"{path.name}:{index}"),
                    modality=modality,
                    prompt_text=prompt_text,
                    media=media,
                    raw_prompt=row.get("prompt", []),
                )
            )
    return samples


def resolve_eval_data_files(args: argparse.Namespace) -> list[str]:
    if args.data_files:
        return list(args.data_files)
    if args.data_dir and args.modalities:
        data_dir = Path(args.data_dir)
        return [str(data_dir / modality / args.data_file_name) for modality in args.modalities]
    return []


def _media_to_data_url(path_or_url: str, modality: str) -> str:
    if path_or_url.startswith(("http://", "https://", "data:")):
        return path_or_url
    path = Path(os.path.expanduser(path_or_url))
    if not path.is_file():
        raise FileNotFoundError(f"{modality} media file does not exist: {path}")
    mime_type = mimetypes.guess_type(path.name)[0]
    if mime_type is None:
        mime_type = {
            "image": "image/png",
            "video": "video/mp4",
            "audio": "audio/wav",
        }[modality]
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_multimodal_content(sample: EvalSample, *, include_media: bool = True) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if include_media:
        for modality, media_key in MEDIA_KEYS:
            for media_path in sample.media[media_key]:
                content_type, field_name = MEDIA_CONTENT_TYPES[modality]
                content.append({content_type: {"url": _media_to_data_url(media_path, modality)}, "type": content_type})
    if sample.modality != "text" and include_media:
        expected_key = f"{sample.modality}s"
        if not sample.media.get(expected_key):
            raise ValueError(f"Sample {sample.uid} is modality={sample.modality} but has no {expected_key} media.")
    content.append({"type": "text", "text": sample.prompt_text})
    return content


def _post_json(router_address: str, path: str, payload: dict[str, Any]) -> dict[str, Any] | str:
    url = f"http://{router_address}{path}"
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=None) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed with HTTP {exc.code}: {body}") from exc
    if not body:
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def _post_chat_completion(router_address: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = _post_json(router_address, "/v1/chat/completions", payload)
    if not isinstance(response, dict):
        raise RuntimeError(f"Expected JSON object from chat completions, got: {response!r}")
    return response


def wait_for_server(router_address: str, timeout_s: float, *, name: str) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://{router_address}/v1/models"
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status < 500:
                    logger.info("%s server is ready at %s", name, router_address)
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for {name} server at {router_address}: {last_error}")


def _message_text(response: dict[str, Any]) -> str:
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Malformed chat completion response: {response}") from exc
    if isinstance(content, list):
        return "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    return "" if content is None else str(content)


def build_judge_prompt(answer_a: str, answer_b: str) -> str:
    return (
        "You are a strict multimodal answer evaluator. Given the original multimodal user input and two candidate "
        "answers, evaluate which answer has higher overall quality.\n\n"
        "Score each answer from 1 to 10 for these dimensions: fluency, relevance, accuracy, reasoning_quality, "
        "and safety. Then provide an overall_score from 1 to 10.\n\n"
        "Candidate A:\n"
        f"{answer_a}\n\n"
        "Candidate B:\n"
        f"{answer_b}\n\n"
        "Return only valid JSON with this schema. Do not include hidden reasoning, chain-of-thought, Markdown, "
        "code fences, or any text outside the JSON object:\n"
        "{\n"
        '  "A": {"overall_score": 0, "fluency": 0, "relevance": 0, "accuracy": 0, '
        '"reasoning_quality": 0, "safety": 0},\n'
        '  "B": {"overall_score": 0, "fluency": 0, "relevance": 0, "accuracy": 0, '
        '"reasoning_quality": 0, "safety": 0},\n'
        '  "winner": "A|B|TIE",\n'
        '  "rationale": "short explanation"\n'
        "}"
    )


def _strip_markdown_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _without_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def _balanced_json_objects(text: str) -> list[str]:
    objects = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            current = text[index]
            if in_string:
                if escape:
                    escape = False
                elif current == "\\":
                    escape = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    objects.append(text[start : index + 1])
                    break
    return objects


def _extract_object_after_key(text: str, key: str) -> dict[str, Any] | None:
    match = re.search(rf'(?:"{re.escape(key)}"|{re.escape(key)})\s*:', text)
    if not match:
        return None
    remainder = text[match.end() :]
    for candidate in _balanced_json_objects(remainder):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            value = _score_block_from_text(candidate)
        if isinstance(value, dict):
            return value
    return None


def _score_block_from_text(text: str) -> dict[str, float] | None:
    block: dict[str, float] = {}
    for key in ("overall_score", *JUDGE_DIMENSIONS):
        key_pattern = re.escape(key).replace("_", r"[_\s-]?")
        match = re.search(
            rf'"?{key_pattern}"?\s*(?:\([^)]*\))?\s*[:=]\s*(?:[^0-9-]{{0,120}})?(-?\d+(?:\.\d+)?)',
            text,
            flags=re.IGNORECASE,
        )
        if match:
            block[key] = float(match.group(1))
    if "overall_score" not in block:
        dimension_scores = [block[key] for key in JUDGE_DIMENSIONS if key in block]
        if dimension_scores:
            block["overall_score"] = sum(dimension_scores) / len(dimension_scores)
    return block or None


def _text_section_after_label(text: str, label: str) -> str | None:
    label_pattern = rf"(?:Candidate|Answer)\s+{re.escape(label)}\b|^\s*(?:\*\*)?{re.escape(label)}(?:\*\*)?\s*:"
    label_match = re.search(label_pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not label_match:
        return None
    next_label = "B" if label == "A" else None
    if next_label is None:
        return text[label_match.end() :]
    next_match = re.search(
        rf"(?:Candidate|Answer)\s+{re.escape(next_label)}\b|^\s*(?:\*\*)?{re.escape(next_label)}(?:\*\*)?\s*:",
        text[label_match.end() :],
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if not next_match:
        return text[label_match.end() :]
    return text[label_match.end() : label_match.end() + next_match.start()]


def _textual_judge_payload(text: str) -> dict[str, Any] | None:
    section_a = _text_section_after_label(text, "A")
    section_b = _text_section_after_label(text, "B")
    if section_a is None or section_b is None:
        return None
    score_a = _score_block_from_text(section_a)
    score_b = _score_block_from_text(section_b)
    if score_a is None or score_b is None:
        return None
    winner_match = re.search(r"(?:winner|better answer)\D{0,80}\b(A|B|TIE)\b", text, flags=re.IGNORECASE)
    return {
        "A": score_a,
        "B": score_b,
        "winner": winner_match.group(1).upper() if winner_match else "TIE",
        "rationale": "Recovered from non-JSON judge response.",
    }


def _fallback_judge_payload(text: str) -> dict[str, Any] | None:
    score_a = _extract_object_after_key(text, "A")
    score_b = _extract_object_after_key(text, "B")
    if score_a is None or score_b is None:
        return _textual_judge_payload(text)
    winner_match = re.search(r'"?winner"?\s*:\s*"([^"]+)"', text)
    rationale_match = re.search(r'"?rationale"?\s*:\s*"([^"]*)"', text)
    return {
        "A": score_a,
        "B": score_b,
        "winner": winner_match.group(1) if winner_match else "TIE",
        "rationale": rationale_match.group(1) if rationale_match else "",
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    candidates = [_strip_markdown_json(text), _strip_markdown_json(_without_think_blocks(text))]
    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
        else:
            if isinstance(payload, dict):
                return payload

        for object_text in reversed(_balanced_json_objects(candidate)):
            try:
                payload = json.loads(object_text)
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if isinstance(payload, dict) and "A" in payload and "B" in payload:
                return payload

        fallback = _fallback_judge_payload(candidate)
        if fallback is not None:
            return fallback

    if last_error is not None:
        raise last_error
    raise json.JSONDecodeError("No JSON object found in judge response", text, 0)


def _score_block(payload: dict[str, Any], label: str) -> tuple[float, dict[str, float]]:
    block = payload.get(label, {})
    if not isinstance(block, dict):
        block = {}
    dimensions = {}
    for key in JUDGE_DIMENSIONS:
        try:
            dimensions[key] = float(block.get(key, 0.0))
        except (TypeError, ValueError):
            dimensions[key] = 0.0
    try:
        overall = float(block.get("overall_score", 0.0))
    except (TypeError, ValueError):
        overall = 0.0
    if overall == 0.0 and dimensions:
        overall = sum(dimensions.values()) / len(dimensions)
    return overall, dimensions


def parse_judge_response(text: str, label_to_model: dict[str, str]) -> JudgeResult:
    payload = _extract_json_object(text)
    score_a, dims_a = _score_block(payload, "A")
    score_b, dims_b = _score_block(payload, "B")
    raw_winner = str(payload.get("winner", "TIE")).upper()
    if raw_winner not in {"A", "B", "TIE"}:
        winner_tokens = [token for token in re.split(r"[^A-Z]+", raw_winner) if token in {"A", "B", "TIE"}]
        raw_winner = winner_tokens[-1] if winner_tokens else "TIE"

    scores = {
        label_to_model["A"]: score_a,
        label_to_model["B"]: score_b,
    }
    dims = {
        label_to_model["A"]: dims_a,
        label_to_model["B"]: dims_b,
    }
    winner = "tie" if raw_winner == "TIE" else label_to_model[raw_winner]
    return JudgeResult(
        reference_score=scores["reference"],
        trained_score=scores["trained"],
        winner=winner,
        rationale=str(payload.get("rationale", "")),
        raw_response=text,
        reference_dimension_scores=dims["reference"],
        trained_dimension_scores=dims["trained"],
    )


def judge_pair(
    *,
    sample: EvalSample,
    reference_text: str,
    trained_text: str,
    router_address: str,
    model_name: str,
    max_tokens: int,
    retry_max_tokens: int,
    temperature: float,
    rng: random.Random,
) -> JudgeResult:
    if rng.random() < 0.5:
        answer_a, answer_b = reference_text, trained_text
        label_to_model = {"A": "reference", "B": "trained"}
    else:
        answer_a, answer_b = trained_text, reference_text
        label_to_model = {"A": "trained", "B": "reference"}

    content = build_multimodal_content(sample)
    content.append({"type": "text", "text": build_judge_prompt(answer_a, answer_b)})

    def request_judge(token_budget: int) -> str:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": token_budget,
            "temperature": temperature,
            "top_p": 1.0,
        }
        return _message_text(_post_chat_completion(router_address, payload))

    response_text = request_judge(max_tokens)
    try:
        return parse_judge_response(response_text, label_to_model)
    except json.JSONDecodeError as exc:
        if retry_max_tokens > max_tokens:
            logger.warning(
                "Failed to parse judge response for uid=%s with max_tokens=%d (%s). Retrying with max_tokens=%d.",
                sample.uid,
                max_tokens,
                exc,
                retry_max_tokens,
            )
            response_text = request_judge(retry_max_tokens)
            try:
                return parse_judge_response(response_text, label_to_model)
            except json.JSONDecodeError:
                logger.error(
                    "Failed to parse retried judge response for uid=%s. Raw response:\n%s", sample.uid, response_text
                )
                raise
        logger.error("Failed to parse judge response for uid=%s. Raw response:\n%s", sample.uid, response_text)
        raise


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rewrite_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sample_key_parts(data_file: str, index: int, uid: str) -> tuple[str, int, str]:
    return (data_file, int(index), uid)


def sample_key(sample: EvalSample) -> tuple[str, int, str]:
    return _sample_key_parts(sample.data_file, sample.index, sample.uid)


def row_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return _sample_key_parts(str(row["data_file"]), int(row["index"]), str(row["uid"]))


def default_stage_jsonl_path(output_jsonl: str, role: str) -> Path:
    output_path = Path(output_jsonl)
    suffix = output_path.suffix or ".jsonl"
    return output_path.with_name(f"{output_path.stem}.{role}{suffix}")


def generation_jsonl_path(args: argparse.Namespace, role: str) -> Path:
    explicit = args.reference_jsonl if role == "reference" else args.trained_jsonl
    if explicit:
        return Path(explicit)
    if role in {"reference", "trained"} and args.stage == role:
        return Path(args.output_jsonl)
    return default_stage_jsonl_path(args.output_jsonl, role)


def summary_json_path(args: argparse.Namespace) -> Path:
    if args.summary_json:
        return Path(args.summary_json)
    output_path = Path(args.output_jsonl)
    if output_path.is_dir():
        return output_path / "summary.json"
    if output_path.suffix:
        return output_path.with_suffix(".summary.json")
    return output_path.with_name(f"{output_path.name}.summary.json")


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _is_judge_result_row(row: dict[str, Any]) -> bool:
    return "reference_score" in row and "trained_score" in row and "modality" in row


def _judge_result_rows(path: Path) -> list[dict[str, Any]]:
    if path.stem.endswith((".reference", ".trained")):
        return []
    rows = read_jsonl_rows(path)
    return [row for row in rows if _is_judge_result_row(row)]


def discover_cached_judge_jsonls(output_path: Path) -> list[Path]:
    if output_path.is_dir():
        candidates = sorted(output_path.glob("*.jsonl"))
    else:
        return []
    return [path for path in candidates if _judge_result_rows(path)]


def _global_step_from_path(path: Path) -> int | None:
    match = _GLOBAL_STEP_MODALITY_RE.search(path.stem)
    if not match:
        return None
    return int(match.group("step"))


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def summarize_cached_judge_results(judge_paths: list[Path]) -> dict[str, Any]:
    grouped: dict[tuple[int | None, str], SummaryStats] = defaultdict(SummaryStats)
    source_files: dict[tuple[int | None, str], set[str]] = defaultdict(set)
    for path in judge_paths:
        global_step = _global_step_from_path(path)
        for row in _judge_result_rows(path):
            result = JudgeResult(
                reference_score=float(row["reference_score"]),
                trained_score=float(row["trained_score"]),
                winner=str(row.get("winner", "tie")),
                rationale=str(row.get("rationale", "")),
                raw_response=str(row.get("judge_raw_response", "")),
            )
            key = (global_step, str(row["modality"]))
            grouped[key].update(result)
            source_files[key].add(str(path))

    table_rows = []
    for (global_step, modality), stats in sorted(
        grouped.items(), key=lambda item: (-1 if item[0][0] is None else item[0][0], item[0][1])
    ):
        stats_dict = stats.to_dict()
        table_rows.append(
            {
                "global_step": "unknown" if global_step is None else global_step,
                "modality": modality,
                "samples": stats.total,
                "reference_mean_score": stats_dict["reference_mean_score"],
                "trained_mean_score": stats_dict["trained_mean_score"],
                "mean_score_margin": stats_dict["mean_score_margin"],
                "trained_win_rate": stats_dict["trained_win_rate"],
                "source_files": sorted(source_files[(global_step, modality)]),
            }
        )

    table = format_cached_judge_table(table_rows)
    return {
        "stage": "cached_judge_summary",
        "skipped_stages": ["reference", "trained", "judge"],
        "judge_jsonls": [str(path) for path in judge_paths],
        "table": table,
        "rows": table_rows,
    }


def format_cached_judge_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| global_step | modality | samples | reference_avg | trained_avg | margin | trained_win_rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['global_step']} | "
            f"{row['modality']} | "
            f"{row['samples']} | "
            f"{_format_float(float(row['reference_mean_score']))} | "
            f"{_format_float(float(row['trained_mean_score']))} | "
            f"{_format_float(float(row['mean_score_margin']))} | "
            f"{_format_float(float(row['trained_win_rate']))} |"
        )
    return "\n".join(lines)


def summarize_judge_rows(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, float | int], dict[str, dict[str, float | int]]]:
    overall = SummaryStats()
    by_modality: dict[str, SummaryStats] = defaultdict(SummaryStats)
    for row in rows:
        if not _is_judge_result_row(row):
            continue
        result = JudgeResult(
            reference_score=float(row["reference_score"]),
            trained_score=float(row["trained_score"]),
            winner=str(row.get("winner", "tie")),
            rationale=str(row.get("rationale", "")),
            raw_response=str(row.get("judge_raw_response", "")),
        )
        overall.update(result)
        by_modality[str(row.get("modality", "unknown"))].update(result)
    return overall.to_dict(), {modality: stats.to_dict() for modality, stats in sorted(by_modality.items())}


def completed_generation_keys(path: Path) -> set[tuple[str, int, str]]:
    return {row_key(row) for row in read_jsonl_rows(path)}


def load_generation_cache(path: Path, role: str) -> dict[tuple[str, int, str], dict[str, Any]]:
    rows = read_jsonl_rows(path)
    cache = {}
    for row in rows:
        if row.get("model_role") != role:
            logger.warning("Ignoring %s row with unexpected model_role=%s in %s", role, row.get("model_role"), path)
            continue
        cache[row_key(row)] = row
    return cache


def _load_fsdp_utils_module():
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "verl_omni" / "utils" / "fsdp_utils.py"
    spec = importlib.util.spec_from_file_location("_verl_omni_fsdp_utils_for_judge", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load FSDP utils from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _export_fsdp_lora_adapter(
    *,
    input_dir: Path,
    output_dir: Path,
    base_model_name_or_path: str | None,
) -> None:
    module = _load_fsdp_utils_module()
    module.export_fsdp_lora_adapter(
        input_dir=input_dir,
        output_dir=output_dir,
        base_model_name_or_path=base_model_name_or_path,
    )


def _is_peft_adapter_dir(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and (
        (path / "adapter_model.safetensors").is_file() or (path / "adapter_model.bin").is_file()
    )


def _is_fsdp_lora_checkpoint_dir(path: Path) -> bool:
    return (path / "fsdp_config.json").is_file() and (path / "lora_train_meta.json").is_file()


def resolve_adapter_path(adapter_path: str, base_model_name_or_path: str | None = None) -> str:
    """Return a PEFT LoRA adapter directory for ``PeftModel.from_pretrained``.

    Accepts a PEFT adapter directory directly. For verl FSDP / ``global_step_*``
    checkpoints, reuses an existing ``lora_adapter/`` export when present,
    otherwise converts the checkpoint with ``export_fsdp_lora_adapter``.
    """

    path = Path(os.path.expanduser(adapter_path)).resolve()
    if _is_peft_adapter_dir(path):
        logger.info("Using PEFT LoRA adapter: %s", path)
        return str(path)

    for candidate in (path / "lora_adapter", path / "actor" / "lora_adapter"):
        if _is_peft_adapter_dir(candidate):
            logger.info("Using exported PEFT LoRA adapter: %s", candidate)
            return str(candidate)

    fsdp_checkpoint_dir = path
    if not _is_fsdp_lora_checkpoint_dir(fsdp_checkpoint_dir) and _is_fsdp_lora_checkpoint_dir(path / "actor"):
        fsdp_checkpoint_dir = path / "actor"

    if _is_fsdp_lora_checkpoint_dir(fsdp_checkpoint_dir):
        output_dir = fsdp_checkpoint_dir / "lora_adapter"
        if not _is_peft_adapter_dir(output_dir):
            logger.info("Exporting FSDP LoRA checkpoint %s to %s", fsdp_checkpoint_dir, output_dir)
            _export_fsdp_lora_adapter(
                input_dir=fsdp_checkpoint_dir,
                output_dir=output_dir,
                base_model_name_or_path=base_model_name_or_path,
            )
        else:
            logger.info("Reusing exported PEFT LoRA adapter: %s", output_dir)
        return str(output_dir)

    raise FileNotFoundError(
        f"`--adapter-path` must point to a PEFT LoRA adapter directory "
        f"(adapter_config.json + adapter_model.safetensors|.bin), a verl FSDP "
        f"actor checkpoint, or a global_step checkpoint containing actor/. Got: {adapter_path}"
    )


def _format_command(template: str, *, model: str, host: str, port: int, args: argparse.Namespace) -> str:
    return template.format(
        model=model,
        host=host,
        port=port,
        trained_model_name=args.trained_model_name,
    )


def _strip_modality_placeholders(text: str) -> str:
    return _MODALITY_PREFIX_RE.sub("", text.strip())


def build_generation_conversation(sample: EvalSample, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Build a Qwen3-Omni chat conversation for transformers + process_mm_info."""
    content: list[dict[str, Any]] = []
    for modality, media_key in MEDIA_KEYS:
        for media_path in sample.media[media_key]:
            media_content: dict[str, Any] = {"type": modality, modality: media_path}
            if modality == "image":
                media_content.update(
                    {
                        "min_pixels": args.image_min_pixels,
                        "max_pixels": args.image_max_pixels,
                    }
                )
            elif modality == "video":
                media_content.update(
                    {
                        "min_pixels": args.video_min_pixels,
                        "max_pixels": args.video_max_pixels,
                        "min_frames": args.video_min_frames,
                        "max_frames": args.video_max_frames,
                        "fps": args.video_fps,
                    }
                )
            content.append(media_content)
    question = _strip_modality_placeholders(sample.prompt_text)
    if not question and not content:
        raise ValueError(f"Sample {sample.uid} has neither text nor media for generation.")
    if question:
        content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def _resolve_torch_dtype(dtype: str) -> str | torch.dtype:
    if dtype == "auto":
        return "auto"
    return getattr(_require_torch(), dtype)


def _resolve_model_device(model: Any) -> torch.device:
    torch_module = _require_torch()
    for owner in (model, getattr(model, "base_model", None), getattr(model, "model", None)):
        device_map = getattr(owner, "hf_device_map", None)
        if not isinstance(device_map, dict):
            continue
        for key in ("base_model.model.thinker", "model.thinker", "thinker", ""):
            if key not in device_map:
                continue
            value = device_map[key]
            if value in {"cpu", "disk", "meta"}:
                continue
            if isinstance(value, int):
                return torch_module.device(f"cuda:{value}")
            return torch_module.device(value)

    device = getattr(model, "device", None)
    if isinstance(device, torch_module.device):
        return device
    if isinstance(device, str):
        return torch_module.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch_module.device("cpu")


def _move_inputs_to_model(inputs: Any, model: Any) -> Any:
    torch_module = _require_torch()
    device = _resolve_model_device(model)
    dtype = getattr(model, "dtype", None)
    moved = inputs.to(device)
    if dtype is None or not hasattr(moved, "items"):
        return moved
    for key, value in list(moved.items()):
        if torch_module.is_tensor(value) and torch_module.is_floating_point(value):
            moved[key] = value.to(dtype)
    return moved


CPU_OFFLOAD_MODULE_PREFIXES = ("talker", "code2wav")


def build_omni_device_map(
    model_path: str,
    *,
    dtype: str | torch.dtype,
    strategy: str,
) -> str | dict[str, Any]:
    """Resolve a transformers/accelerate device_map for Qwen3-Omni.

    ``cuda-offload-non-thinker`` keeps the active thinker path on one visible CUDA
    device and offloads unused talker/code2wav modules to CPU. This avoids the
    cross-device thinker sharding that Qwen3-Omni generation cannot currently
    tolerate while still reducing VRAM.
    """
    if strategy == "auto":
        return "auto"
    if strategy in {"cuda", "cpu"}:
        return strategy

    if strategy != "cuda-offload-non-thinker":
        raise ValueError(f"Unsupported device-map strategy: {strategy}")

    torch_module = _require_torch()
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3OmniMoeForConditionalGeneration

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    with init_empty_weights():
        # Qwen3-Omni exposes `_from_config` (PreTrainedModel), not public `from_config`.
        empty = Qwen3OmniMoeForConditionalGeneration._from_config(config)

    offload_prefixes = [name for name in CPU_OFFLOAD_MODULE_PREFIXES if hasattr(empty, name)]
    # Drop non-thinker modules before memory planning so thinker can use that VRAM.
    for prefix in offload_prefixes:
        try:
            delattr(empty, prefix)
        except Exception:
            setattr(empty, prefix, None)

    if not torch_module.cuda.is_available():
        raise RuntimeError("`cuda-offload-non-thinker` requires at least one visible CUDA device.")
    device_map: dict[str, Any] = {"": 0}
    for prefix in offload_prefixes:
        device_map[prefix] = "cpu"

    logger.info(
        "Built cuda-offload device_map: cuda_root=0 cpu_offload=%s placed_modules=%d",
        offload_prefixes,
        len(device_map),
    )
    return device_map


def _adapter_context(model: Any, *, trained: bool):
    """Enable LoRA for trained generation; disable it for the reference baseline."""
    if trained:
        enable_adapter_layers = getattr(model, "enable_adapter_layers", None)
        if callable(enable_adapter_layers):
            start = time.perf_counter()
            enable_adapter_layers()
            logger.info("Enabled PEFT adapter layers in %.3fs", time.perf_counter() - start)
        return nullcontext()

    disable_adapter = getattr(model, "disable_adapter", None)
    if callable(disable_adapter):
        return disable_adapter()

    disable_adapter_layers = getattr(model, "disable_adapter_layers", None)
    enable_adapter_layers = getattr(model, "enable_adapter_layers", None)
    if callable(disable_adapter_layers) and callable(enable_adapter_layers):
        disable_adapter_layers()

        class _ReenableAdapters:
            def __enter__(self):
                return model

            def __exit__(self, exc_type, exc, tb) -> bool:
                enable_adapter_layers()
                return False

        return _ReenableAdapters()

    return nullcontext()


THINKER_ATTN_EXCLUDE_MODULES = r"^(?!.*thinker\.model\.layers\.).*(q_proj|k_proj|v_proj|o_proj)$"


def _prepare_generation_peft_config(peft_config: Any, _model: Any):
    """Rewrite fused-MoE LoRA configs so PEFT can inject them.

    FSDP export infers ``target_modules=['experts', ...]`` from keys like
    ``...mlp.experts.lora_A.weight``. PEFT cannot wrap
    ``Qwen3OmniMoeThinkerTextExperts`` as Linear; fused experts must use
    ``target_parameters=['gate_up_proj', 'down_proj']``.
    """
    modules, params = _load_fsdp_utils_module().split_fused_moe_lora_targets(
        getattr(peft_config, "target_modules", None),
        getattr(peft_config, "target_parameters", None),
    )
    peft_config.target_modules = modules
    if params:
        peft_config.target_parameters = params
    peft_config.exclude_modules = THINKER_ATTN_EXCLUDE_MODULES
    logger.info(
        "Prepared PEFT LoRA config: target_modules=%s target_parameters=%s exclude_modules=%s",
        peft_config.target_modules,
        getattr(peft_config, "target_parameters", None),
        peft_config.exclude_modules,
    )
    return peft_config


class TransformersOmniGenerator(AbstractContextManager):
    """Qwen3-Omni generation via transformers ``generate`` with a PEFT LoRA adapter."""

    def __init__(self, args: argparse.Namespace, *, load_adapter: bool):
        self.args = args
        self.load_adapter = load_adapter
        self.model = None
        self.processor = None

    def __enter__(self):
        from qwen_omni_utils import process_mm_info
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        self._process_mm_info = process_mm_info
        dtype = _resolve_torch_dtype(self.args.dtype)
        device_map = build_omni_device_map(
            self.args.model_path,
            dtype=dtype,
            strategy=self.args.device_map,
        )
        logger.info(
            "Loading Qwen3-Omni with transformers: model=%s dtype=%s attn=%s device_map=%s",
            self.args.model_path,
            self.args.dtype,
            self.args.attn_implementation,
            self.args.device_map,
        )
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            self.args.model_path,
            dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            offload_state_dict=False,
            attn_implementation=self.args.attn_implementation,
            trust_remote_code=True,
        )
        # Text-only eval: drop talker if it was materialized.
        disable_talker = getattr(model, "disable_talker", None)
        if callable(disable_talker):
            try:
                disable_talker()
            except Exception as exc:
                logger.warning("disable_talker() failed after load: %s", exc)

        if self.load_adapter:
            from peft import PeftConfig, PeftModel

            logger.info(
                "Loading PEFT LoRA adapter: name=%s path=%s",
                self.args.trained_model_name,
                self.args.adapter_path,
            )
            peft_config = _prepare_generation_peft_config(
                PeftConfig.from_pretrained(self.args.adapter_path),
                model,
            )
            self.model = PeftModel.from_pretrained(
                model,
                self.args.adapter_path,
                config=peft_config,
                low_cpu_mem_usage=True,
            ).eval()
        else:
            self.model = model.eval()
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.args.model_path, trust_remote_code=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.model = None
        self.processor = None
        torch_module = _require_torch()
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
        return False

    def adapter_context(self, *, trained: bool):
        if self.model is None:
            raise RuntimeError("TransformersOmniGenerator is not initialized.")
        return _adapter_context(self.model, trained=trained)

    def generate(self, sample: EvalSample, *, trained: bool) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("TransformersOmniGenerator is not initialized.")

        conversation = build_generation_conversation(sample, self.args)
        use_audio_in_video = bool(self.args.use_audio_in_video)
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = self._process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
        )
        inputs = _move_inputs_to_model(inputs, self.model)

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.args.generation_max_tokens,
            "return_audio": False,
            "thinker_return_dict_in_generate": True,
            "use_audio_in_video": use_audio_in_video,
        }
        if self.args.generation_temperature and self.args.generation_temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = self.args.generation_temperature
            generate_kwargs["top_p"] = self.args.generation_top_p
        else:
            generate_kwargs["do_sample"] = False

        torch_module = _require_torch()
        with torch_module.inference_mode():
            generate_output = self.model.generate(**inputs, **generate_kwargs)

        if isinstance(generate_output, tuple) and not hasattr(generate_output, "sequences"):
            generate_output = generate_output[0]
        sequences = getattr(generate_output, "sequences", None)
        if sequences is None and isinstance(generate_output, dict):
            sequences = generate_output.get("sequences")
        if sequences is None:
            sequences = generate_output
        if isinstance(sequences, str):
            return sequences.strip()
        if isinstance(sequences, Sequence) and sequences and isinstance(sequences[0], str):
            return str(sequences[0]).strip()

        prompt_len = inputs["input_ids"].shape[1]
        decoded = self.processor.batch_decode(
            sequences[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not decoded:
            raise RuntimeError(f"Transformers generate returned no text for sample {sample.uid}.")
        return str(decoded[0]).strip()


def build_generation_client(args: argparse.Namespace, *, load_adapter: bool) -> AbstractContextManager:
    return TransformersOmniGenerator(args, load_adapter=load_adapter)


def run_generation_stage(args: argparse.Namespace, *, role: str, output_path: Path) -> dict[str, Any]:
    trained = role == "trained"
    data_files = resolve_eval_data_files(args)
    samples = read_samples(data_files, args.max_samples)
    if not samples:
        raise ValueError("No evaluation samples were loaded.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = completed_generation_keys(output_path)
    generated = 0
    skipped = 0

    if trained:
        args.adapter_path = resolve_adapter_path(args.adapter_path, args.model_path)

    with build_generation_client(args, load_adapter=trained) as generation_client:
        adapter_context = getattr(generation_client, "adapter_context", None)
        if not callable(adapter_context):
            raise RuntimeError("Generation client does not expose adapter_context().")
        with adapter_context(trained=trained):
            for sample_id, sample in enumerate(samples):
                key = sample_key(sample)
                if key in done:
                    skipped += 1
                    logger.info(
                        "Skipping cached %s generation %d/%d uid=%s", role, sample_id + 1, len(samples), sample.uid
                    )
                    continue

                logger.info(
                    "Generating %s sample %d/%d uid=%s modality=%s",
                    role,
                    sample_id + 1,
                    len(samples),
                    sample.uid,
                    sample.modality,
                )
                start = time.perf_counter()
                generated_text = generation_client.generate(sample, trained=trained)
                logger.info(
                    "%s generation uid=%s took %.3fs", role.capitalize(), sample.uid, time.perf_counter() - start
                )
                logger.info("%s text uid=%s:\n%s", role.capitalize(), sample.uid, generated_text)
                write_jsonl_row(
                    output_path,
                    {
                        "data_file": sample.data_file,
                        "index": sample.index,
                        "uid": sample.uid,
                        "modality": sample.modality,
                        "prompt_text": sample.prompt_text,
                        "media": sample.media,
                        "model_role": role,
                        "generated_text": generated_text,
                    },
                )
                done.add(key)
                generated += 1

    return {
        "stage": role,
        "generated": generated,
        "skipped": skipped,
        "output_jsonl": str(output_path),
    }


def run_judge_stage(args: argparse.Namespace, *, reference_path: Path, trained_path: Path) -> dict[str, Any]:
    rng = random.Random(args.seed)
    data_files = resolve_eval_data_files(args)
    samples = read_samples(data_files, args.max_samples)
    if not samples:
        raise ValueError("No evaluation samples were loaded.")

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    judge_rows = {row_key(row): row for row in read_jsonl_rows(output_path)}
    judged = set(judge_rows)
    reference_cache = load_generation_cache(reference_path, "reference")
    trained_cache = load_generation_cache(trained_path, "trained")
    sample_keys = {sample_key(sample) for sample in samples}
    if sample_keys.issubset(judged):
        ordered_rows = [judge_rows[sample_key(sample)] for sample in samples if sample_key(sample) in judge_rows]
        rewrite_jsonl_rows(output_path, ordered_rows)
        overall_summary, modality_summary = summarize_judge_rows(ordered_rows)
        return {
            "stage": "judge",
            "judged": 0,
            "skipped": len(samples),
            "overall": overall_summary,
            "by_modality": modality_summary,
            "output_jsonl": str(output_path),
        }

    judge_command = (
        _format_command(
            args.judge_server_command,
            model=args.judge_model,
            host=args.judge_server_host,
            port=args.judge_server_port,
            args=args,
        )
        if args.launch_judge_server
        else None
    )

    overall = SummaryStats()
    by_modality: dict[str, SummaryStats] = defaultdict(SummaryStats)
    judged_count = 0
    skipped = 0

    with ManagedServer(
        command=judge_command,
        router_address=args.judge_router_address,
        timeout_s=args.server_timeout_s,
        name="judge",
    ):
        for sample_id, sample in enumerate(samples):
            key = sample_key(sample)
            if key in judged:
                skipped += 1
                logger.info("Skipping cached judge result %d/%d uid=%s", sample_id + 1, len(samples), sample.uid)
                continue
            if key not in reference_cache:
                raise KeyError(f"Missing reference generation for sample key={key} in {reference_path}")
            if key not in trained_cache:
                raise KeyError(f"Missing trained generation for sample key={key} in {trained_path}")

            logger.info(
                "Judging sample %d/%d uid=%s modality=%s", sample_id + 1, len(samples), sample.uid, sample.modality
            )
            reference_text = str(reference_cache[key]["generated_text"])
            trained_text = str(trained_cache[key]["generated_text"])
            logger.info("Reference text uid=%s:\n%s", sample.uid, reference_text)
            logger.info("Trained text uid=%s:\n%s", sample.uid, trained_text)
            judge_result = judge_pair(
                sample=sample,
                reference_text=reference_text,
                trained_text=trained_text,
                router_address=args.judge_router_address,
                model_name=args.judge_model,
                max_tokens=args.judge_max_tokens,
                retry_max_tokens=args.judge_retry_max_tokens,
                temperature=args.judge_temperature,
                rng=rng,
            )
            logger.info("Judge result uid=%s: %s", sample.uid, judge_result)
            overall.update(judge_result)
            by_modality[sample.modality].update(judge_result)
            judged_count += 1

            row = {
                "data_file": sample.data_file,
                "index": sample.index,
                "uid": sample.uid,
                "modality": sample.modality,
                "prompt_text": sample.prompt_text,
                "media": sample.media,
                "reference_text": reference_text,
                "trained_text": trained_text,
                "reference_score": judge_result.reference_score,
                "trained_score": judge_result.trained_score,
                "reference_dimension_scores": judge_result.reference_dimension_scores,
                "trained_dimension_scores": judge_result.trained_dimension_scores,
                "winner": judge_result.winner,
                "rationale": judge_result.rationale,
                "judge_raw_response": judge_result.raw_response,
            }
            write_jsonl_row(output_path, row)
            judge_rows[key] = row
            judged.add(key)

    ordered_rows = [judge_rows[sample_key(sample)] for sample in samples if sample_key(sample) in judge_rows]
    rewrite_jsonl_rows(output_path, ordered_rows)
    overall_summary, modality_summary = summarize_judge_rows(ordered_rows)

    return {
        "stage": "judge",
        "judged": judged_count,
        "skipped": skipped,
        "overall": overall_summary,
        "by_modality": modality_summary,
        "output_jsonl": str(output_path),
    }


def validate_args_for_uncached_run(args: argparse.Namespace) -> None:
    if not resolve_eval_data_files(args):
        raise ValueError(
            "--data-files or both --data-dir and --modalities are required "
            "when no cached judge jsonl files are available."
        )
    if args.stage in {"reference", "trained", "all"} and not args.model_path:
        raise ValueError(
            f"--model-path is required for stage={args.stage!r} when no cached judge jsonl files are available."
        )
    if args.stage in {"trained", "all"} and not args.adapter_path:
        raise ValueError(
            f"--adapter-path is required for stage={args.stage!r} when no cached judge jsonl files are available."
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    reference_path = generation_jsonl_path(args, "reference")
    trained_path = generation_jsonl_path(args, "trained")
    cached_judge_paths = discover_cached_judge_jsonls(Path(args.output_jsonl))
    if cached_judge_paths:
        logger.info(
            "Found %d cached judge jsonl file(s) under %s; skipping all stages.",
            len(cached_judge_paths),
            Path(args.output_jsonl),
        )
        return summarize_cached_judge_results(cached_judge_paths)

    validate_args_for_uncached_run(args)

    if args.stage == "reference":
        return run_generation_stage(args, role="reference", output_path=reference_path)
    if args.stage == "trained":
        return run_generation_stage(args, role="trained", output_path=trained_path)
    if args.stage == "judge":
        return run_judge_stage(args, reference_path=reference_path, trained_path=trained_path)

    summaries = {
        "reference": run_generation_stage(args, role="reference", output_path=reference_path),
        "trained": run_generation_stage(args, role="trained", output_path=trained_path),
        "judge": run_judge_stage(args, reference_path=reference_path, trained_path=trained_path),
    }
    return {"stage": "all", "stages": summaries}


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    try:
        summary = run(args)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        raise
    summary_path = summary_json_path(args)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Saved summary JSON to %s", summary_path)
    if "table" in summary:
        print(summary["table"])
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
