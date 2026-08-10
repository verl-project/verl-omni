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

import pandas as pd
import torch

logger = logging.getLogger("qwen3_omni_minicpm_judge")

DEFAULT_JUDGE_SERVER_COMMAND = (
    "vllm serve {model} --host {host} --port {port} --dtype bfloat16 --trust-remote-code --enforce-eager"
)
DEFAULT_JUDGE_MODEL = "openbmb/MiniCPM-o-4_5"
DEFAULT_TRAINED_MODEL_NAME = "qwen3-omni-trained"
MEDIA_KEYS = (("image", "images"), ("video", "videos"), ("audio", "audios"))
MEDIA_CONTENT_TYPES = {
    "image": ("image_url", "image_url"),
    "video": ("video_url", "video_url"),
    "audio": ("audio_url", "audio_url"),
}
JUDGE_DIMENSIONS = ("fluency", "relevance", "accuracy", "reasoning_quality", "safety")
_MODALITY_PREFIX_RE = re.compile(r"^<(image|video|audio)>\s*", flags=re.IGNORECASE)


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
        "--data-files", nargs="+", required=True, help="Held-out Omni-Preference parquet/json/jsonl files."
    )
    parser.add_argument(
        "--output-jsonl", required=True, help="Where per-sample generation and judge results are written."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum number of samples to evaluate from each input data file.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for A/B judge order.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    parser.add_argument("--model-path", required=True, help="Base Qwen3-Omni model path or HF repo id.")
    parser.add_argument(
        "--adapter-path",
        required=True,
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
        default="meta-offload-non-thinker",
        choices=["auto", "meta-offload-non-thinker", "cuda", "cpu"],
        help="How to place Qwen3-Omni modules. "
        "'meta-offload-non-thinker' keeps talker/code2wav on the meta device (no VRAM) and "
        "shards thinker with accelerate auto placement; "
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
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--server-timeout-s", type=float, default=900.0)
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


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
        return bool(pd.isna(value))
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
    samples: list[EvalSample] = []
    for data_file in data_files:
        path = Path(data_file)
        if path.suffix == ".jsonl":
            dataframe = pd.read_json(path, lines=True)
        elif path.suffix == ".json":
            dataframe = pd.read_json(path)
        else:
            dataframe = pd.read_parquet(path)
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
        "Return only valid JSON with this schema:\n"
        "{\n"
        '  "A": {"overall_score": 0, "fluency": 0, "relevance": 0, "accuracy": 0, '
        '"reasoning_quality": 0, "safety": 0},\n'
        '  "B": {"overall_score": 0, "fluency": 0, "relevance": 0, "accuracy": 0, '
        '"reasoning_quality": 0, "safety": 0},\n'
        '  "winner": "A|B|TIE",\n'
        '  "rationale": "short explanation"\n'
        "}"
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


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
        raw_winner = "TIE"

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
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
    }
    response_text = _message_text(_post_chat_completion(router_address, payload))
    return parse_judge_response(response_text, label_to_model)


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _export_fsdp_lora_adapter(
    *,
    input_dir: Path,
    output_dir: Path,
    base_model_name_or_path: str | None,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    module_path = repo_root / "verl_omni" / "utils" / "fsdp_utils.py"
    spec = importlib.util.spec_from_file_location("_verl_omni_fsdp_utils_for_judge", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load FSDP utils from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

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


def build_generation_conversation(sample: EvalSample) -> list[dict[str, Any]]:
    """Build a Qwen3-Omni chat conversation for transformers + process_mm_info."""
    content: list[dict[str, Any]] = []
    for modality, media_key in MEDIA_KEYS:
        for media_path in sample.media[media_key]:
            content.append({"type": modality, modality: media_path})
    question = _strip_modality_placeholders(sample.prompt_text)
    if not question and not content:
        raise ValueError(f"Sample {sample.uid} has neither text nor media for generation.")
    if question:
        content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def _resolve_torch_dtype(dtype: str) -> str | torch.dtype:
    if dtype == "auto":
        return "auto"
    return getattr(torch, dtype)


def _resolve_model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _move_inputs_to_model(inputs: Any, model: Any) -> Any:
    device = _resolve_model_device(model)
    dtype = getattr(model, "dtype", None)
    moved = inputs.to(device)
    if dtype is None or not hasattr(moved, "items"):
        return moved
    for key, value in list(moved.items()):
        if torch.is_tensor(value) and torch.is_floating_point(value):
            moved[key] = value.to(dtype)
    return moved


META_OFFLOAD_MODULE_PREFIXES = ("talker", "code2wav")


def _concrete_dtype_for_memory_planning(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "auto":
        return torch.bfloat16
    return getattr(torch, str(dtype))


def build_omni_device_map(
    model_path: str,
    *,
    dtype: str | torch.dtype,
    strategy: str,
) -> str | dict[str, Any]:
    """Resolve a transformers/accelerate device_map for Qwen3-Omni.

    ``meta-offload-non-thinker`` keeps talker/code2wav on the meta device so their
    weights are never materialized into VRAM, then auto-shards the remaining
    modules (primarily thinker) across visible GPUs. Loading still uses Accelerate
    meta-init via ``low_cpu_mem_usage=True``.
    """
    if strategy == "auto":
        return "auto"
    if strategy in {"cuda", "cpu"}:
        return strategy

    if strategy != "meta-offload-non-thinker":
        raise ValueError(f"Unsupported device-map strategy: {strategy}")

    from accelerate import init_empty_weights
    from accelerate.utils import get_balanced_memory, infer_auto_device_map
    from transformers import AutoConfig, Qwen3OmniMoeForConditionalGeneration

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    plan_dtype = _concrete_dtype_for_memory_planning(dtype)
    with init_empty_weights():
        # Qwen3-Omni exposes `_from_config` (PreTrainedModel), not public `from_config`.
        empty = Qwen3OmniMoeForConditionalGeneration._from_config(config)

    offload_prefixes = [name for name in META_OFFLOAD_MODULE_PREFIXES if hasattr(empty, name)]
    # Drop non-thinker modules before memory planning so thinker can use that VRAM.
    for prefix in offload_prefixes:
        try:
            delattr(empty, prefix)
        except Exception:
            setattr(empty, prefix, None)

    no_split = list(getattr(empty, "_no_split_modules", None) or [])
    max_memory = get_balanced_memory(empty, dtype=plan_dtype, no_split_module_classes=no_split)
    device_map = infer_auto_device_map(
        empty,
        max_memory=max_memory,
        no_split_module_classes=no_split,
        dtype=plan_dtype,
    )
    for prefix in offload_prefixes:
        device_map[prefix] = "meta"

    logger.info(
        "Built meta-offload device_map: offload=%s placed_modules=%d",
        offload_prefixes,
        len(device_map),
    )
    return device_map


def _adapter_context(model: Any, *, trained: bool):
    """Enable LoRA for trained generation; disable it for the reference baseline."""
    if trained:
        enable_adapters = getattr(model, "enable_adapters", None)
        if callable(enable_adapters):
            enable_adapters()
        return nullcontext()

    disable_adapter = getattr(model, "disable_adapter", None)
    if callable(disable_adapter):
        return disable_adapter()

    disable_adapters = getattr(model, "disable_adapters", None)
    enable_adapters = getattr(model, "enable_adapters", None)
    if callable(disable_adapters) and callable(enable_adapters):
        disable_adapters()

        class _ReenableAdapters:
            def __enter__(self):
                return model

            def __exit__(self, exc_type, exc, tb) -> bool:
                enable_adapters()
                return False

        return _ReenableAdapters()

    raise RuntimeError("Loaded model does not expose PEFT adapter enable/disable APIs.")


class TransformersOmniGenerator(AbstractContextManager):
    """Qwen3-Omni generation via transformers ``generate`` with a PEFT LoRA adapter."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model = None
        self.processor = None

    def __enter__(self):
        from peft import PeftModel
        from qwen_omni_utils import process_mm_info
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        # Install PEFT remapping patches used during verl Qwen3-Omni LoRA training,
        # then unfuse MoE experts so adapter keys match the training module layout.
        from verl_omni.models.transformers.qwen3_omni_thinker_experts import (
            unfuse_qwen3_omni_thinker_experts,
        )

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

        converted = unfuse_qwen3_omni_thinker_experts(model)
        logger.info("Unfused %d Qwen3-Omni thinker expert module(s) before LoRA load", converted)

        logger.info(
            "Loading PEFT LoRA adapter: name=%s path=%s",
            self.args.trained_model_name,
            self.args.adapter_path,
        )
        self.model = PeftModel.from_pretrained(
            model,
            self.args.adapter_path,
            low_cpu_mem_usage=True,
        ).eval()
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.args.model_path, trust_remote_code=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False

    def generate(self, sample: EvalSample, *, trained: bool) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("TransformersOmniGenerator is not initialized.")

        conversation = build_generation_conversation(sample)
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

        with torch.inference_mode(), _adapter_context(self.model, trained=trained):
            text_ids, _audio = self.model.generate(**inputs, **generate_kwargs)

        sequences = getattr(text_ids, "sequences", text_ids)
        prompt_len = inputs["input_ids"].shape[1]
        decoded = self.processor.batch_decode(
            sequences[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not decoded:
            raise RuntimeError(f"Transformers generate returned no text for sample {sample.uid}.")
        return str(decoded[0]).strip()


def build_generation_client(args: argparse.Namespace) -> AbstractContextManager:
    return TransformersOmniGenerator(args)


def run(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    samples = read_samples(args.data_files, args.max_samples)
    if not samples:
        raise ValueError("No evaluation samples were loaded.")

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    args.adapter_path = resolve_adapter_path(args.adapter_path, args.model_path)
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

    with (
        build_generation_client(args) as generation_client,
        ManagedServer(
            command=judge_command,
            router_address=args.judge_router_address,
            timeout_s=args.server_timeout_s,
            name="judge",
        ),
    ):
        for sample_id, sample in enumerate(samples):
            logger.info(
                "Evaluating sample %d/%d uid=%s modality=%s", sample_id + 1, len(samples), sample.uid, sample.modality
            )
            reference_text = generation_client.generate(sample, trained=False)
            trained_text = generation_client.generate(sample, trained=True)
            judge_result = judge_pair(
                sample=sample,
                reference_text=reference_text,
                trained_text=trained_text,
                router_address=args.judge_router_address,
                model_name=args.judge_model,
                max_tokens=args.judge_max_tokens,
                temperature=args.judge_temperature,
                rng=rng,
            )
            overall.update(judge_result)
            by_modality[sample.modality].update(judge_result)

            write_jsonl_row(
                output_path,
                {
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
                },
            )

    return {
        "overall": overall.to_dict(),
        "by_modality": {modality: stats.to_dict() for modality, stats in sorted(by_modality.items())},
        "output_jsonl": str(output_path),
    }


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
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
