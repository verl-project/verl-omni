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

1. a Qwen3-Omni / vLLM-Omni generation backend. By default this uses a real
   ``AsyncOmni`` engine and passes ``LoRARequest`` directly to
   ``AsyncOmni.generate()``.
2. a MiniCPM-o OpenAI-compatible judge endpoint, by default
   ``openbmb/MiniCPM-o-4_5``.

It reads held-out Omni-Preference rows, generates one answer with the reference
model and one with the trained LoRA for each prompt, then asks MiniCPM-o to
score and compare the two answers under the original multimodal input.
"""

from __future__ import annotations

import argparse
import asyncio
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
import tempfile
import time
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

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

    parser.add_argument(
        "--model-path", required=True, help="Base Qwen3-Omni model path or HF repo id for server launch."
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="PEFT LoRA adapter directory, or a verl FSDP / global_step checkpoint. "
        "Non-PEFT checkpoints are exported via export_fsdp_lora_adapter, then used "
        "as the trained LoRARequest.",
    )
    parser.add_argument("--trained-model-name", default=DEFAULT_TRAINED_MODEL_NAME)
    parser.add_argument(
        "--deploy-config",
        required=True,
        help="Real vLLM-Omni deploy/stage YAML used by AsyncOmni.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Override stage-0 tensor_parallel_size in --deploy-config before AsyncOmni starts. "
        "runtime.devices is always derived from CUDA_VISIBLE_DEVICES as 0..(N-1). "
        "If omitted, TP defaults to the CUDA_VISIBLE_DEVICES count when that env is set.",
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


def _parse_cuda_visible_devices(value: str | None = None) -> list[str] | None:
    raw = os.environ["CUDA_VISIBLE_DEVICES"] if value is None and "CUDA_VISIBLE_DEVICES" in os.environ else value
    if raw is None:
        return None
    parts = [part.strip() for part in str(raw).split(",")]
    devices = [part for part in parts if part]
    if not devices:
        raise ValueError(f"CUDA_VISIBLE_DEVICES must list at least one GPU id, got: {raw!r}")
    return devices


def _devices_string_for_count(count: int) -> str:
    if count < 1:
        raise ValueError(f"device count must be >= 1, got: {count}")
    return ",".join(str(idx) for idx in range(count))


def resolve_parallelism_overrides(
    *,
    tensor_parallel_size: int | None,
    cuda_visible_devices: str | None = None,
) -> tuple[int | None, str | None]:
    """Resolve TP/devices from optional CLI TP and CUDA_VISIBLE_DEVICES.

    ``runtime.devices`` always uses remapped ids ``0..(N-1)`` matching the
    visible CUDA device count. Returns ``(None, None)`` when neither an
    explicit TP nor ``CUDA_VISIBLE_DEVICES`` is available.
    """
    visible = _parse_cuda_visible_devices(cuda_visible_devices)
    if tensor_parallel_size is None and visible is None:
        return None, None

    if visible is not None:
        visible_count = len(visible)
        if tensor_parallel_size is None:
            tensor_parallel_size = visible_count
        elif tensor_parallel_size != visible_count:
            raise ValueError(
                f"--tensor-parallel-size ({tensor_parallel_size}) must match "
                f"CUDA_VISIBLE_DEVICES count ({visible_count}): {','.join(visible)}"
            )
    assert tensor_parallel_size is not None
    if tensor_parallel_size < 1:
        raise ValueError(f"--tensor-parallel-size must be >= 1, got: {tensor_parallel_size}")
    return tensor_parallel_size, _devices_string_for_count(tensor_parallel_size)


def _stage_entries(config_data: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(config_data.get("stage_args"), list) and config_data["stage_args"]:
        return config_data["stage_args"]
    if isinstance(config_data.get("stages"), list) and config_data["stages"]:
        return config_data["stages"]
    raise ValueError("Deploy config must contain a non-empty stage_args or stages list.")


def apply_deploy_config_overrides(
    deploy_config: str | Path,
    *,
    tensor_parallel_size: int | None = None,
) -> tuple[Path, Path | None]:
    """Rewrite stage-0 TP/devices into a temp YAML when overrides are set.

    Devices are taken from ``CUDA_VISIBLE_DEVICES`` (remapped to ``0..(N-1)``).
    Returns ``(config_path_to_use, temp_path_or_none)``. Callers should delete
    ``temp_path_or_none`` after the engine shuts down.
    """
    source = Path(os.path.expanduser(str(deploy_config))).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Deploy config YAML does not exist: {source}")

    tp, device_ids = resolve_parallelism_overrides(tensor_parallel_size=tensor_parallel_size)
    if tp is None:
        return source, None

    with source.open("r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)
    if not isinstance(config_data, dict):
        raise ValueError(f"Deploy config must be a YAML mapping: {source}")

    stage0 = _stage_entries(config_data)[0]
    if not isinstance(stage0, dict):
        raise ValueError("stage-0 entry must be a mapping.")

    runtime = stage0.get("runtime")
    if runtime is None:
        runtime = {}
        stage0["runtime"] = runtime
    if not isinstance(runtime, dict):
        raise ValueError("stage-0 runtime must be a mapping.")
    runtime["devices"] = device_ids

    engine_args = stage0.get("engine_args")
    if engine_args is None:
        engine_args = {}
        stage0["engine_args"] = engine_args
    if not isinstance(engine_args, dict):
        raise ValueError("stage-0 engine_args must be a mapping.")
    engine_args["tensor_parallel_size"] = tp

    fd, temp_name = tempfile.mkstemp(prefix="vlm_as_judge_deploy_", suffix=".yaml")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_data, f, sort_keys=False)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    logger.info(
        "Wrote overridden deploy config: source=%s tp=%s devices=%s cuda_visible_devices=%s temp=%s",
        source,
        tp,
        device_ids,
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        temp_path,
    )
    return temp_path, temp_path


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
    """Return a PEFT LoRA adapter directory for ``LoRARequest``.

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


def _async_omni_output_text(output: Any) -> str:
    request_output = getattr(output, "request_output", output)
    outputs = getattr(request_output, "outputs", None)
    if outputs:
        first_output = outputs[0]
        for attr in ("text", "output_text"):
            text = getattr(first_output, attr, None)
            if text is not None:
                return str(text).strip()
    text = getattr(request_output, "text", None)
    if text is not None:
        return str(text).strip()
    return str(output).strip()


def choose_async_omni_config_kwarg(config_path: str | Path) -> dict[str, str]:
    """Pick AsyncOmni YAML kwarg based on file schema.

    Legacy thinker-only YAMLs use ``stage_args`` and must be passed as
    ``stage_configs_path``. Passing them as ``deploy_config`` makes vLLM-Omni
    fall back to the registry multi-stage pipeline and ignore local TP/devices.
    """
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict) and isinstance(data.get("stage_args"), list):
        return {"stage_configs_path": str(path)}
    return {"deploy_config": str(path)}


class AsyncOmniGenerator(AbstractContextManager):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.engine = None
        self.sampling_params_cls = None
        self.lora_request = None
        self._temp_deploy_config: Path | None = None

    def __enter__(self):
        if not self.args.deploy_config:
            raise ValueError("--deploy-config is required for AsyncOmni generation.")

        deploy_config, self._temp_deploy_config = apply_deploy_config_overrides(
            self.args.deploy_config,
            tensor_parallel_size=getattr(self.args, "tensor_parallel_size", None),
        )

        from vllm import SamplingParams
        from vllm_omni.entrypoints import AsyncOmni
        from vllm_omni.lora.request import LoRARequest

        self.sampling_params_cls = SamplingParams
        config_kwargs = choose_async_omni_config_kwarg(deploy_config)
        logger.info("Starting AsyncOmni with %s", config_kwargs)
        self.engine = AsyncOmni(model=self.args.model_path, **config_kwargs)
        self.lora_request = LoRARequest(self.args.trained_model_name, 1, self.args.adapter_path)

        stage_client = self.engine.engine.stage_clients[0]
        loaded = asyncio.run(stage_client.add_lora_async(self.lora_request))
        if not loaded:
            raise RuntimeError(f"Failed to load LoRA adapter: {self.args.adapter_path}")
        logger.info(
            "Loaded AsyncOmni LoRA adapter: name=%s path=%s", self.args.trained_model_name, self.args.adapter_path
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.engine is not None:
            self.engine.shutdown()
        if self._temp_deploy_config is not None:
            self._temp_deploy_config.unlink(missing_ok=True)
            self._temp_deploy_config = None
        return False

    def generate(self, sample: EvalSample, *, trained: bool) -> str:
        return asyncio.run(self._generate(sample, self.lora_request if trained else None))

    async def _generate(self, sample: EvalSample, lora_request) -> str:
        if self.engine is None or self.sampling_params_cls is None:
            raise RuntimeError("AsyncOmniGenerator is not initialized.")

        sampling_params = self.sampling_params_cls(
            max_tokens=self.args.generation_max_tokens,
            temperature=self.args.generation_temperature,
            top_p=self.args.generation_top_p,
        )
        final_output = None
        async for output in self.engine.generate(
            sample.prompt_text,
            sampling_params_list=[sampling_params],
            output_modalities=["text"],
            lora_request=lora_request,
        ):
            final_output = output
        if final_output is None:
            raise RuntimeError(f"AsyncOmni returned no output for sample {sample.uid}.")
        return _async_omni_output_text(final_output)


def build_generation_client(args: argparse.Namespace) -> AbstractContextManager:
    return AsyncOmniGenerator(args)


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
