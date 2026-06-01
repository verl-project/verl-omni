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

"""Generate offline DPO triples from one prompt file using a frozen reference pipeline.

The output schema is one logical preference pair per row:

    {prompt, negative_prompt, img_win, img_lose, win_score, lose_score,
     img_win_latents, img_lose_latents, prompt_embeds, ...}

Training expands each row to adjacent ``win, lose`` samples and consumes the
precomputed SD3 latents and text embeddings directly.
"""

import argparse
import asyncio
import importlib.util
import io
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

try:
    import torch_npu
except ImportError:
    torch_npu = None

DEFAULT_SYSTEM_PROMPT = "You are a helpful image generation assistant."
DEFAULT_REWARD_SERVER_COMMAND = "vllm serve {model} --host {host} --port {port} --dtype bfloat16 --enforce-eager"
# Due to memory size differences between NPU and GPU devices, we adjust
# --gpu-memory-utilization and --max-model-len for NPU backends.
DEFAULT_REWARD_SERVER_COMMAND_NPU = "vllm serve {model} --host {host} --port {port} --dtype bfloat16 --gpu-memory-utilization 0.85 --max-model-len 40000"


def _read_prompts_from_txt(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _prompt_to_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        user_parts = []
        for message in prompt:
            if isinstance(message, dict) and message.get("role") == "user":
                user_parts.append(str(message.get("content", "")))
        if user_parts:
            return "\n".join(user_parts)
    return "" if prompt is None else str(prompt)


def _read_prompts(path: Path, prompt_key: str) -> list[str]:
    if path.suffix == ".txt":
        return _read_prompts_from_txt(path)
    if path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif path.suffix == ".json":
        df = pd.read_json(path)
    else:
        df = pd.read_parquet(path)
    return [_prompt_to_text(prompt) for prompt in df[prompt_key].tolist()]


def _build_messages(prompt: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def _load_reward_fn(path: str | None, name: str | None):
    if path is None or name is None:
        return None
    module_path = Path(path)
    if module_path.exists():
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load reward function module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module_name = path[:-3].replace("/", ".") if path.endswith(".py") else path
        module = importlib.import_module(module_name)
    return getattr(module, name)


async def _score_image(reward_fn, image: Image.Image, prompt: str, args: argparse.Namespace) -> float:
    if reward_fn is None:
        return 0.0
    image_array = np.asarray(image).astype("float32") / 255.0
    result = reward_fn(
        data_source=args.data_source,
        solution_image=image_array,
        ground_truth=prompt,
        extra_info={"raw_prompt": prompt, "prompt": prompt},
        reward_router_address=args.reward_router_address,
        model_name=args.reward_model_name,
    )
    if asyncio.iscoroutine(result):
        result = await result
    if isinstance(result, dict):
        return float(result.get("score", 0.0))
    return float(result)


def _make_generator(seed: int, device: str) -> torch.Generator:
    generator_device = device if device != "cpu" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    return buffer.getvalue()


def _encode_image_latent(pipe, image: Image.Image, args: argparse.Namespace) -> torch.Tensor:
    pixel_values = pipe.image_processor.preprocess([image], height=args.height, width=args.width)
    pixel_values = pixel_values.to(device=args.device, dtype=pipe.vae.dtype)
    with torch.no_grad():
        latents = pipe.vae.encode(pixel_values).latent_dist.sample()
    scaling_factor = getattr(pipe.vae.config, "scaling_factor", 1.0)
    shift_factor = getattr(pipe.vae.config, "shift_factor", 0.0)
    latents = (latents - shift_factor) * scaling_factor
    return latents[0].detach().cpu()


def _encode_prompt_tensors(pipe, prompt: str, negative_prompt: str, args: argparse.Namespace) -> dict[str, list | None]:
    do_cfg = args.guidance_scale is not None and args.guidance_scale > 1.0
    with torch.no_grad():
        encoded = pipe.encode_prompt(
            prompt=[prompt],
            prompt_2=None,
            prompt_3=None,
            device=args.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=[negative_prompt] if do_cfg else None,
            negative_prompt_2=None,
            negative_prompt_3=None,
            max_sequence_length=args.max_sequence_length,
        )

    if len(encoded) == 4:
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = encoded
    elif len(encoded) == 2:
        prompt_embeds, pooled_prompt_embeds = encoded
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None
    else:
        raise ValueError(f"Unexpected SD3 encode_prompt output length: {len(encoded)}")

    prompt_embeds_mask = torch.ones(
        prompt_embeds.shape[0],
        prompt_embeds.shape[1],
        dtype=torch.int32,
        device=prompt_embeds.device,
    )
    result = {
        "prompt_embeds": _tensor_to_bytes(prompt_embeds[0]),
        "prompt_embeds_mask": _tensor_to_bytes(prompt_embeds_mask[0]),
        "pooled_prompt_embeds": _tensor_to_bytes(pooled_prompt_embeds[0]),
        "negative_prompt_embeds": None,
        "negative_prompt_embeds_mask": None,
        "negative_pooled_prompt_embeds": None,
    }
    if negative_prompt_embeds is not None:
        negative_prompt_embeds_mask = torch.ones(
            negative_prompt_embeds.shape[0],
            negative_prompt_embeds.shape[1],
            dtype=torch.int32,
            device=negative_prompt_embeds.device,
        )
        result["negative_prompt_embeds"] = _tensor_to_bytes(negative_prompt_embeds[0])
        result["negative_prompt_embeds_mask"] = _tensor_to_bytes(negative_prompt_embeds_mask[0])
        result["negative_pooled_prompt_embeds"] = _tensor_to_bytes(negative_pooled_prompt_embeds[0])
    return result


def _router_url(host: str, port: int, path: str) -> str:
    return f"http://{host}:{port}{path}"


def _wait_for_reward_server(host: str, port: int, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    url = _router_url(host, port, "/v1/models")
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 500:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(5)
    raise TimeoutError(f"Reward server did not become ready at {url} within {timeout_s}s: {last_error}")


class _ChunkedParquetWriter:
    """Write parquet rows incrementally to bound peak memory usage."""

    def __init__(self, path: Path, flush_every: int):
        self.path = path
        self.flush_every = max(1, flush_every)
        self._chunk: list[dict[str, Any]] = []
        self._writer: pq.ParquetWriter | None = None
        self.row_count = 0

    def __enter__(self) -> "_ChunkedParquetWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write(self, row: dict[str, Any]) -> None:
        self._chunk.append(row)
        self.row_count += 1
        if len(self._chunk) >= self.flush_every:
            self._flush()

    def close(self) -> None:
        if self._chunk:
            self._flush()
        if self._writer is not None:
            self._writer.close()
        elif self.row_count == 0:
            pd.DataFrame().to_parquet(self.path)

    def _flush(self) -> None:
        if not self._chunk:
            return
        table = pa.Table.from_pandas(pd.DataFrame(self._chunk), preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, table.schema)
        self._writer.write_table(table)
        self._chunk.clear()


def _apply_gpu_device_defaults(args: argparse.Namespace) -> None:
    """Prefer reward on the first NPU/GPU and image gen on the second; share device 0 when only one is visible."""
    if torch_npu is not None and torch.npu.is_available():
        n = torch.npu.device_count()
    elif torch.cuda.is_available():
        n = torch.cuda.device_count()
    else:
        n = 0

    if n <= 1:
        args.reward_gpu = 0
        args.image_gpu = 0
    else:
        args.reward_gpu = 0 if args.reward_gpu is None else args.reward_gpu
        args.image_gpu = 1 if args.image_gpu is None else args.image_gpu

    if args.device is None:
        if torch_npu is not None and torch.npu.is_available():
            args.device = f"npu:{args.image_gpu}"
        elif torch.cuda.is_available():
            args.device = f"cuda:{args.image_gpu}"
        else:
            args.device = "cpu"


@contextmanager
def _maybe_launch_reward_server(args: argparse.Namespace):
    if not args.launch_reward_server:
        yield
        return

    if args.reward_model_name is None:
        raise ValueError("--launch_reward_server requires --reward_model_name.")

    is_npu = torch_npu is not None and torch.npu.is_available()
    if is_npu and args.reward_server_command == DEFAULT_REWARD_SERVER_COMMAND:
        server_command = DEFAULT_REWARD_SERVER_COMMAND_NPU
    else:
        server_command = args.reward_server_command
    command = server_command.format(
        model=args.reward_model_name,
        host=args.reward_server_host,
        port=args.reward_server_port,
    )
    env = os.environ.copy()
    env.setdefault("VLLM_USE_DEEP_GEMM", "0")
    if is_npu:
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.reward_gpu)
        env["CUDA_VISIBLE_DEVICES"] = str(args.reward_gpu)
        dev_msg = f"ASCEND_RT_VISIBLE_DEVICES={env.get('ASCEND_RT_VISIBLE_DEVICES')}"
    elif torch.cuda.is_available():
        env["CUDA_VISIBLE_DEVICES"] = str(args.reward_gpu)
        dev_msg = f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}"
    else:
        dev_msg = "CPU Mode"
    print(
        f"Launching reward server (Device index {args.reward_gpu}, "
        f"{dev_msg}): {command}"
    )
    process = subprocess.Popen(shlex.split(command), env=env)
    try:
        _wait_for_reward_server(args.reward_server_host, args.reward_server_port, args.reward_server_startup_timeout)
        print(f"Reward server is ready at {args.reward_router_address}")
        yield
    finally:
        print("Stopping reward server.")
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


async def _generate_split(args: argparse.Namespace, split: str) -> Path:
    from diffusers import StableDiffusion3Pipeline

    input_path = Path(os.path.expanduser(args.input_file))
    output_path = Path(os.path.expanduser(args.output_file))
    prompts = _read_prompts(input_path, args.prompt_key)
    if args.max_samples > 0:
        prompts = prompts[: args.max_samples]

    if args.image_dir is None:
        image_dir = output_path.parent / "images" / output_path.stem
    else:
        image_dir = Path(os.path.expanduser(args.image_dir))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_path, torch_dtype=dtype)
    pipe.to(args.device)
    pipe.set_progress_bar_config(disable=args.disable_progress)
    reward_fn = _load_reward_fn(args.reward_function_path, args.reward_function_name)

    with _ChunkedParquetWriter(output_path, args.parquet_flush_every) as writer:
        for prompt_idx, prompt in enumerate(prompts):
            prompt_tensors = _encode_prompt_tensors(pipe, prompt, args.negative_prompt, args)
            candidates = []
            for sample_idx in range(args.num_images_per_prompt):
                seed = args.seed + prompt_idx * args.num_images_per_prompt + sample_idx
                image = pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    max_sequence_length=args.max_sequence_length,
                    generator=_make_generator(seed, args.device),
                ).images[0]
                image_path = image_dir / f"{prompt_idx:06d}_{sample_idx:02d}.png"
                image.save(image_path)
                score = await _score_image(reward_fn, image, prompt, args)
                candidates.append(
                    {
                        "path": str(image_path),
                        "latents": _tensor_to_bytes(_encode_image_latent(pipe, image, args)),
                        "score": score,
                        "seed": seed,
                    }
                )

            candidates.sort(key=lambda item: item["score"], reverse=True)
            win = candidates[0]
            lose = candidates[-1]
            writer.write(
                {
                    "data_source": args.data_source,
                    "prompt": _build_messages(prompt, args.system_prompt),
                    "negative_prompt": _build_messages(args.negative_prompt, args.system_prompt),
                    "img_win": os.path.relpath(win["path"], output_path.parent),
                    "img_lose": os.path.relpath(lose["path"], output_path.parent),
                    "img_win_latents": win["latents"],
                    "img_lose_latents": lose["latents"],
                    **prompt_tensors,
                    "win_score": win["score"],
                    "lose_score": lose["score"],
                    "reward_model": {"style": "model", "ground_truth": prompt},
                    "extra_info": {
                        "split": split,
                        "index": prompt_idx,
                        "raw_prompt": prompt,
                        "raw_negative_prompt": args.negative_prompt,
                        "num_candidates": len(candidates),
                        "win_seed": win["seed"],
                        "lose_seed": lose["seed"],
                        "candidate_scores": [item["score"] for item in candidates],
                    },
                }
            )

    print(f"Wrote {writer.row_count} offline DPO pairs to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate offline DPO triples with a frozen diffusion model.")
    parser.add_argument("--input_file", required=True, help="Prompt file. Supports .txt, .json, .jsonl and parquet.")
    parser.add_argument("--output_file", required=True, help="Parquet file to write.")
    parser.add_argument("--image_dir", default=None, help="Directory to write generated images.")
    parser.add_argument("--prompt_key", default="prompt", help="Prompt column for parquet/json inputs.")
    parser.add_argument("--model_path", default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--data_source", default="offline_dpo")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--negative_prompt", default=" ")
    parser.add_argument("--num_images_per_prompt", type=int, default=4)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Diffusers device (e.g. npu:1, cuda:1). "
            "Default: npu:0 / cuda:0 on a single visible card; npu:1 / cuda:1 on the second card when two or more are visible."
        ),
    )
    parser.add_argument(
        "--reward-gpu",
        type=int,
        default=None,
        dest="reward_gpu",
        help=(
            "Physical CUDA device index for the vLLM reward server (via CUDA_VISIBLE_DEVICES). "
            "Default: 0, or forced to 0 when only one GPU is visible."
        ),
    )
    parser.add_argument(
        "--image-gpu",
        type=int,
        default=None,
        dest="image_gpu",
        help=(
            "CUDA device index for image generation when --device is not set. "
            "Default: 0 with one visible GPU, else 1 (second GPU)."
        ),
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument(
        "--parquet_flush_every",
        type=int,
        default=32,
        help="Number of DPO rows to buffer in memory before flushing to parquet.",
    )
    parser.add_argument("--reward_function_path", default=None)
    parser.add_argument("--reward_function_name", default=None)
    parser.add_argument("--reward_router_address", default=None)
    parser.add_argument("--reward_model_name", default=None)
    parser.add_argument(
        "--launch_reward_server",
        action="store_true",
        help="Launch an OpenAI-compatible reward server subprocess before scoring.",
    )
    parser.add_argument("--reward_server_host", default="127.0.0.1")
    parser.add_argument("--reward_server_port", type=int, default=8000)
    parser.add_argument(
        "--reward_server_command",
        default=DEFAULT_REWARD_SERVER_COMMAND,
        help=("Command template used with --launch_reward_server. Available placeholders: {model}, {host}, {port}."),
    )
    parser.add_argument("--reward_server_startup_timeout", type=int, default=900)
    parser.add_argument("--disable_progress", action="store_true")
    parser.add_argument("--split", default=None, help="Optional split name stored in extra_info.split.")
    args = parser.parse_args()
    _apply_gpu_device_defaults(args)
    print(f"Image generation device: {args.device}.")

    if args.num_images_per_prompt < 2:
        raise ValueError("--num_images_per_prompt must be at least 2 for DPO pair construction.")
    if (args.reward_function_path is None) != (args.reward_function_name is None):
        raise ValueError("Set both --reward_function_path and --reward_function_name, or neither.")
    if args.launch_reward_server and args.reward_router_address is None:
        args.reward_router_address = f"{args.reward_server_host}:{args.reward_server_port}"
    if args.reward_function_path is not None and args.reward_router_address is None:
        raise ValueError(
            "Reward scoring requires --reward_router_address, or use --launch_reward_server to start one automatically."
        )

    output_path = Path(os.path.expanduser(args.output_file))
    split = args.split or output_path.stem
    with _maybe_launch_reward_server(args):
        asyncio.run(_generate_split(args, split))


if __name__ == "__main__":
    main()
