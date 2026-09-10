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
"""Real-engine GPU check of typed request -> adapter -> named output -> reward/export.

Uses one explicitly selected local checkpoint and a registered adapter. This
unconditional fixture covers text-to-image/video/audio-video; image-edit and
reference-conditioned inputs use their existing end-to-end recipe fixtures.
The process owns and closes only its own AsyncOmni instance, never a Ray cluster.
"""

import argparse
import asyncio
import json
import os
import socket
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer
from vllm_omni.entrypoints import AsyncOmni

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.rollout_request import OmniRolloutRequest
from verl_omni.utils.reward_score.reward_utils import image_tensor_to_pil, visual_reward_frames
from verl_omni.utils.tracking import _export_video
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy import DiffusionStrategy

_SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:"
)


def _request(model, architecture, index, tokenizer_path=None):
    text = f"A red circle and a blue square in soft light, example {index}."
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path or Path(model) / "tokenizer", local_files_only=True, trust_remote_code=True
    )
    if architecture == "QwenImagePipeline":
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    extra = {}
    if architecture == "StableDiffusion3Pipeline":
        t5 = AutoTokenizer.from_pretrained(Path(model) / "tokenizer_3", local_files_only=True)
        extra["extra_prompt_ids"] = {
            "clip": tokenizer.encode(text),
            "t5": t5.encode(text),
        }
    return OmniRolloutRequest.from_generate_kwargs(prompt_ids=tokenizer.encode(text), **extra)


async def check(args):
    """Exercise production lowering and output parsing against the pinned engine."""
    config = SimpleNamespace(
        external_lib=None,
        enable_prompt_embed_cache=False,
        prompt_embed_cache_size=16,
        step_execution=args.step_execution,
    )
    server = SimpleNamespace(
        global_steps=1,
        config=config,
        model_config=SimpleNamespace(architecture=args.architecture, algorithm=args.algorithm),
    )
    strategy = DiffusionStrategy(server)
    engine_args = {
        "model": args.model,
        "model_class_name": args.engine_class_name or args.architecture,
        "trust_remote_code": True,
        "enforce_eager": True,
        "enable_sleep_mode": False,
        "enable_cpu_offload": args.cpu_offload,
        "skip_tokenizer_init": True,
        "max_num_seqs": args.num_requests,
        "diffusion_batch_size": args.num_requests if args.require_request_batch else 1,
        "request_batch_max_wait_ms": 200.0 if args.require_request_batch else 0.0,
        "step_execution": args.step_execution,
        "diffusion_attention_config": {"default": {"backend": "TORCH_SDPA"}},
        "dtype": "bfloat16",
    }
    if args.deploy_config:
        engine_args["deploy_config"] = args.deploy_config
    strategy.prepare_engine_args(engine_args, Namespace())
    cls = VllmOmniPipelineBase.get_class(args.architecture, args.algorithm)
    if args.require_request_batch and not getattr(cls, "supports_request_batch", False):
        raise ValueError(f"{args.architecture}/{args.algorithm} does not advertise request batching")
    requested = list(cls.diffusion_io_spec.artifacts)
    sampling = {
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "frame_rate": 24.0,
        "num_inference_steps": 4,
        "seed": 42,
        "output_type": args.output_type,
        "guidance_scale": 1.0,
        "true_cfg_scale": 1.0,
        "max_sequence_length": 256,
        "noise_level": 0.5,
        "sde_type": "sde",
        "sde_window_size": 2,
        "sde_window_range": [0, 4],
        "logprobs": args.algorithm in ("flow_grpo", "dance_grpo", "mix_grpo", "dual_grpo"),
        "requested_outputs": requested,
        **json.loads(args.extra_json),
    }
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    print("Engine args:", engine_args, flush=True)
    requests = [
        _request(args.model, args.architecture, index, args.tokenizer_path) for index in range(args.num_requests)
    ]
    server.engine = AsyncOmni(**engine_args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:

        async def generate(index):
            prompt, params = strategy.preprocess_input(requests[index], sampling, None)
            raw = await strategy.run_generation(prompt, params, f"artifact-check-{index}", None, 0)
            producer_context = raw.multimodal_output["metadata"]["media_artifacts"]["context"]
            if args.require_request_batch:
                assert args.num_requests > 1 and not args.step_execution
                assert all(f"artifact-check-{i}" in producer_context for i in range(args.num_requests)), (
                    producer_context
                )
            result = strategy.process_output(raw, params, sampling)
            assert set(result.artifacts) == set(requested), (requested, result.artifacts.keys())
            assert all(item.data.device.type == "cpu" for item in result.artifacts.values())
            extra_info = {"media_artifacts": result.artifacts, "preview_artifact": result.preview_artifact}
            frames = visual_reward_frames(result.diffusion_output, extra_info)
            assert frames.dtype == torch.uint8
            preview = result.artifacts[result.preview_artifact]
            if preview.spec.modality == "image":
                image_tensor_to_pil(preview.data).save(output_dir / f"{index}.png")
            else:
                _export_video(
                    preview,
                    str(output_dir / f"{index}.mp4"),
                    audio=result.extra_fields.get("audio"),
                    audio_sample_rate=result.extra_fields.get("audio_sample_rate"),
                )
            return {
                "request_id": raw.request_id,
                "producer_context": producer_context,
                "primary": result.primary_artifact,
                "artifacts": {
                    name: {**asdict(item.spec), "shape": list(item.data.shape), "dtype": str(item.data.dtype)}
                    for name, item in result.artifacts.items()
                },
                "training_fields": {
                    name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in result.extra_fields.items()
                    if isinstance(value, torch.Tensor)
                },
            }

        results = await asyncio.gather(*(generate(index) for index in range(args.num_requests)))
        (output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print("PASS", args.architecture, args.algorithm, "step_execution=", args.step_execution, flush=True)
    finally:
        server.engine.shutdown()


def main():
    """Run one explicitly configured adapter GPU check without downloading checkpoints."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--engine-class-name")
    parser.add_argument("--deploy-config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-type", default="image")
    parser.add_argument("--num-requests", type=int, default=1)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--step-execution", action="store_true")
    parser.add_argument("--require-request-batch", action="store_true")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--extra-json", default="{}")
    args = parser.parse_args()
    asyncio.run(check(args))


if __name__ == "__main__":
    main()
