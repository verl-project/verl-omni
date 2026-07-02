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

"""
E2E test for Qwen-Image-Edit-Plus vLLMOmniHttpServer generate flow.

Usage:
    pytest tests/workers/rollout/rollout_vllm/test_vllm_omni_qwen_image_edit_generate.py -v -s
    python tests/workers/rollout/rollout_vllm/test_vllm_omni_qwen_image_edit_generate.py
"""

import os
from pathlib import Path
from uuid import uuid4

import pytest
import ray
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoProcessor
from verl.utils.chat_template import apply_chat_template
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import RolloutMode

from verl_omni.workers.rollout.replica import DiffusionOutput
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer

MODEL_PATH = Path(os.path.expanduser("~/models/tiny-random/qwen-image-edit-plus"))


# ---------------------------------------------------------------------
#                Test Helper Functions & Fixtures
# ---------------------------------------------------------------------


def _ensure_tiny_processor_config() -> None:
    """Add minimal processor metadata missing from the tiny-random test model."""
    config_path = MODEL_PATH / "processor" / "config.json"
    if config_path.parent.is_dir() and not config_path.exists():
        config_path.write_text('{"model_type":"qwen2_vl"}\n')


def _make_image(color: tuple[int, int, int]) -> Image.Image:
    """Create a deterministic condition image for image-edit generation."""
    return Image.new("RGB", (256, 256), color=color)


def _tokenize_prompt(text: str, image: Image.Image) -> list[int]:
    """Tokenize a multimodal edit prompt into valid token IDs for the model."""
    _ensure_tiny_processor_config()
    processor = AutoProcessor.from_pretrained(os.path.join(MODEL_PATH, "processor"), trust_remote_code=True)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }
    ]
    raw_prompt = apply_chat_template(processor, messages, add_generation_prompt=True, tokenize=False)
    model_inputs = processor(text=[raw_prompt], images=[image], return_tensors="pt", do_sample_frames=False)
    return normalize_token_ids(model_inputs.pop("input_ids"))


@pytest.fixture
def init_server():
    """Create and launch a vLLMOmniHttpServer Ray actor with Qwen-Image-Edit-Plus."""
    _ensure_tiny_processor_config()
    model_path = MODEL_PATH

    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
            }
        },
        ignore_reinit_error=True,
    )

    rollout_cfg = OmegaConf.create(
        {
            "_target_": "verl_omni.workers.config.diffusion.DiffusionRolloutConfig",
            "name": "vllm_omni",
            "mode": "async",
            "tensor_model_parallel_size": 1,
            "data_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "gpu_memory_utilization": float(os.getenv("TEST_GPU_MEMORY_UTILIZATION", "0.8")),
            "max_num_batched_tokens": 8192,
            "max_num_seqs": 256,
            "max_model_len": 1024,
            "dtype": "bfloat16",
            "load_format": "safetensors",
            "enforce_eager": True,
            "enable_chunked_prefill": False,
            "enable_prefix_caching": False,
            "enable_sleep_mode": False,
            "free_cache_engine": True,
            "disable_log_stats": True,
            "n": 1,
            "pipeline": {
                "_target_": "verl_omni.workers.config.diffusion.rollout.DiffusionPipelineConfig",
                "height": 512,
                "width": 512,
                "num_inference_steps": 4,
                "true_cfg_scale": 4.0,
                "max_sequence_length": 512,
            },
        }
    )

    model_cfg = OmegaConf.create(
        {
            "_target_": "verl_omni.workers.config.diffusion.DiffusionModelConfig",
            "path": model_path,
            "tokenizer_path": os.path.join(model_path, "tokenizer"),
            "trust_remote_code": True,
            "load_tokenizer": True,
            "algorithm": "flow_grpo",
        }
    )

    ServerCls = ray.remote(vLLMOmniHttpServer)
    server = ServerCls.options(
        runtime_env={
            "env_vars": {
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1",
                "NCCL_CUMEM_ENABLE": "0",
            }
        },
        max_concurrency=10,
    ).remote(
        config=rollout_cfg,
        model_config=model_cfg,
        rollout_mode=RolloutMode.STANDALONE,
        workers=[],
        replica_rank=0,
        node_rank=0,
        gpus_per_node=1,
        nnodes=1,
        cuda_visible_devices=os.getenv("TEST_CUDA_VISIBLE_DEVICES", "0"),
    )

    ray.get(server.launch_server.remote())

    yield server

    ray.shutdown()


def test_generate(init_server):
    """Generate with a condition image, negative prompt, logprobs, and edit latent fields."""
    server = init_server
    image = _make_image((120, 40, 200))
    rid = f"qwen_image_edit_{uuid4().hex[:8]}"

    output = ray.get(
        server.generate.remote(
            prompt_ids=_tokenize_prompt("Change the background to blue.", image),
            negative_prompt_ids=_tokenize_prompt(" ", image),
            image_data=[image],
            sampling_params={
                "num_inference_steps": 4,
                "true_cfg_scale": 4.0,
                "height": 512,
                "width": 512,
                "max_sequence_length": 512,
                "logprobs": True,
                "noise_level": 1.0,
                "sde_type": "sde",
                "sde_window_size": 2,
                "sde_window_range": [0, 4],
            },
            request_id=rid,
        ),
        timeout=600,
    )

    assert isinstance(output, DiffusionOutput), "expected DiffusionOutput"
    assert len(output.diffusion_output) == 3, "expected 3 channels (CHW)"
    h, w = len(output.diffusion_output[0]), len(output.diffusion_output[0][0])
    assert h > 0 and w > 0, "image dimensions must be positive"
    assert output.stop_reason in ("completed", "aborted", None), f"unexpected stop_reason: {output.stop_reason}"
    assert 0.0 <= output.diffusion_output[0][0][0] <= 1.0, "pixel values must be in [0, 1]"

    lp = output.log_probs
    assert lp is not None, "log_probs should be present when logprobs=True"
    if isinstance(lp, torch.Tensor):
        assert lp.numel() > 0
    else:
        assert len(lp) > 0

    assert output.extra_fields["image_latents"] is not None
    assert output.extra_fields["img_shapes"] is not None
    assert len(output.extra_fields["img_shapes"]) == 2

    print("Qwen-Image-Edit request returned valid DiffusionOutput")
