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
"""Processor/encoder 与 DiT LoRA 双端一致性；需要本地 checkpoint 和 NVIDIA GPU。

运行：QWEN21_MODEL_PATH=/models/Qwen-Image-2.1 python -m pytest -q \
tests/pipelines/test_qwen_image_21_consistency.py
参考图、DiT 输入和非零 LoRA 由 fixture 生成，不评估生成质量或收敛。
"""

import gc
import inspect
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import QwenImage21Pipeline

from verl_omni.pipelines.qwen_image_21_diffusion_nft.diffusers_training_adapter import QwenImage21DiffusionNFT
from verl_omni.pipelines.qwen_image_21_diffusion_nft.prompt import prepare_reference_media, tokenize_prompt
from verl_omni.pipelines.qwen_image_21_diffusion_nft.vllm_omni_rollout_adapter import QwenImage21DiffusionNFTPipeline


@pytest.fixture(scope="module")
def reference_images():
    """提供颜色和尺寸各异的四张固定参考图。"""
    return [
        Image.new("RGB", (96 + 32 * index, 80 + 16 * index), color)
        for index, color in enumerate(((220, 30, 30), (30, 180, 50), (30, 70, 220), (210, 180, 30)))
    ]


@pytest.fixture(scope="module")
def encoder_pipeline():
    """只加载真实 encoder，保持权重固定并在测试结束后释放。"""
    model_path = os.environ.get("QWEN21_MODEL_PATH")
    if not model_path or not Path(model_path).is_dir():
        pytest.fail("Set QWEN21_MODEL_PATH to an existing local Qwen-Image-2.1 checkpoint.")
    root = Path(model_path).resolve()
    processor = Qwen3VLProcessor.from_pretrained(root, subfolder="processor", local_files_only=True)
    assert torch.cuda.is_available(), "The encoder comparison requires one NVIDIA GPU."
    encoder = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            root,
            subfolder="text_encoder",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .eval()
        .requires_grad_(False)
        .to("cuda")
    )
    # 只装配 encode_prompt 所需属性，避免为 encoder 对比额外加载 DiT/VAE。
    pipeline = object.__new__(QwenImage21DiffusionNFTPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cuda")
    pipeline.text_encoder = encoder
    pipeline.processor = processor
    pipeline.vae_scale_factor = 16
    pipeline.sys_prompt = "Comprehend and analyze the provided prompt."
    pipeline.prompt_template_t2i = (
        "<|im_start|>system\n" + pipeline.sys_prompt + "<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )
    pipeline.prompt_template_ti2i = pipeline.prompt_template_t2i.replace(
        "<|im_start|>user\n", "<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>"
    )
    system = [{"role": "system", "content": [{"type": "text", "text": pipeline.sys_prompt}]}]
    pipeline._drop_idx = len(processor.apply_chat_template(system, tokenize=True, return_dict=False)[0])
    pipeline._img_token_id = processor.tokenizer.encode("<|image_pad|>")[0]
    pipeline._max_length = 8192
    try:
        yield pipeline
    finally:
        pipeline.text_encoder = None
        del encoder
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.parametrize("order", [[], [0], [0, 1], [0, 1, 2], [0, 1, 2, 3], [3, 2, 1, 0]])
@torch.no_grad()
def test_token_id_encoder_matches_upstream_text(encoder_pipeline, reference_images, order):
    """单次分词路径与上游文本路径的 encoder 输入及输出必须完全相同。"""
    pipeline = encoder_pipeline
    processor = pipeline.processor
    texts = ["Arrange the objects on a wooden table.", "Create a balanced composition in soft daylight."]
    requests, conditioned = [], []
    agent_processor = Mock(wraps=processor)
    for text in texts:
        media = prepare_reference_media({"images": [reference_images[i] for i in order]})
        ids = tokenize_prompt([{"role": "user", "content": text}], media, agent_processor)
        requests.append({"ids": ids})
        conditioned.append(media.get("images", []))
    assert agent_processor.call_count == len(texts)
    encoder_inputs = []
    handle = pipeline.text_encoder.register_forward_pre_hook(
        lambda module, positional, kwargs: encoder_inputs.append(
            {name: value.detach().cpu().clone() for name, value in kwargs.items() if isinstance(value, torch.Tensor)}
        ),
        with_kwargs=True,
    )
    try:
        baseline = QwenImage21Pipeline.encode_prompt(
            pipeline, texts, conditioned, num_images_per_prompt=2, max_sequence_length=2048
        )
        pipeline.processor = Mock(wraps=processor)
        pipeline.processor.tokenizer = processor.tokenizer
        actual = pipeline.encode_prompt(requests, conditioned, num_images_per_prompt=2, max_sequence_length=2048)
        assert pipeline.processor.call_count == 0
        assert pipeline.processor.image_processor.call_count == (1 if order else 0)
    finally:
        pipeline.processor = processor
        handle.remove()
    assert len(encoder_inputs) == 2
    first, second = encoder_inputs
    for name in ("input_ids", "attention_mask", "mm_token_type_ids"):
        torch.testing.assert_close(first[name], second[name], atol=0, rtol=0)
    if order:
        torch.testing.assert_close(first["image_grid_thw"], second["image_grid_thw"], atol=0, rtol=0)
        torch.testing.assert_close(first["pixel_values"].to(torch.bfloat16), second["pixel_values"], atol=0, rtol=0)
    for expected, result in zip(baseline, actual, strict=True):
        torch.testing.assert_close(result, expected, atol=0, rtol=0)
        assert torch.isfinite(result).all()


def _dit_inputs(model, config, references, target_hw):
    generator = torch.Generator().manual_seed(79900 + references)

    def random_tensor(*shape):
        return torch.randn(*shape, generator=generator).to(device="cuda", dtype=torch.bfloat16)

    condition_hw = [(16, 16), (16, 24), (24, 16), (16, 32)][:references]
    condition_tokens = sum(height * width for height, width in condition_hw)
    length = 8 + condition_tokens // 4
    mask = torch.ones(1, length, dtype=torch.long, device="cuda")
    tags = torch.zeros_like(mask, dtype=torch.bool)
    tags[:, : condition_tokens // 4] = True
    shapes = torch.zeros(1, 4, 3, dtype=torch.long)
    if references:
        shapes[:, :references] = torch.tensor([[1, height, width] for height, width in condition_hw])
    positive, negative = QwenImage21DiffusionNFT.prepare_model_inputs(
        model,
        config,
        random_tensor(1, target_hw * target_hw, 64),
        torch.tensor([500.0], device="cuda"),
        random_tensor(1, length, model.config.context_in_dim),
        mask,
        None,
        None,
        {
            "condition_image_shapes": shapes,
            "target_image_shape": torch.tensor([[1, target_hw, target_hw]]),
            "condition_image_rows": random_tensor(1, condition_tokens, 64),
            "prompt_token_tags": tags,
        },
        0,
    )
    assert negative is None
    return positive


def _prediction_errors(expected, actual):
    assert expected.shape == actual.shape
    assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
    reference, value = expected.double(), actual.double()
    difference = value - reference
    return {
        "max_abs": difference.abs().max().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "relative_l2": (difference.norm() / reference.norm().clamp_min(1e-12)).item(),
    }


@torch.no_grad()
def test_dit_lora_first_load_replacement_and_unload(record_property):
    """固定同一基座和输入，分别验证首次加载、同 ID 替换与零卸载残差。"""
    from diffusers import QwenImage21Transformer2DModel
    from peft import LoraConfig
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import (
        AttentionConfig,
        AttentionSpec,
        DiffusionParallelConfig,
        OmniDiffusionConfig,
        TransformerConfig,
    )
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
    from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
        QwenImage21Transformer2DModel as OmniTransformer,
    )

    from tests.pipelines.test_qwen_image_lora_mapping_on_cpu import _assert_bound_weights
    from verl_omni.utils.vllm_omni.utils import OmniTensorLoRARequest, VLLMOmniHijack
    from verl_omni.workers.engine.fsdp.diffusers_impl import _cast_loaded_diffusers_module
    from verl_omni.workers.engine.lora_export import export_lora_for_sync

    root = Path(os.environ["QWEN21_MODEL_PATH"])
    config = SimpleNamespace(
        pipeline=SimpleNamespace(true_cfg_scale=1.0),
        lora_rank=64,
        lora_alpha=128,
        lora={},
        fsdp_layer_prefixes=["transformer_blocks."],
    )
    actor = QwenImage21Transformer2DModel.from_pretrained(
        root, subfolder="transformer", torch_dtype=torch.bfloat16, local_files_only=True
    )
    _cast_loaded_diffusers_module(
        actor,
        torch.bfloat16,
        preserve_fp32_modules=QwenImage21DiffusionNFT.preserve_fp32_modules(),
        preserve_buffer_names=QwenImage21DiffusionNFT.preserve_buffer_names,
    )
    assert actor.time_text_embed.time_proj.freqs.dtype == torch.float32
    architecture = dict(actor.config)
    # 保留 CPU 基座供真实 loader 使用；两端顺序占用 GPU，避免同时加载两份真实 DiT。
    base_weights = {name: value.detach().cpu() for name, value in actor.state_dict().items()}
    actor = actor.eval().requires_grad_(False).to("cuda")
    cases = [(count, 32) for count in range(5)] + [(4, 64)]
    inputs = [_dit_inputs(actor, config, count, target_hw) for count, target_hw in cases]

    def actor_predictions(module):
        return [QwenImage21DiffusionNFT.forward(module, config, item).cpu() for item in inputs]

    expected = {"base": actor_predictions(actor)}
    lora = LoraConfig(r=64, lora_alpha=128, target_modules=["to_q", "to_k", "to_v", "to_out.0"])
    actor.add_adapter(lora, adapter_name="default")
    actor.add_adapter(lora, adapter_name="old")
    actor.set_adapter("old")
    _cast_loaded_diffusers_module(
        actor,
        torch.bfloat16,
        preserve_fp32_modules=QwenImage21DiffusionNFT.preserve_fp32_modules(),
        preserve_buffer_names=QwenImage21DiffusionNFT.preserve_buffer_names,
    )
    actor.eval()
    generator = torch.Generator().manual_seed(79921)
    for name, parameter in actor.named_parameters():
        if ".lora_" in name and ".old." in name:
            parameter.copy_(torch.randn(parameter.shape, generator=generator).to(parameter) * 0.03)
    exports = {}
    for phase in ("first_load", "replacement"):
        if phase == "replacement":
            for name, parameter in actor.named_parameters():
                if ".lora_B.old." in name:
                    parameter.mul_(-1.5)
        expected[phase] = actor_predictions(actor)
        tensors, peft_config = export_lora_for_sync(
            actor,
            model_config=config,
            base_sync_done=True,
            adapter_name="old",
            is_diffusers=True,
            key_prefix="transformer.",
        )
        exports[phase] = ({name: value.clone() for name, value in tensors}, peft_config)
    del actor
    gc.collect()
    torch.cuda.empty_cache()

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    rows = []
    manager = pipeline = transformer = None
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0, distributed_init_method=f"tcp://127.0.0.1:{port}", backend="nccl"
        )
        initialize_model_parallel(tensor_model_parallel_size=1, backend="nccl")
        try:
            od_config = OmniDiffusionConfig(
                model=str(root),
                tf_model_config=TransformerConfig.from_dict(architecture),
                dtype=torch.bfloat16,
                enforce_eager=True,
                parallel_config=DiffusionParallelConfig(tensor_parallel_size=1),
                diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
            )
            kwargs = {k: v for k, v in architecture.items() if k in inspect.signature(OmniTransformer).parameters}
            previous_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                with set_current_diffusion_config(od_config), torch.device("cuda"):
                    transformer = OmniTransformer(od_config, **kwargs).eval().requires_grad_(False)
            finally:
                torch.set_default_dtype(previous_dtype)
            loaded = transformer.load_weights(base_weights.items())
            assert set(base_weights) <= loaded
            del base_weights
            pipeline = object.__new__(QwenImage21DiffusionNFTPipeline)
            torch.nn.Module.__init__(pipeline)
            pipeline.transformer = transformer
            pipeline.od_config = od_config
            VLLMOmniHijack.hijack()
            manager = DiffusionLoRAManager(pipeline, device=torch.device("cuda"), dtype=torch.bfloat16)

            def rollout_predictions():
                predictions = []
                for item in inputs:
                    arguments = dict(item)
                    target = arguments.pop("_target_seq_len")
                    arguments.pop("kv_cache_mode")
                    arguments["cache_branch"] = "cond"
                    with (
                        set_forward_context(omni_diffusion_config=od_config),
                        sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION),
                    ):
                        predictions.append(transformer(**arguments)[0][:, -target:].cpu())
                return predictions

            baseline = rollout_predictions()
            previous = baseline
            for phase in ("base", "first_load", "replacement"):
                if phase != "base":
                    tensors, peft_config = exports[phase]
                    mapped, _ = pipeline.map_lora_update_to_engine(
                        {name: value.clone() for name, value in tensors.items()}, peft_config
                    )
                    manager.set_active_adapter(
                        OmniTensorLoRARequest(
                            lora_name="old",
                            lora_int_id=1,
                            lora_path="in-memory",
                            lora_tensors=tensors,
                            peft_config=peft_config,
                        )
                    )
                    _assert_bound_weights(manager, mapped)
                predictions = baseline if phase == "base" else rollout_predictions()
                for count, (reference, actual) in enumerate(zip(expected[phase], predictions, strict=True)):
                    errors = _prediction_errors(reference, actual)
                    rows.append(
                        {"phase": phase, "references": cases[count][0], "resolution": cases[count][1] * 16, **errors}
                    )
                    if phase != "base":
                        assert not torch.equal(actual, baseline[count]), "Nonzero LoRA must affect predictions."
                        assert not torch.equal(actual, previous[count]), "Replacement must change predictions."
                if phase != "base":
                    assert manager.remove_adapter(1)
                    for count, (reference, restored) in enumerate(zip(baseline, rollout_predictions(), strict=True)):
                        rows.append(
                            {
                                "phase": phase + "_unloaded",
                                "references": cases[count][0],
                                "resolution": cases[count][1] * 16,
                                **_prediction_errors(reference, restored),
                            }
                        )
                        torch.testing.assert_close(restored, reference, atol=0, rtol=0)
                    previous = predictions
        finally:
            record_property("dit_lora_prediction_errors", json.dumps(rows))
            manager = pipeline = transformer = None
            destroy_model_parallel()
            destroy_distributed_environment()
            gc.collect()
            torch.cuda.empty_cache()
