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
"""CPU tests for vLLM-Omni LoRA loader hijack and request types."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from verl_omni.utils.vllm_omni.utils import (
    DiffusionLoRAManager,
    OmniLoRARequest,
    OmniTensorLoRARequest,
    VLLMOmniHijack,
)


@pytest.fixture(autouse=True)
def ensure_hijacked():
    VLLMOmniHijack.hijack()


def _make_mock_lora_model(model_id=1):
    mock_lora = MagicMock()
    mock_lora.optimize = MagicMock()
    mock_model = MagicMock()
    mock_model.id = model_id
    mock_model.loras = {"mod": mock_lora}
    return mock_model


def _make_manager(pipeline=None):
    manager = object.__new__(DiffusionLoRAManager)
    manager._expected_lora_modules = ["transformer.attn"]
    manager.dtype = torch.float32
    manager.pipeline = pipeline
    return manager


def test_load_adapter_tensor_request_without_unbound_local(monkeypatch):
    """OmniTensorLoRARequest must not raise UnboundLocalError for 'loaded'."""
    manager = _make_manager()
    req = OmniTensorLoRARequest(
        lora_name="test_tensor_lora",
        lora_int_id=42,
        lora_path="memory://lora",
        peft_config={"r": 8, "lora_alpha": 16, "target_modules": ["attn"]},
        lora_tensors={"attn.lora_A.weight": torch.zeros(8, 16)},
    )

    mock_peft_helper = SimpleNamespace(r=8, lora_alpha=16, target_modules=["attn"])
    mock_lora_model = _make_mock_lora_model(42)

    with (
        patch("verl_omni.utils.vllm_omni.utils.PEFTHelper.from_dict", return_value=mock_peft_helper) as mock_from_dict,
        patch(
            "verl_omni.utils.vllm_omni.utils.LoRAModel.from_lora_tensors", return_value=mock_lora_model
        ) as mock_from_tensors,
    ):
        model, helper = manager._load_adapter(req)

        mock_from_dict.assert_called_once_with(req.peft_config)
        assert mock_from_tensors.call_count == 1
        call_kwargs = mock_from_tensors.call_args.kwargs
        assert torch.equal(call_kwargs["tensors"]["attn.lora_A.weight"], req.lora_tensors["attn.lora_A.weight"])
        assert call_kwargs["peft_helper"] is mock_peft_helper
        assert call_kwargs["lora_model_id"] == 42
        assert call_kwargs["device"] == "cpu"
        assert call_kwargs["dtype"] == torch.float32
        assert model is mock_lora_model
        assert helper is mock_peft_helper
        mock_lora_model.loras["mod"].optimize.assert_called_once()


def test_load_adapter_tensor_request_with_pipeline_mapper():
    """Tensors and PEFT config pass through pipeline.map_lora_update_to_engine when present."""
    pipeline = SimpleNamespace(
        map_lora_update_to_engine=MagicMock(
            return_value=({"mapped.weight": torch.ones(4, 4)}, {"r": 4, "lora_alpha": 8, "target_modules": ["mapped"]})
        )
    )
    manager = _make_manager(pipeline=pipeline)
    req = OmniTensorLoRARequest(
        lora_name="mapped_lora",
        lora_int_id=7,
        lora_path="memory://lora",
        peft_config={"r": 8, "lora_alpha": 16, "target_modules": ["raw"]},
        lora_tensors={"raw.weight": torch.zeros(8, 8)},
    )

    mock_peft_helper = SimpleNamespace(r=4, lora_alpha=8, target_modules=["mapped"])
    mock_lora_model = _make_mock_lora_model(7)

    with (
        patch("verl_omni.utils.vllm_omni.utils.PEFTHelper.from_dict", return_value=mock_peft_helper) as mock_from_dict,
        patch(
            "verl_omni.utils.vllm_omni.utils.LoRAModel.from_lora_tensors", return_value=mock_lora_model
        ) as mock_from_tensors,
    ):
        model, helper = manager._load_adapter(req)

        pipeline.map_lora_update_to_engine.assert_called_once_with(req.lora_tensors, req.peft_config)
        mock_from_dict.assert_called_once_with({"r": 4, "lora_alpha": 8, "target_modules": ["mapped"]})
        assert mock_from_tensors.call_count == 1
        call_kwargs = mock_from_tensors.call_args.kwargs
        assert torch.equal(call_kwargs["tensors"]["mapped.weight"], torch.ones(4, 4))
        assert call_kwargs["peft_helper"] is mock_peft_helper
        assert call_kwargs["lora_model_id"] == 7
        assert call_kwargs["device"] == "cpu"
        assert call_kwargs["dtype"] == torch.float32
        assert model is mock_lora_model
        assert helper is mock_peft_helper


def test_load_adapter_file_hook():
    """Pipeline-owned _load_diffusion_lora_adapter hook takes precedence."""
    mock_peft_helper = SimpleNamespace(r=16, lora_alpha=32, target_modules=["attn"])
    mock_lora_model = _make_mock_lora_model(11)
    hook = MagicMock(return_value=(mock_lora_model, mock_peft_helper))

    pipeline = SimpleNamespace(_load_diffusion_lora_adapter=hook)
    manager = _make_manager(pipeline=pipeline)
    req = OmniLoRARequest(
        lora_name="file_hook_lora",
        lora_int_id=11,
        lora_path="/fake/checkpoint/dir",
    )

    with (
        patch("verl_omni.utils.vllm_omni.utils.get_adapter_absolute_path", return_value="/resolved/checkpoint/dir"),
        patch("verl_omni.utils.vllm_omni.utils.LoRAModel.from_local_checkpoint") as mock_from_ckpt,
    ):
        model, helper = manager._load_adapter(req)

        hook.assert_called_once_with(
            lora_request=req,
            lora_path="/resolved/checkpoint/dir",
            dtype=torch.float32,
        )
        mock_from_ckpt.assert_not_called()
        assert model is mock_lora_model
        assert helper is mock_peft_helper
        mock_lora_model.loras["mod"].optimize.assert_called_once()


def test_load_adapter_file_fallback():
    """Without pipeline hook, fall back to from_local_dir + from_local_checkpoint."""
    manager = _make_manager()
    req = OmniLoRARequest(
        lora_name="file_fallback_lora",
        lora_int_id=13,
        lora_path="/fake/fallback/dir",
    )

    mock_peft_helper = SimpleNamespace(r=32, lora_alpha=64, target_modules=["transformer.attn"])
    mock_lora_model = _make_mock_lora_model(13)

    with (
        patch("verl_omni.utils.vllm_omni.utils.get_adapter_absolute_path", return_value="/resolved/fallback/dir"),
        patch(
            "verl_omni.utils.vllm_omni.utils.PEFTHelper.from_local_dir", return_value=mock_peft_helper
        ) as mock_from_dir,
        patch(
            "verl_omni.utils.vllm_omni.utils.LoRAModel.from_local_checkpoint", return_value=mock_lora_model
        ) as mock_from_ckpt,
    ):
        model, helper = manager._load_adapter(req)

        mock_from_dir.assert_called_once_with(
            "/resolved/fallback/dir",
            max_position_embeddings=None,
            tensorizer_config_dict=req.tensorizer_config_dict,
        )
        mock_from_ckpt.assert_called_once_with(
            "/resolved/fallback/dir",
            expected_lora_modules=["transformer.attn"],
            peft_helper=mock_peft_helper,
            lora_model_id=13,
            device="cpu",
            dtype=torch.float32,
            model_vocab_size=None,
            tensorizer_config_dict=req.tensorizer_config_dict,
            weights_mapper=None,
        )
        assert model is mock_lora_model
        assert helper is mock_peft_helper
