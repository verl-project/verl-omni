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
"""CPU regressions for Qwen-Image live LoRA name mapping."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    ("architecture", "algorithm"),
    [
        ("QwenImagePipeline", "flow_grpo"),
        ("QwenImagePipeline", "dual_grpo"),
        ("QwenImagePipeline", "mix_grpo"),
        ("QwenImagePipeline", "diffusion_nft"),
        ("QwenImage21Pipeline", "diffusion_nft"),
        ("QwenImagePipeline", "dpo"),
        ("QwenImageEditPlusPipeline", "flow_grpo"),
    ],
)
@pytest.mark.parametrize("container", [list, tuple, set])
def test_registered_pipelines_map_keys_and_targets_without_mutating_inputs(architecture, algorithm, container):
    from verl_omni.pipelines.model_base import VllmOmniPipelineBase

    pipeline = VllmOmniPipelineBase.get_class(architecture, algorithm)
    tensor = torch.ones(2, 3)
    prefix = "transformer.transformer_blocks.0.attn."
    wrapped = "transformer.base_model.model.transformer_blocks.0.attn."
    tensors = {f"{wrapped}to_out.0.lora_A.weight": tensor, f"{prefix}to_q.lora_B.weight": tensor}
    targets = container(["to_out.0", "to_q", "transformer_blocks.0.attn.to_out.0", "img_mlp.net.0.proj"])
    config = {"target_modules": targets, "r": 4, "lora_alpha": 8}
    mapped, mapped_config = pipeline.map_lora_update_to_engine(tensors, config)
    assert list(mapped) == [f"{prefix}to_out.lora_A.weight", f"{prefix}to_q.lora_B.weight"]
    assert all(value is tensor for value in mapped.values())
    assert set(mapped_config["target_modules"]) == {
        "to_out",
        "to_q",
        "transformer_blocks.0.attn.to_out",
        "img_mlp.net.0.proj",
    }
    assert config["target_modules"] is targets and "to_out.0" in targets
    assert f"{wrapped}to_out.0.lora_A.weight" in tensors
    assert mapped_config["r"] == 4 and mapped_config["lora_alpha"] == 8


_QWEN_TARGETS = [
    "to_q",
    "to_k",
    "to_v",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_out.0",
    "to_add_out",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
]


def _export_qwen_actor(monkeypatch, backend, architecture):
    from diffusers import QwenImageTransformer2DModel
    from peft import LoraConfig, get_peft_model

    from tests.workers.test_diffusers_fsdp_merged_lora_on_cpu import _make_engine, _patch_sync_helpers

    model_class = QwenImageTransformer2DModel
    targets = _QWEN_TARGETS
    model_kwargs = {"joint_attention_dim": 32}
    adapter_name = "default"
    if architecture == "qwen21":
        from diffusers import QwenImage21Transformer2DModel

        model_class = QwenImage21Transformer2DModel
        targets = ["to_q", "to_k", "to_v", "to_out.0"]
        model_kwargs = {"context_in_dim": 32}
        adapter_name = "old"
    model = model_class(
        num_layers=1,
        num_attention_heads=1,
        attention_head_dim=32,
        axes_dims_rope=(8, 12, 12),
        **model_kwargs,
    )
    if backend == "veomni":
        lora = pytest.importorskip("veomni.lora")
        from tests.workers.veomni_lora_helpers import export_veomni_params, make_veomni_engine

        model = lora.VeOmniLoraModel(model, lora.VeOmniLoraConfig(r=4, lora_alpha=8, target_modules=targets))
    else:
        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=targets))
        if adapter_name == "old":
            model.add_adapter("old", LoraConfig(r=4, lora_alpha=8, target_modules=targets))
    with torch.no_grad():
        for index, (name, param) in enumerate(model.named_parameters()):
            if ".lora_" in name:
                # Distinct signs catch accidentally exporting the current policy instead of NFT's old policy.
                value = -0.25 if adapter_name == "old" and ".default." in name else (index + 1) / 128
                param.fill_(value)
    if backend == "veomni":
        engine = make_veomni_engine(model, lora_rank=4)
        params, config = export_veomni_params(engine, monkeypatch, base_sync_done=True)
    else:
        _patch_sync_helpers(monkeypatch)
        engine = _make_engine(model, lora_config={})
        params, config = engine.get_per_tensor_param(base_sync_done=True, adapter_name=adapter_name)
        params = dict(params)
        assert model.active_adapter == "default"
    assert len(params) == 2 * len(targets)
    assert all(torch.all(tensor > 0) for tensor in params.values())
    # The rollout receives copied transport tensors, not live actor Parameters.
    return {name: tensor.detach().clone() for name, tensor in params.items()}, config


@pytest.fixture(
    params=[("fsdp2", "qwen"), ("veomni", "qwen"), ("fsdp2", "qwen21")],
    ids=["qwen-fsdp2", "qwen-veomni", "qwen21-nft-fsdp2"],
)
def runtime_manager(monkeypatch, request):
    import vllm.distributed.parallel_state as parallel_state
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.qwen_image import qwen_image_transformer as qwen
    from vllm_omni.lora.manager import DiffusionLoRAManager

    from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb
    from verl_omni.utils.vllm_omni.utils import VLLMOmniHijack

    backend, architecture = request.param
    model_kwargs = {"joint_attention_dim": 32}
    transformer_class = qwen.QwenImageTransformer2DModel
    pipeline_class = QwenImagePipelineWithLogProb
    if architecture == "qwen21":
        from vllm_omni.diffusion.models.qwen_image_21 import qwen_image_21_transformer as qwen

        from verl_omni.pipelines.qwen_image_21_diffusion_nft.vllm_omni_rollout_adapter import (
            QwenImage21DiffusionNFTPipeline,
        )

        transformer_class = qwen.QwenImage21Transformer2DModel
        pipeline_class = QwenImage21DiffusionNFTPipeline
        model_kwargs = {"context_in_dim": 32}

    # Match CPU-only CI even when the local host supports pinned-memory copies.
    monkeypatch.setattr("vllm.lora.lora_model.PIN_MEMORY", False)
    monkeypatch.setattr(parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1))
    monkeypatch.setattr(qwen, "Attention", lambda **_: torch.nn.Identity())
    monkeypatch.setattr(VLLMOmniHijack, "_patched", False)
    monkeypatch.setattr(DiffusionLoRAManager, "_load_adapter", DiffusionLoRAManager._load_adapter)
    monkeypatch.setattr("verl_omni.utils.vllm_omni.utils.VLLMHijack.hijack", lambda: None)
    VLLMOmniHijack.hijack()
    with set_current_vllm_config(VllmConfig()):
        transformer = transformer_class(
            OmniDiffusionConfig(),
            num_layers=1,
            num_attention_heads=1,
            attention_head_dim=32,
            axes_dims_rope=(8, 12, 12),
            **model_kwargs,
        )
        transformer.load_weights([])  # Installs the real QKV stacked-parameter mapping.
        pipeline = object.__new__(pipeline_class)
        torch.nn.Module.__init__(pipeline)
        pipeline.transformer = transformer
        manager = DiffusionLoRAManager(pipeline, device=torch.device("cpu"), dtype=torch.float32)
        params, config = _export_qwen_actor(monkeypatch, backend, architecture)
        yield manager, params, config


@pytest.mark.parametrize("case", ["unmapped", "full", "zero_init", "output_only"])
def test_export_load_bind_activate_contract(runtime_manager, monkeypatch, case):
    from verl_omni.utils.vllm_omni.utils import OmniTensorLoRARequest

    manager, params, config = runtime_manager
    if case == "unmapped":
        params = {name.replace("transformer.base_model.model.", "transformer.", 1): t for name, t in params.items()}
        monkeypatch.setattr(manager.pipeline, "map_lora_update_to_engine", lambda tensors, config: (tensors, config))
    elif case == "zero_init":
        params = {name: torch.zeros_like(tensor) if ".lora_B." in name else tensor for name, tensor in params.items()}
    elif case == "output_only":
        params = {name: tensor for name, tensor in params.items() if ".to_out.0." in name}
        config = {**config, "target_modules": ["to_out.0"]}
    # The loader may scale B in place when CPU tensors share storage.
    mapped, _ = manager.pipeline.map_lora_update_to_engine(
        {name: tensor.clone() for name, tensor in params.items()}, config
    )
    request = OmniTensorLoRARequest(
        lora_name="actor", lora_int_id=1, lora_path="in-memory", lora_tensors=params, peft_config=config
    )
    if case == "unmapped":
        with pytest.raises(ValueError, match=r"binding is incomplete.*to_out\.0"):
            manager.set_active_adapter(request)
        return

    manager.set_active_adapter(request)
    assert manager._active_adapter_id == 1

    _assert_bound_weights(manager, mapped)


def _assert_bound_weights(manager, mapped):
    assert any(name.endswith(".to_out") for name in manager._lora_modules)
    bound = 0
    for name, layer in manager._lora_modules.items():
        prefix, _, suffix = name.rpartition(".")
        sublayers = manager._packed_modules_mapping.get(suffix, [suffix])
        for index, sublayer in enumerate(sublayers):
            key = f"{prefix}.{sublayer}"
            expected_a = mapped[f"{key}.lora_A.weight"].float()
            expected_b = mapped[f"{key}.lora_B.weight"].float() * 2
            rank = expected_a.shape[0]
            torch.testing.assert_close(
                layer.lora_a_stacked[index][0, 0, :rank].float().cpu(), expected_a, rtol=0, atol=0
            )
            torch.testing.assert_close(
                layer.lora_b_stacked[index][0, 0, :, :rank].float().cpu(), expected_b, rtol=0, atol=0
            )
            bound += 1
    assert bound == len(mapped) // 2


def test_export_replace_and_remove_adapter_clears_previous_weights(runtime_manager):
    from verl_omni.utils.vllm_omni.utils import OmniTensorLoRARequest

    manager, params, config = runtime_manager
    # The worker removes the previous adapter before reusing the same ID for an update.
    for factor in (1.0, -0.5):
        tensors = {name: tensor * factor for name, tensor in params.items()}
        mapped, _ = manager.pipeline.map_lora_update_to_engine(
            {name: tensor.clone() for name, tensor in tensors.items()}, config
        )
        request = OmniTensorLoRARequest(
            lora_name="actor", lora_int_id=1, lora_path="in-memory", lora_tensors=tensors, peft_config=config
        )
        manager.set_active_adapter(request)
        assert manager._active_adapter_id == 1
        _assert_bound_weights(manager, mapped)

        assert manager.remove_adapter(1)
        assert manager._active_adapter_id is None
        assert manager.list_adapters() == []
        for layer in manager._lora_modules.values():
            for buffer in (*layer.lora_a_stacked, *layer.lora_b_stacked):
                torch.testing.assert_close(buffer, torch.zeros_like(buffer), rtol=0, atol=0)
