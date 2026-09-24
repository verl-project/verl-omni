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
"""CPU regressions for Boogu-Image live LoRA name mapping."""

from types import SimpleNamespace

import pytest
import torch

_JOINT_TARGETS = [
    "img_to_q",
    "img_to_k",
    "img_to_v",
    "img_out",
    "instruct_to_q",
    "instruct_to_k",
    "instruct_to_v",
    "instruct_out",
]

# All 18 targets from examples/flowgrpo_trainer/boogu_image/run_boogu_image_ocr_lora.sh.
_BOOGU_TARGETS = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    *_JOINT_TARGETS,
    "feed_forward.linear_1",
    "feed_forward.linear_2",
    "feed_forward.linear_3",
    "img_feed_forward.linear_1",
    "img_feed_forward.linear_2",
    "img_feed_forward.linear_3",
]


@pytest.mark.parametrize("container", [list, tuple, set])
def test_boogu_maps_keys_and_targets_without_mutating_inputs(container):
    from verl_omni.pipelines.model_base import VllmOmniPipelineBase

    pipeline = VllmOmniPipelineBase.get_class("BooguImagePipeline", "flow_grpo")
    tensor = torch.ones(4, 32)
    prefix = "transformer.double_stream_layers.0.img_instruct_attn."
    unrelated = "transformer.other_attn.processor.to_q.lora_A.weight"
    tensors = {f"{prefix}to_out.0.lora_A.weight": tensor, unrelated: tensor}
    tensors.update(
        {f"{prefix}processor.{target}.lora_{side}.weight": tensor for target in _JOINT_TARGETS for side in ("A", "B")}
    )
    targets = container([*_BOOGU_TARGETS, "double_stream_layers.0.img_instruct_attn.to_out.0"])
    config = {"target_modules": targets, "r": 4, "lora_alpha": 8}
    mapped, mapped_config = pipeline.map_lora_update_to_engine(tensors, config)
    assert set(mapped) == {
        f"{prefix}to_out.lora_A.weight",
        unrelated,
        *(f"{prefix}{target}.lora_{side}.weight" for target in _JOINT_TARGETS for side in ("A", "B")),
    }
    assert all(value is tensor for value in mapped.values())
    assert set(mapped_config["target_modules"]) == (set(_BOOGU_TARGETS) - {"to_out.0"}) | {
        "to_out",
        "double_stream_layers.0.img_instruct_attn.to_out",
    }
    assert config["target_modules"] is targets and "to_out.0" in targets
    assert f"{prefix}to_out.0.lora_A.weight" in tensors
    assert f"{prefix}processor.img_to_q.lora_A.weight" in tensors
    assert mapped_config["r"] == 4 and mapped_config["lora_alpha"] == 8


@pytest.fixture
def boogu_manager(monkeypatch):
    import vllm.distributed.parallel_state as parallel_state
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
    from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
    from vllm_omni.diffusion.models.boogu_image import boogu_image_transformer as boogu

    from verl_omni.pipelines.boogu_image_flow_grpo.vllm_omni_rollout_adapter import BooguImagePipelineWithLogProb
    from verl_omni.utils.vllm_omni.utils import VLLMOmniHijack

    # Match CPU-only CI even when the local host supports pinned-memory copies.
    monkeypatch.setattr("vllm.lora.lora_model.PIN_MEMORY", False)
    monkeypatch.setattr(parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1))
    monkeypatch.setattr(boogu, "Attention", lambda **_: torch.nn.Identity())
    monkeypatch.setattr(VLLMOmniHijack, "_patched", False)
    monkeypatch.setattr(DiffusionLoRAManager, "_load_adapter", DiffusionLoRAManager._load_adapter)
    monkeypatch.setattr("verl_omni.utils.vllm_omni.utils.VLLMHijack.hijack", lambda: None)
    VLLMOmniHijack.hijack()
    with set_current_vllm_config(VllmConfig()):
        transformer = boogu.BooguImageTransformer2DModel(
            OmniDiffusionConfig(
                tf_model_config=TransformerConfig.from_dict(
                    {
                        "hidden_size": 32,
                        "num_layers": 2,
                        "num_double_stream_layers": 1,
                        "num_refiner_layers": 1,
                        "num_attention_heads": 1,
                        "num_kv_heads": 1,
                        "multiple_of": 32,
                        "axes_dim_rope": (8, 12, 12),
                        "axes_lens": (16, 16, 16),
                        "instruction_feature_configs": {
                            "instruction_feat_dim": 32,
                            "reduce_type": "mean",
                            "num_instruction_feature_layers": 1,
                        },
                    }
                )
            )
        )
        assert isinstance(transformer.single_stream_layers[0].attn.to_out, boogu.RowParallelLinear)
        assert isinstance(transformer.double_stream_layers[0].img_instruct_attn.to_out, boogu.ReplicatedLinear)
        pipeline = object.__new__(BooguImagePipelineWithLogProb)
        torch.nn.Module.__init__(pipeline)
        pipeline.transformer = transformer
        params = {}
        matched_targets = set()
        # Use real runtime sizes with actor names documented by Boogu's load_weights.
        # This models exported tensors, not a real Boogu actor/checkpoint export.
        for name, module in transformer.named_modules():
            actor_name = name
            if name.rsplit(".", 1)[-1] in _JOINT_TARGETS:
                actor_name = name.replace(".img_instruct_attn.", ".img_instruct_attn.processor.")
            if actor_name.endswith(".to_out"):
                actor_name += ".0"
            matches = {target for target in _BOOGU_TARGETS if actor_name.endswith(f".{target}")}
            if not matches:
                continue
            matched_targets.update(matches)
            out_features, in_features = module.weight.shape
            prefix = f"transformer.{actor_name}"
            params[f"{prefix}.lora_A.weight"] = torch.full((4, in_features), 0.125)
            params[f"{prefix}.lora_B.weight"] = torch.full((out_features, 4), 0.25)
        assert matched_targets == set(_BOOGU_TARGETS)
        assert len(params) == 88  # 44 module paths across all blocks and refiners.
        assert sum(".processor." in name for name in params) == 16
        config = {"r": 4, "lora_alpha": 8, "target_modules": _BOOGU_TARGETS}
        manager = DiffusionLoRAManager(pipeline, device=torch.device("cpu"), dtype=torch.float32)
        yield manager, params, config


@pytest.mark.parametrize("case", ["unmapped", "output_rename_only", "full", "zero_init", "output_only"])
def test_boogu_load_bind_activate_contract(boogu_manager, monkeypatch, case):
    from verl_omni.utils.vllm_omni.utils import OmniTensorLoRARequest

    manager, params, config = boogu_manager
    if case == "unmapped":
        monkeypatch.setattr(manager.pipeline, "map_lora_update_to_engine", lambda tensors, config: (tensors, config))
    elif case == "output_rename_only":
        # Reproduce the old mapper: output projections bind, processor projections do not.
        monkeypatch.setattr(
            manager.pipeline,
            "map_lora_update_to_engine",
            lambda tensors, config: (
                {name.replace(".to_out.0.", ".to_out."): tensor for name, tensor in tensors.items()},
                {**config, "target_modules": ["to_out" if t == "to_out.0" else t for t in config["target_modules"]]},
            ),
        )
    elif case == "zero_init":
        params = {name: torch.zeros_like(tensor) if ".lora_B." in name else tensor for name, tensor in params.items()}
    elif case == "output_only":
        params = {name: tensor for name, tensor in params.items() if ".to_out.0." in name}
        config = {**config, "target_modules": ["to_out.0"]}
    # The loader may scale B in place when CPU tensors share storage.
    mapped, _ = manager.pipeline.map_lora_update_to_engine(
        {name: tensor.clone() for name, tensor in params.items()}, config
    )
    manager.set_active_adapter(
        OmniTensorLoRARequest(
            lora_name="boogu", lora_int_id=1, lora_path="in-memory", lora_tensors=params, peft_config=config
        )
    )
    assert manager._active_adapter_id == 1
    loaded = manager._registered_adapters[1]
    bound = {name for name in manager._lora_modules if manager._get_lora_weights(loaded, name) is not None}
    if case == "unmapped":
        assert len(bound) == 30  # Six output and eight processor projections stay unbound.
        assert any(name.endswith(".to_out.0") for name in loaded.loras)
        return
    if case == "output_rename_only":
        assert len(manager._lora_modules) == 44 and len(bound) == 36
        missing = set(manager._lora_modules) - bound
        assert missing == {f"transformer.double_stream_layers.0.img_instruct_attn.{t}" for t in _JOINT_TARGETS}
        for name in missing:
            layer = manager._lora_modules[name]
            assert all(torch.count_nonzero(t) == 0 for t in (*layer.lora_a_stacked, *layer.lora_b_stacked))
        return

    assert len(bound) == len(manager._lora_modules) == len(params) // 2
    assert any(name.endswith(".to_out") for name in bound)
    for name, layer in manager._lora_modules.items():
        expected_a = mapped[f"{name}.lora_A.weight"]
        expected_b = mapped[f"{name}.lora_B.weight"] * 2
        torch.testing.assert_close(layer.lora_a_stacked[0][0, 0, :4], expected_a, rtol=0, atol=0)
        torch.testing.assert_close(layer.lora_b_stacked[0][0, 0, :, :4], expected_b, rtol=0, atol=0)
