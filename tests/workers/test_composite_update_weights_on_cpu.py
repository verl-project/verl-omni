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
"""CPU tests for composite (Dual-GRPO) trainer→rollout weight sync."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf

import verl_omni.workers.engine_workers as engine_workers_module
from verl_omni.workers.engine.fsdp.diffusers_impl import CompositeFSDPEngine
from verl_omni.workers.engine_workers import ActorRolloutRefWorker


def _make_diffusion_model_config(*, lora_rank: int = 0) -> object:
    from verl_omni.workers.config.diffusion.model import DiffusionModelConfig

    model_cfg = object.__new__(DiffusionModelConfig)
    object.__setattr__(model_cfg, "lora_rank", lora_rank)
    object.__setattr__(model_cfg, "lora_alpha", 16)
    object.__setattr__(model_cfg, "target_modules", "all-linear")
    object.__setattr__(model_cfg, "target_parameters", None)
    object.__setattr__(model_cfg, "exclude_modules", ".*visual.*")
    return model_cfg


def _make_composite_engine(enable_lora: bool = False) -> CompositeFSDPEngine:
    # Apply or not lora for both AR and DiT
    engine = CompositeFSDPEngine.__new__(CompositeFSDPEngine)
    engine.ar_engine = MagicMock(name="ar_engine")
    engine.dit_engine = MagicMock(name="dit_engine")
    engine.ar_stage = True
    engine.current_engine = engine.ar_engine
    engine.ar_engine.is_param_offload_enabled = False
    engine.dit_engine.is_param_offload_enabled = False
    engine.model_config = _make_diffusion_model_config(lora_rank=8 if enable_lora else 0)

    def _attach_lora(sub_engine: MagicMock, enabled: bool):
        if not enabled:
            sub_engine.module = MagicMock(spec=[])
            return
        peft_cfg = {"r": 8, "lora_alpha": 16, "target_modules": ["all-linear"]}
        sub_engine.module = SimpleNamespace(
            _fsdp_wrapped_module=SimpleNamespace(peft_config={"default": peft_cfg}),
        )

    _attach_lora(engine.dit_engine, enable_lora)
    _attach_lora(engine.ar_engine, enable_lora)
    return engine


def _make_actor_rollout_worker(
    composite_engine: CompositeFSDPEngine,
    *,
    peft_merge: bool = False,
    base_sync_done: bool = False,
    free_cache_engine: bool = False,
    load_format: str = "safetensors",
) -> ActorRolloutRefWorker:
    worker = ActorRolloutRefWorker.__new__(ActorRolloutRefWorker)
    worker.role = "actor_rollout"
    worker.peft_merge = peft_merge
    worker.layered_summon = False
    worker._zmq_update_seq = 0
    # base_sync_done is derived from load_format in _init_weight_sync_knobs (dummy => False).
    if not base_sync_done:
        load_format = "dummy"
    worker.config = OmegaConf.create(
        {
            "rollout": {
                "free_cache_engine": free_cache_engine,
                "load_format": load_format,
                "layered_summon": False,
                "rollout_adapter": "default",
                "checkpoint_engine": {
                    "backend": "naive",
                    "update_weights_bucket_megabytes": 1,
                },
            },
            "lora": {"merge": peft_merge},
        }
    )
    worker.actor = MagicMock()
    worker.actor.engine = composite_engine
    worker.rollout = MagicMock()
    worker.rollout.update_weights = AsyncMock()
    worker.rollout.resume = AsyncMock()
    worker.rollout.use_shm = False
    worker.rollout.zmq_handle = "ipc://test"
    worker.rollout._execute_method = AsyncMock(return_value=None)
    worker._init_weight_sync_knobs(worker.config)
    worker._rank = 0

    return worker


class TestCompositeEngineLoraDetection:
    def test_actor_has_lora_uses_composite_has_lora(self):
        engine = _make_composite_engine(True)
        assert engine.has_lora is True

        engine = _make_composite_engine(False)
        assert engine.has_lora is False

    def test_get_per_tensor_param_merges_shared_rollout_peft(self):
        engine = _make_composite_engine(True)
        engine.dit_engine.get_per_tensor_param.return_value = (iter([]), {"r": 8, "lora_alpha": 16})
        engine.ar_engine.get_per_tensor_param.return_value = (iter([]), {"r": 8, "lora_alpha": 16})

        _, cfg = engine.get_per_tensor_param()
        assert cfg is not None
        assert cfg["r"] == 8
        assert cfg["lora_alpha"] == 16
        assert cfg["exclude_modules"] == ".*visual.*"


class TestCompositeUpdateWeightsNaive:
    @pytest.mark.asyncio
    async def test_full_weight_sync_merges_dit_and_ar_keys(self):
        engine = _make_composite_engine()
        dit_tensor = torch.tensor([1.0])
        ar_tensor = torch.tensor([2.0])
        engine.dit_engine.get_per_tensor_param.return_value = (
            iter([("transformer.blocks.0.weight", dit_tensor)]),
            None,
        )
        engine.ar_engine.get_per_tensor_param.return_value = (
            iter([("model.layers.0.weight", ar_tensor)]),
            None,
        )
        worker = _make_actor_rollout_worker(engine, base_sync_done=True)

        with (
            patch.object(engine_workers_module, "set_expandable_segments"),
            patch.object(engine_workers_module, "log_gpu_memory_usage"),
            patch.object(engine_workers_module, "aggressive_empty_cache"),
            patch.object(worker, "_offload_actor_and_empty_cache"),
        ):
            await worker.update_weights(global_steps=1, mode="naive")

        worker.rollout.update_weights.assert_awaited_once()
        call_args = worker.rollout.update_weights.call_args
        assert call_args.kwargs["base_sync_done"] is True
        synced = dict(call_args.args[0])
        assert synced["transformer.blocks.0.weight"] is dit_tensor
        assert synced["text_encoder.model.layers.0.weight"] is ar_tensor

    @pytest.mark.asyncio
    async def test_lora_first_base_sync_then_adapter(self):
        engine = _make_composite_engine(enable_lora=True)
        base_dit = torch.tensor([1.0])
        base_ar = torch.tensor([2.0])
        lora_dit = torch.tensor([3.0])
        lora_ar = torch.tensor([4.0])
        peft_meta = {"r": 8, "exclude_modules": ".*visual.*"}

        def fake_get_per_tensor(*, layered_summon=False, base_sync_done=False, adapter_name=None, **kwargs):
            del layered_summon, adapter_name, kwargs
            if base_sync_done:
                return (
                    iter(
                        [
                            ("transformer.blocks.0.lora_A.weight", lora_dit),
                            ("text_encoder.model.layers.0.lora_A.weight", lora_ar),
                        ]
                    ),
                    peft_meta,
                )
            return (
                iter(
                    [
                        ("transformer.blocks.0.weight", base_dit),
                        ("text_encoder.model.layers.0.weight", base_ar),
                    ]
                ),
                peft_meta,
            )

        engine.get_per_tensor_param = MagicMock(side_effect=fake_get_per_tensor)
        worker = _make_actor_rollout_worker(engine, base_sync_done=False)

        with (
            patch.object(engine_workers_module, "set_expandable_segments"),
            patch.object(engine_workers_module, "log_gpu_memory_usage"),
            patch.object(engine_workers_module, "aggressive_empty_cache"),
            patch.object(worker, "_offload_actor_and_empty_cache"),
        ):
            await worker.update_weights(global_steps=0, mode="naive")

        assert worker.rollout.update_weights.await_count == 2
        base_call, adapter_call = worker.rollout.update_weights.await_args_list
        base_weights = dict(base_call.args[0])
        adapter_weights = dict(adapter_call.args[0])
        assert base_call.kwargs["base_sync_done"] is False
        assert adapter_call.kwargs["base_sync_done"] is True
        assert "transformer.blocks.0.weight" in base_weights
        assert "text_encoder.model.layers.0.weight" in base_weights
        assert "transformer.blocks.0.lora_A.weight" in adapter_weights
        assert "text_encoder.model.layers.0.lora_A.weight" in adapter_weights
        assert worker.base_sync_done is True

    @pytest.mark.asyncio
    async def test_lora_fast_path_after_base_sync(self):
        # DiT only
        engine = _make_composite_engine(enable_lora=True)
        lora_tensor = torch.tensor([5.0])
        peft_meta = {"r": 8, "exclude_modules": ".*visual.*"}
        engine.get_per_tensor_param = MagicMock(
            return_value=(
                iter([("transformer.blocks.0.lora_A.weight", lora_tensor)]),
                peft_meta,
            )
        )
        worker = _make_actor_rollout_worker(engine, base_sync_done=True, load_format="safetensors")

        sent: dict[str, torch.Tensor] = {}

        class _Sender:
            async def async_send_weights(self, items):
                sent.update(dict(items))

        with (
            patch.object(engine_workers_module, "set_expandable_segments"),
            patch.object(engine_workers_module, "log_gpu_memory_usage"),
            patch.object(engine_workers_module, "aggressive_empty_cache"),
            patch.object(engine_workers_module, "BucketedWeightSender", return_value=_Sender()),
            patch.object(worker, "_offload_actor_and_empty_cache"),
        ):
            await worker.update_weights(global_steps=2, mode="naive")

        worker.rollout._execute_method.assert_awaited_once()
        ipc_kwargs = worker.rollout._execute_method.await_args.kwargs["kwargs"]
        assert ipc_kwargs["base_sync_done"] is True
        assert ipc_kwargs["peft_config"] == peft_meta
        assert sent["transformer.blocks.0.lora_A.weight"] is lora_tensor
        worker.rollout.update_weights.assert_not_awaited()


class TestCompositeGetPerTensorParamIntegration:
    def test_merge_prefixes_ar_weights_and_peft(self):
        engine = _make_composite_engine(enable_lora=True)
        engine.dit_engine.get_per_tensor_param.return_value = (
            iter([("transformer.w", torch.tensor(1.0))]),
            {"r": 8, "lora_alpha": 16},
        )
        engine.ar_engine.get_per_tensor_param.return_value = (
            iter(
                [
                    ("model.layers.0.weight", torch.tensor(2.0)),
                    ("model.visual.blocks.0.weight", torch.tensor(99.0)),
                ]
            ),
            {"r": 8, "lora_alpha": 16},
        )

        weights, peft = engine.get_per_tensor_param()
        merged = dict(weights)

        assert merged.keys() == {
            "transformer.w",
            "text_encoder.model.layers.0.weight",
            "text_encoder.model.visual.blocks.0.weight",
        }
        assert peft["r"] == 8
        assert peft["lora_alpha"] == 16
        assert peft["exclude_modules"] == ".*visual.*"
