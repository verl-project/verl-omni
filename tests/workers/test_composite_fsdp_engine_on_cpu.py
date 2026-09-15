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
"""CPU-only wiring tests for ``CompositeFSDPEngine`` (Dual-GRPO AR + DiT).

Mirrors ``tests/workers/test_composite_fsdp_engine.py`` without GPU, Ray, or model
weights. Uses ``unittest.mock`` to verify stage switching, delegation, checkpoint
layout, and Dual-GRPO batch-size relationships.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch
from tensordict import TensorDict
from transformers import Qwen2_5_VLConfig
from verl.workers.engine.base import BaseEngine

from verl_omni.workers.config.diffusion.model import DiffusionModelARConfig, DiffusionModelConfig
from verl_omni.workers.engine.fsdp.diffusers_impl import CompositeFSDPEngine, _CompositeEngineCtx

from .test_composite_fsdp_engine import create_ar_infer_batch, create_ar_train_batch

ROLLOUT_N = 2
ROLLOUT_M = 4


def _make_diffusion_model_config(**ar_overrides) -> DiffusionModelConfig:
    cfg = object.__new__(DiffusionModelConfig)
    object.__setattr__(cfg, "text_encoder_subfolder", "text_encoder")
    object.__setattr__(cfg, "trust_remote_code", False)
    object.__setattr__(cfg, "ar", DiffusionModelARConfig(**ar_overrides))
    return cfg


def _make_composite_engine() -> CompositeFSDPEngine:
    engine = CompositeFSDPEngine.__new__(CompositeFSDPEngine)
    engine.ar_engine = MagicMock(name="ar_engine")
    engine.dit_engine = MagicMock(name="dit_engine")
    engine.ar_stage = True
    engine.current_engine = engine.ar_engine
    engine.mode = None
    return engine


def _dual_grpo_batch_sizes(*, device_count: int = 1) -> tuple[int, int]:
    """Return (ar_batch_size, dit_batch_size) matching the GPU integration test."""
    ar_batch_size = ROLLOUT_M * device_count
    dit_batch_size = ar_batch_size * ROLLOUT_N
    return ar_batch_size, dit_batch_size


@contextmanager
def _patch_base_engine_method(method_name: str):
    with patch.object(BaseEngine, method_name) as mock_method:
        yield mock_method


class TestCompositeEngineRegistry:
    def test_engine_registered_for_diffusion_composite_model(self):
        from verl.workers.engine.base import EngineRegistry

        from verl_omni.workers.engine.fsdp import diffusers_impl  # noqa: F401

        engines = EngineRegistry._engines
        assert "diffusion_composite_model" in engines
        composite_registry = engines["diffusion_composite_model"]
        for backend in ("fsdp", "fsdp2"):
            assert backend in composite_registry
            for device in ("cuda", "npu"):
                entry = composite_registry[backend].get(device)
                assert entry is not None
                assert entry.__name__ == "CompositeFSDPEngine"

    def test_engine_workers_lists_composite_model_type(self):
        import inspect

        from verl_omni.workers import engine_workers

        init_model_src = inspect.getsource(engine_workers.ActorRolloutRefWorker.init_model)
        assert "diffusion_composite_model" in init_model_src


class TestBuildARHfModelConfig:
    config = Qwen2_5_VLConfig(architectures=["Qwen2_5_VLForConditionalGeneration"])
    tmp_dir = tempfile.mkdtemp(prefix="composite_fsdp_engine_cpu_")

    def test_points_at_text_encoder_subfolder(self):
        engine = CompositeFSDPEngine.__new__(CompositeFSDPEngine)
        dm_cfg = _make_diffusion_model_config(
            override_config={"attn_implementation": "sdpa"},
            lora_rank=8,
            lora_alpha=16,
        )
        object.__setattr__(dm_cfg, "local_path", self.tmp_dir)
        local_te_dir = os.path.join(self.tmp_dir, "text_encoder")
        self.config.save_pretrained(local_te_dir)
        hf_cfg = engine.build_ar_hf_model_config(dm_cfg)

        assert hf_cfg.path == local_te_dir
        assert hf_cfg.load_tokenizer is False
        assert hf_cfg.override_config == {"attn_implementation": "sdpa"}
        assert hf_cfg.lora_rank == 8
        assert hf_cfg.lora_alpha == 16

    def test_respects_custom_text_encoder_subfolder(self):
        engine = CompositeFSDPEngine.__new__(CompositeFSDPEngine)
        dm_cfg = _make_diffusion_model_config()
        object.__setattr__(dm_cfg, "local_path", self.tmp_dir)
        object.__setattr__(dm_cfg, "text_encoder_subfolder", "custom_te")
        local_te_dir = os.path.join(self.tmp_dir, "custom_te")
        self.config.save_pretrained(local_te_dir)

        hf_cfg = engine.build_ar_hf_model_config(dm_cfg)

        assert hf_cfg.path == local_te_dir


class TestNextStage:
    def test_next_stage_toggles_current_engine(self):
        engine = _make_composite_engine()

        engine.next_stage()
        assert engine.ar_stage is False
        assert engine.current_engine is engine.dit_engine

        engine.next_stage()
        assert engine.ar_stage is True
        assert engine.current_engine is engine.ar_engine

    def test_full_trainer_cycle_returns_to_ar_stage(self):
        engine = _make_composite_engine()
        for _ in range(4):
            engine.next_stage()
        assert engine.ar_stage is True
        assert engine.current_engine is engine.ar_engine


class TestDualGRPOBatchSizes:
    def test_ar_and_dit_batch_sizes_differ(self):
        ar_batch_size, dit_batch_size = _dual_grpo_batch_sizes(device_count=2)
        assert ar_batch_size == ROLLOUT_M * 2
        assert dit_batch_size == ar_batch_size * ROLLOUT_N
        assert ar_batch_size != dit_batch_size

    def test_ar_infer_batch_matches_rollout_m(self):
        ar_batch_size, _ = _dual_grpo_batch_sizes(device_count=1)
        batch = create_ar_infer_batch(ar_batch_size, micro_batch_size_per_gpu=2)
        assert batch.batch_size[0] == ar_batch_size

    def test_dit_infer_batch_matches_rollout_n_times_ar(self):
        ar_batch_size, dit_batch_size = _dual_grpo_batch_sizes(device_count=1)
        model_cfg = _make_diffusion_model_config()
        object.__setattr__(model_cfg, "path", "Qwen/Qwen-Image")
        object.__setattr__(model_cfg, "algorithm", "dual_grpo")

        with patch(
            "tests.workers.test_composite_fsdp_engine.build_scheduler",
            return_value=MagicMock(timesteps=torch.linspace(1, 0, 10)),
        ):
            from .test_composite_fsdp_engine import create_dit_infer_batch

            batch = create_dit_infer_batch(
                dit_batch_size,
                model_cfg,
                micro_batch_size_per_gpu=2,
            )
        assert batch.batch_size[0] == dit_batch_size
        assert batch.batch_size[0] == ar_batch_size * ROLLOUT_N


class TestInferBatchStageSwitching:
    def test_infer_batch_routes_ar_then_dit_like_trainer(self):
        engine = _make_composite_engine()
        ar_batch_size, dit_batch_size = _dual_grpo_batch_sizes(device_count=1)
        ar_data = create_ar_infer_batch(ar_batch_size, micro_batch_size_per_gpu=2)
        dit_data = TensorDict({"old_log_probs": torch.zeros(dit_batch_size, 10)}, batch_size=dit_batch_size)

        with _patch_base_engine_method("infer_batch") as mock_infer:
            mock_infer.side_effect = [{"log_probs": torch.zeros(1)}, {"log_probs": torch.zeros(2)}]

            assert engine.current_engine is engine.ar_engine
            ar_out = engine.infer_batch(ar_data, loss_function=None)
            assert ar_out == {"log_probs": torch.zeros(1)}
            mock_infer.assert_called_once_with(ar_data, None)
            assert engine.current_engine is engine.dit_engine

            mock_infer.reset_mock()
            dit_out = engine.infer_batch(dit_data, loss_function=None)
            assert (dit_out["log_probs"] == torch.zeros(2)).all()
            mock_infer.assert_called_once_with(dit_data, None)
            assert engine.current_engine is engine.ar_engine

    def test_infer_batch_delegates_to_current_engine_via_super(self):
        engine = _make_composite_engine()
        data = TensorDict({"x": torch.zeros(2)}, batch_size=2)
        loss_fn = MagicMock()

        with _patch_base_engine_method("infer_batch") as mock_infer:
            mock_infer.return_value = {"ok": True}
            engine.infer_batch(data, loss_function=loss_fn)

        mock_infer.assert_called_once_with(data, loss_fn)


class TestTrainBatchStageSwitching:
    def test_train_batch_routes_ar_then_dit_like_trainer(self):
        engine = _make_composite_engine()
        ar_batch_size, dit_batch_size = _dual_grpo_batch_sizes(device_count=1)
        ar_data = create_ar_train_batch(ar_batch_size, micro_batch_size_per_gpu=2)
        dit_data = TensorDict(
            {
                "old_log_probs": torch.randn(dit_batch_size, 10),
                "advantages": torch.randn(dit_batch_size, 10),
            },
            batch_size=dit_batch_size,
        )
        loss_fn = MagicMock()

        with _patch_base_engine_method("train_batch") as mock_train:
            mock_train.side_effect = [{"metrics": {"ar": 1.0}}, {"metrics": {"dit": 2.0}}]

            assert engine.current_engine is engine.ar_engine
            ar_out = engine.train_batch(ar_data, loss_fn)
            assert ar_out["metrics"]["ar"] == pytest.approx(1.0)
            mock_train.assert_called_once_with(ar_data, loss_fn)
            assert engine.current_engine is engine.dit_engine

            mock_train.reset_mock()
            dit_out = engine.train_batch(dit_data, loss_fn)
            assert dit_out["metrics"]["dit"] == pytest.approx(2.0)
            mock_train.assert_called_once_with(dit_data, loss_fn)
            assert engine.current_engine is engine.ar_engine


class TestCompositeEngineDelegation:
    def test_forward_backward_batch_delegates_to_current_engine(self):
        engine = _make_composite_engine()
        data = TensorDict({}, batch_size=1)
        loss_fn = MagicMock()
        engine.ar_engine.forward_backward_batch.return_value = ["ar"]

        result = engine.forward_backward_batch(data, loss_fn, forward_only=True)

        engine.ar_engine.forward_backward_batch.assert_called_once_with(data, loss_fn, forward_only=True)
        assert result == ["ar"]

    def test_optimizer_helpers_delegate_to_current_engine(self):
        engine = _make_composite_engine()
        engine.ar_engine.optimizer_step.return_value = 0.5
        engine.ar_engine.lr_scheduler_step.return_value = 1e-4

        engine.optimizer_zero_grad()
        assert engine.ar_engine.optimizer_zero_grad.called

        assert engine.optimizer_step() == pytest.approx(0.5)
        assert engine.lr_scheduler_step() == pytest.approx(1e-4)

    def test_get_data_parallel_helpers_delegate(self):
        engine = _make_composite_engine()
        engine.dit_engine.get_data_parallel_rank.return_value = 1
        engine.current_engine = engine.dit_engine

        assert engine.get_data_parallel_rank() == 1
        engine.dit_engine.get_data_parallel_rank.assert_called_once()

    def test_save_checkpoint_writes_split_subdirs(self):
        engine = _make_composite_engine()

        with (
            patch("torch.distributed.barrier"),
            patch("verl_omni.workers.engine.fsdp.diffusers_impl.aggressive_empty_cache"),
            patch("verl_omni.workers.engine.fsdp.diffusers_impl.gc.collect"),
        ):
            engine.save_checkpoint("/tmp/ckpt", global_step=3)

        engine.dit_engine.save_checkpoint.assert_called_once()
        dit_kwargs = engine.dit_engine.save_checkpoint.call_args.kwargs
        assert dit_kwargs["local_path"] == os.path.join("/tmp/ckpt", "transformer")

        engine.ar_engine.save_checkpoint.assert_called_once()
        ar_kwargs = engine.ar_engine.save_checkpoint.call_args.kwargs
        assert ar_kwargs["local_path"] == os.path.join("/tmp/ckpt", "text_encoder")

    def test_load_checkpoint_reads_split_subdirs(self):
        engine = _make_composite_engine()

        with patch("os.path.isdir", return_value=True), patch("torch.distributed.barrier"):
            engine.load_checkpoint("/tmp/ckpt")

        engine.dit_engine.load_checkpoint.assert_called_once()
        assert engine.dit_engine.load_checkpoint.call_args.kwargs["local_path"] == os.path.join(
            "/tmp/ckpt", "transformer"
        )
        engine.ar_engine.load_checkpoint.assert_called_once()
        assert engine.ar_engine.load_checkpoint.call_args.kwargs["local_path"] == os.path.join(
            "/tmp/ckpt", "text_encoder"
        )

    def test_get_per_tensor_param_prefixes_ar_weights(self):
        engine = _make_composite_engine()
        engine.dit_engine.get_per_tensor_param.return_value = (
            iter([("transformer.weight", torch.tensor(1.0))]),
            {"default": {}},
        )
        engine.ar_engine.get_per_tensor_param.return_value = (
            iter([("lm_head.weight", torch.tensor(2.0))]),
            None,
        )

        params, peft = engine.get_per_tensor_param()
        merged = dict(params)

        assert merged["transformer.weight"] == pytest.approx(1.0)
        assert merged["text_encoder.lm_head.weight"] == pytest.approx(2.0)
        assert peft == {"default": {}}


class TestCompositeEngineCtx:
    def test_train_mode_enters_both_sub_engines(self):
        engine = _make_composite_engine()
        ar_ctx = MagicMock()
        dit_ctx = MagicMock()
        engine.ar_engine.train_mode.return_value = ar_ctx
        engine.dit_engine.train_mode.return_value = dit_ctx

        ctx = _CompositeEngineCtx(engine, mode="train", disable_auto_offload=True)
        with ctx:
            assert engine.mode == "train"
            ar_ctx.__enter__.assert_called_once()
        ar_ctx.__exit__.assert_called_once()
        assert engine.mode is None

        engine.next_stage()
        ctx = _CompositeEngineCtx(engine, mode="train", disable_auto_offload=True)
        with ctx:
            assert engine.mode == "train"
            dit_ctx.__enter__.assert_called_once()
        dit_ctx.__exit__.assert_called_once()
        assert engine.mode is None

    def test_eval_mode_enters_both_sub_engines(self):
        engine = _make_composite_engine()
        ar_ctx = MagicMock()
        dit_ctx = MagicMock()
        engine.ar_engine.eval_mode.return_value = ar_ctx
        engine.dit_engine.eval_mode.return_value = dit_ctx

        ctx = _CompositeEngineCtx(engine, mode="eval")
        with ctx:
            engine.ar_engine.eval_mode.assert_called_once_with()
        ar_ctx.__exit__.assert_called_once()

        engine.next_stage()
        ctx = _CompositeEngineCtx(engine, mode="eval")
        with ctx:
            engine.dit_engine.eval_mode.assert_called_once_with()
        dit_ctx.__exit__.assert_called_once()
