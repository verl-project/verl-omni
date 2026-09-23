# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Catch FSDP dispatch and stale V0 config keys before allocating Megatron GPUs."""

import inspect
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from omegaconf import OmegaConf
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import McoreActorConfig, McoreEngineConfig

from verl_omni.trainer.omni.ray_omni_trainer_separate_async import OmniPPOTrainerSeparateAsync
from verl_omni.workers.omni_engine_workers import OmniDetachActorWorker

REPO_ROOT = Path(__file__).parents[3]
LAUNCHER = REPO_ROOT / "examples/gspo_trainer/qwen3_omni/run_qwen3_omni_megatron_audiomcq_separate_async.sh"


@pytest.fixture(scope="module")
def public_recipe_config(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("audiomcq-config") / "artifacts"
    env = os.environ.copy()
    env.update(
        {
            "AUDIO_MCQ_CONFIG_ONLY": "1",
            "MODEL_PATH": "/tmp/model",
            "OUTPUT_DIR": str(output_dir),
            "PYTHON": sys.executable,
            "TRAIN_FILE": "/tmp/train.parquet",
            "VAL_FILE": "/tmp/validation.parquet",
        }
    )
    # The real Hydra entry point may take over a minute to import on CPU CI.
    # Own the entire process group so a timeout cannot orphan shell/tee children.
    process = subprocess.Popen(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        pytest.fail(f"AudioMCQ config-only launcher timed out; stdout={stdout[-2000:]}, stderr={stderr[-2000:]}")
    assert process.returncode == 0, f"AudioMCQ config-only launcher failed: {stderr[-2000:]}"
    [config_path] = output_dir.glob("run.*/config.yaml")
    return config_path


def test_public_recipe_selects_megatron_and_v1_separate_async(public_recipe_config):
    config = OmegaConf.load(public_recipe_config)
    command = (public_recipe_config.parent / "command.txt").read_text()
    assert "--config-name omni_megatron_trainer" in command
    assert "audiomcq_megatron_separate_async" not in command
    actor = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    assert isinstance(actor, McoreActorConfig)
    assert isinstance(actor.engine, McoreEngineConfig)
    assert actor.engine.tensor_model_parallel_size == 4
    assert actor.engine.expert_model_parallel_size == 4
    assert actor.engine.expert_tensor_parallel_size == 1
    assert actor.engine.pipeline_model_parallel_size == 1
    transformer_config = actor.engine.override_transformer_config
    assert transformer_config["gradient_accumulation_fusion"] is False
    assert transformer_config["freeze_language_model"] is False
    assert transformer_config["freeze_vision_model"] is True
    assert transformer_config["freeze_audio_model"] is True
    assert config.actor_rollout_ref.ref.megatron.override_transformer_config.gradient_accumulation_fusion is False
    assert config.actor_rollout_ref.model.model_type == "omni_model"
    assert not config.actor_rollout_ref.model.use_remove_padding
    assert not actor.engine.use_remove_padding
    assert not actor.use_dynamic_bsz
    assert not config.actor_rollout_ref.ref.log_prob_use_dynamic_bsz
    assert not config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz
    assert config.actor_rollout_ref.model.lora_rank == 0
    assert config.actor_rollout_ref.model.lora.rank == 0
    assert config.actor_rollout_ref.rollout.engine_kwargs.vllm_omni.limit_mm_per_prompt == {
        "audio": 1,
        "image": 1,
        "video": 0,
    }
    assert config.actor_rollout_ref.rollout.nnodes == 4
    assert config.actor_rollout_ref.rollout.n_gpus_per_node == 4
    assert config.actor_rollout_ref.rollout.tensor_model_parallel_size == 4
    assert config.actor_rollout_ref.rollout.n == 8
    assert config.trainer.v1.trainer_mode == "omni_separate_async"
    assert config.trainer.nnodes == 4
    assert config.trainer.n_gpus_per_node == 4
    assert config.trainer.total_training_steps == 150
    assert config.trainer.test_freq == 10
    assert config.reward.custom_reward_function.path == "pkg://verl_omni.utils.reward_score.audio_mcq"
    assert config.reward.custom_reward_function.name == "compute_score"
    trainer = OmniPPOTrainerSeparateAsync(config)
    assert trainer.parameter_sync_step == 1
    assert config.data.train_batch_size == trainer.parameter_sync_step * actor.ppo_mini_batch_size
    assert not config.algorithm.rollout_correction.bypass_mode
    env = OmegaConf.to_container(config.ray_kwargs.ray_init.runtime_env.env_vars, resolve=True)
    assert env["TENSORBOARD_DIR"] == str(public_recipe_config.parent / "tensorboard")
    assert env["VERL_USE_EXTERNAL_MODULES"] == "verl_omni"
    assert all(isinstance(value, str) for value in env.values())


def test_megatron_detach_preserves_shard_list_protocol():
    worker = object.__new__(OmniDetachActorWorker)
    worker._strategy_handlers = None
    modules = [object(), object()]
    worker.actor = SimpleNamespace(engine=SimpleNamespace(module=modules))
    worker.config = SimpleNamespace(actor=SimpleNamespace(strategy="megatron"))
    snapshots = [object(), object()]
    # Exercise the real strategy dispatcher without requiring Megatron/CUDA in
    # CPU CI. Native tensor copies are covered by the Megatron GPU smoke.
    helpers = ModuleType("verl.utils.megatron_utils")
    helpers.copy_megatron_model_to_cpu = save = Mock(return_value=snapshots)
    helpers.restore_megatron_model_from_cpu = restore = Mock()
    with patch.dict(sys.modules, {helpers.__name__: helpers}):
        worker.save_model_to_cpu(1)
        worker.restore_model_from_cpu(1)
        worker.clear_cpu_model(1)
    save.assert_called_once_with(modules)
    restore.assert_called_once_with(modules, snapshots)
    assert not worker.cpu_saved_models


def test_megatron_worker_initialization_selects_ppo_loss(monkeypatch, public_recipe_config):
    import verl_omni.workers.engine_workers as workers

    config = OmegaConf.load(public_recipe_config)
    worker = object.__new__(workers.ActorRolloutRefWorker)
    worker.config = config.actor_rollout_ref
    worker.role = "actor"
    worker.distillation_enabled = False
    model_config = OmegaConf.create({"model_type": "omni_model", "use_remove_padding": False})
    original_convert = workers.omega_conf_to_dataclass
    monkeypatch.setattr(
        workers,
        "omega_conf_to_dataclass",
        lambda value: model_config if value is worker.config.model else original_convert(value),
    )

    class EngineBoundaryReached(Exception):
        pass

    def build_engine(config):
        assert isinstance(config.engine_config, McoreEngineConfig)
        assert config.model_type == "omni_model"
        raise EngineBoundaryReached

    monkeypatch.setattr(workers, "TrainingWorker", build_engine)
    with pytest.raises(EngineBoundaryReached):
        inspect.unwrap(workers.ActorRolloutRefWorker.init_model)(worker)
    assert worker.loss_fn.func is workers.ppo_loss
    assert model_config.trainer_type == "policy_gradient"


def test_native_megatron_adapter_dispatch_and_forward_binding(monkeypatch):
    from verl_omni.workers.engine import OmniMegatronEngine

    if OmniMegatronEngine is None:
        pytest.skip("Megatron is an optional dependency in CPU CI")
    from verl.models.mcore import util as mcore_util
    from verl.workers.engine.base import EngineRegistry
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

    from verl_omni.pipelines.qwen3_omni.thinker_training_adapter import Qwen3OmniThinkerAdapter
    from verl_omni.workers.engine.megatron import omni_impl

    monkeypatch.setenv("VERL_ENGINE_DEVICE", "cuda")
    monkeypatch.setattr(omni_impl, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(mcore_util.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(mcore_util.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(mcore_util.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    assert EngineRegistry.get_engine_cls("omni_model", "megatron") is OmniMegatronEngine
    assert issubclass(OmniMegatronEngine, MegatronEngineWithLMHead)
    assert OmniMegatronEngine.optimizer_step is MegatronEngineWithLMHead.optimizer_step
    assert OmniMegatronEngine.get_per_tensor_param is MegatronEngineWithLMHead.get_per_tensor_param

    class Model(torch.nn.Module):
        pre_process = True
        post_process = False
        config = SimpleNamespace(fp8=None)

        def forward(self, **kwargs):
            self.seen = kwargs
            return kwargs["input_features"].sum()

    features = torch.ones(1, 128, 5, requires_grad=True)
    ids = torch.nested.nested_tensor([torch.tensor([1, 2, 3, 4])], layout=torch.jagged)
    loss_mask = torch.nested.nested_tensor([torch.ones(4, dtype=torch.bool)], layout=torch.jagged)
    events = []

    class Batch(dict):
        def to(self, device):
            assert device == "cpu"
            events.append("batch moved")
            return self

    batch = Batch(input_ids=ids, loss_mask=loss_mask, temperature=torch.tensor([1.0]))

    def upstream_prepare(_engine, batch):
        assert events == ["batch moved"]
        events.append("inputs prepared")
        if getattr(_engine, "fail_after_prepare", False):
            raise RuntimeError("forward failed after input preparation")
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": None,
            "loss_mask": batch["loss_mask"],
            "multi_modal_inputs": {"input_features": features},
        }

    monkeypatch.setattr(MegatronEngineWithLMHead, "prepare_model_inputs", upstream_prepare)
    model = Model()
    engine = object.__new__(OmniMegatronEngine)
    engine.model_forward = Qwen3OmniThinkerAdapter.get_megatron_forward()
    engine.engine_config = SimpleNamespace(use_fused_kernels=False)
    engine.model_config = SimpleNamespace(hf_config=SimpleNamespace(), tokenizer=SimpleNamespace(pad_token_id=0))
    output, postprocess = engine.forward_step(iter([batch]), model, None, lambda _, **kwargs: kwargs)
    output.backward()
    assert postprocess(None) == {"data": batch, "local_cp_size": None}
    assert events == ["batch moved", "inputs prepared"]
    assert model.seen["position_ids"] is None
    assert torch.equal(features.grad, torch.ones_like(features))
    assert not model._forward_pre_hooks
    assert not hasattr(engine, "_forward_model")
    assert not hasattr(engine, "_input_adapters")

    events.clear()
    engine.fail_after_prepare = True
    with pytest.raises(RuntimeError, match="forward failed"):
        engine.forward_step(iter([batch]), model, None, None)
    assert events == ["batch moved", "inputs prepared"]
    assert not model._forward_pre_hooks
    assert not hasattr(engine, "_forward_model")
    assert not hasattr(engine, "_input_adapters")


def test_native_megatron_adapter_rejects_mtp():
    from verl_omni.workers.engine import OmniMegatronEngine

    if OmniMegatronEngine is None:
        pytest.skip("Megatron is an optional dependency in CPU CI")
    from transformers import Qwen3OmniMoeConfig

    model_config = SimpleNamespace(
        architecture="Qwen3OmniMoeForConditionalGeneration",
        hf_config=Qwen3OmniMoeConfig(),
        model_stage="thinker",
        mtp=SimpleNamespace(enable=True),
    )
    engine_config = SimpleNamespace(
        use_remove_padding=False,
        use_fused_kernels=False,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
    )

    with pytest.raises(ValueError, match="does not support MTP"):
        OmniMegatronEngine(model_config, engine_config, None, None)


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("dynamic_context_parallel", True, "dynamic CP"),
        ("router_replay", SimpleNamespace(mode="R2"), "router replay"),
        ("use_remove_padding", True, "use_remove_padding=false"),
        ("use_fused_kernels", True, "use_fused_kernels=false"),
        ("pipeline_model_parallel_size", 2, "PP=CP=1"),
        ("context_parallel_size", 2, "PP=CP=1"),
    ],
)
def test_native_megatron_adapter_fails_closed_on_unsupported_modes(setting, value, message):
    from verl_omni.workers.engine import OmniMegatronEngine

    if OmniMegatronEngine is None:
        pytest.skip("Megatron is an optional dependency in CPU CI")
    from transformers import Qwen3OmniMoeConfig

    model_config = SimpleNamespace(
        architecture="Qwen3OmniMoeForConditionalGeneration", hf_config=Qwen3OmniMoeConfig(), model_stage="thinker"
    )
    engine_config = SimpleNamespace(
        use_remove_padding=False,
        use_fused_kernels=False,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        dynamic_context_parallel=False,
        router_replay=SimpleNamespace(mode="disabled"),
    )
    setattr(engine_config, setting, value)
    with pytest.raises(ValueError, match=message):
        OmniMegatronEngine(model_config, engine_config, None, None)


def test_native_megatron_config_view_does_not_mutate_rollout_config(monkeypatch):
    from verl_omni.workers.engine import OmniMegatronEngine

    if OmniMegatronEngine is None:
        pytest.skip("Megatron is an optional dependency in CPU CI")
    from transformers import Qwen3OmniMoeConfig
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

    from verl_omni.pipelines.qwen3_omni.thinker_training_adapter import Qwen3OmniThinkerAdapter

    model_config = SimpleNamespace(
        architecture="Qwen3OmniMoeForConditionalGeneration", hf_config=Qwen3OmniMoeConfig(), model_stage="thinker"
    )
    engine_config = SimpleNamespace(
        use_remove_padding=False,
        use_fused_kernels=False,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
    )

    def parent_init(engine, model_config, *_args):
        engine.model_config = model_config

    monkeypatch.setattr(MegatronEngineWithLMHead, "__init__", parent_init)
    engine = OmniMegatronEngine(model_config, engine_config, None, None)
    assert engine.model_adapter_cls is Qwen3OmniThinkerAdapter
    assert engine.model_forward is Qwen3OmniThinkerAdapter.get_megatron_forward()
    assert not hasattr(engine, "_forward_model")
    assert not hasattr(engine, "_input_adapters")
    assert engine.model_config is not model_config
    assert engine.model_config.hf_config is not model_config.hf_config
    assert (
        engine.model_config.hf_config.text_config.hidden_size
        == model_config.hf_config.thinker_config.text_config.hidden_size
    )
    engine.model_config.hf_config.text_config.hidden_size = 128
    assert not hasattr(model_config.hf_config, "text_config")
    assert model_config.hf_config.thinker_config.text_config.hidden_size != 128


def test_native_megatron_uses_registered_pipeline_without_qwen_config(monkeypatch):
    from verl_omni.workers.engine import OmniMegatronEngine

    if OmniMegatronEngine is None:
        pytest.skip("Megatron is an optional dependency in CPU CI")
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

    from verl_omni.pipelines.model_base import OmniModelBase

    model_config = SimpleNamespace(architecture="TestOmni", model_stage="thinker")
    engine_config = object()
    prepared_config = object()
    events = []

    def model_forward(*args, **kwargs):
        return args, kwargs

    class Adapter(OmniModelBase):
        @classmethod
        def prepare_megatron_config(cls, config, engine):
            assert config is model_config
            assert engine is engine_config
            events.append("config prepared")
            return prepared_config

        @classmethod
        def get_megatron_forward(cls):
            events.append("forward selected")
            return model_forward

    def parent_init(engine, config, engine_cfg, *_args):
        assert config is prepared_config
        assert engine_cfg is engine_config
        events.append("parent initialized")

    monkeypatch.setitem(OmniModelBase._registry, ("TestOmni", "thinker"), Adapter)
    monkeypatch.setattr(MegatronEngineWithLMHead, "__init__", parent_init)
    engine = OmniMegatronEngine(model_config, engine_config, None, None)
    assert events == ["config prepared", "forward selected", "parent initialized"]
    assert engine.model_adapter_cls is Adapter
    assert engine.model_forward is model_forward


def test_native_megatron_rejects_adapter_without_backend_support(monkeypatch):
    from verl_omni.workers.engine import OmniMegatronEngine

    if OmniMegatronEngine is None:
        pytest.skip("Megatron is an optional dependency in CPU CI")
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

    from verl_omni.pipelines.model_base import OmniModelBase

    class FsdpOnlyAdapter(OmniModelBase):
        pass

    def parent_init(*_args):
        pytest.fail("Unsupported adapters must fail before Megatron initialization")

    monkeypatch.setitem(OmniModelBase._registry, ("FsdpOnlyOmni", "thinker"), FsdpOnlyAdapter)
    monkeypatch.setattr(MegatronEngineWithLMHead, "__init__", parent_init)
    model_config = SimpleNamespace(architecture="FsdpOnlyOmni", model_stage="thinker")
    with pytest.raises(NotImplementedError, match="FsdpOnlyAdapter does not support Megatron training"):
        OmniMegatronEngine(model_config, None, None, None)
    with pytest.raises(NotImplementedError, match="does not provide a Megatron forward"):
        FsdpOnlyAdapter.get_megatron_forward()
