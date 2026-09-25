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

"""Diffusion ETP, SP and VAE configuration through the engine boundary, without GPU workers."""

import os
from argparse import Namespace
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch.nn as nn
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import DistributedVaeMixin
from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelInput
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

from verl_omni.workers.config import DiffusionRolloutConfig
from verl_omni.workers.rollout.vllm_rollout import vllm_omni_async_server as server_module
from verl_omni.workers.rollout.vllm_rollout import vllm_omni_diffusion_strategy as strategy_module


def _prepare(monkeypatch, *, config=None, algorithm="diffusion_nft", typed=1, cli=None, parallel=None, **kwargs):
    monkeypatch.setattr(strategy_module, "import_external_libs", lambda _: None)
    monkeypatch.setattr(strategy_module.VllmOmniPipelineBase, "get_pipeline_path", lambda **_: "test.Pipeline")
    monkeypatch.setattr(
        strategy_module.VllmOmniPipelineBase, "get_class", lambda **_: SimpleNamespace(supports_request_batch=True)
    )
    if config is None:
        config = DiffusionRolloutConfig(tensor_model_parallel_size=4, text_encoder_tp_size=typed)
    server = SimpleNamespace(
        config=config, model_config=SimpleNamespace(architecture="MiniMaxH3Pipeline", algorithm=algorithm)
    )
    args = Namespace(tensor_parallel_size=config.tensor_model_parallel_size, text_encoder_tp_size=cli, **kwargs)
    engine_args = asdict(OmniEngineArgs.from_cli_args(args))
    if parallel is not None:
        engine_args["parallel_config"] = parallel
    strategy_module.DiffusionStrategy(server).prepare_engine_args(engine_args, args)
    return engine_args


def _prepare_parallel(monkeypatch, config, *, algorithm="flow_grpo", nested=None, **kwargs):
    engine_args = _prepare(monkeypatch, config=config, algorithm=algorithm, parallel=nested, **kwargs)
    stage = AsyncOmniEngine._create_default_diffusion_stage_cfg(engine_args)[0]
    return OmniDiffusionConfig(
        parallel_config=stage["engine_args"]["parallel_config"],
        vae_use_tiling=stage["engine_args"]["vae_use_tiling"],
    )


@pytest.mark.parametrize("algorithm", ["diffusion_nft", "flow_grpo"])
@pytest.mark.parametrize("typed,cli,expected", [(1, None, 1), (4, None, 4), (1, 4, 4), (4, 4, 4), (1, 1, 1)])
def test_text_encoder_tp_reaches_fused_diffusion_config(monkeypatch, algorithm, typed, cli, expected):
    engine_args = _prepare(monkeypatch, algorithm=algorithm, typed=typed, cli=cli)
    stages = AsyncOmniEngine._create_default_diffusion_stage_cfg(engine_args)
    parallel = stages[0]["engine_args"]["parallel_config"]
    od_config = OmniDiffusionConfig(parallel_config=parallel)
    assert od_config.parallel_config.tensor_parallel_size == 4
    assert od_config.parallel_config.text_encoder_tp_size == expected


@pytest.mark.parametrize("cli", [0, 2, 8])
def test_legacy_cli_rejects_invalid_encoder_groups(monkeypatch, cli):
    with pytest.raises(ValueError, match="text_encoder_tp_size"):
        _prepare(monkeypatch, cli=cli)


def test_conflicting_typed_and_legacy_etp_fails(monkeypatch):
    with pytest.raises(ValueError, match="Conflicting text_encoder_tp_size"):
        _prepare(monkeypatch, typed=4, cli=1)


@pytest.mark.parametrize("as_object", [False, True])
def test_explicit_parallel_config_keeps_etp_and_other_fields(monkeypatch, as_object):
    parallel = {"tensor_parallel_size": 4, "text_encoder_tp_size": 4, "vae_patch_parallel_size": 4}
    if as_object:
        parallel = DiffusionParallelConfig.from_dict(parallel)
    engine_args = _prepare(monkeypatch, parallel=parallel)
    stages = AsyncOmniEngine._create_default_diffusion_stage_cfg(engine_args)
    od_config = OmniDiffusionConfig(parallel_config=stages[0]["engine_args"]["parallel_config"])
    assert od_config.parallel_config.text_encoder_tp_size == 4
    assert od_config.parallel_config.vae_patch_parallel_size == 4


def test_missing_nested_etp_is_filled_without_mutating_input(monkeypatch):
    parallel = {"tensor_parallel_size": 4, "vae_patch_parallel_size": 4}
    engine_args = _prepare(monkeypatch, typed=4, parallel=parallel)
    stages = AsyncOmniEngine._create_default_diffusion_stage_cfg(engine_args)
    assert stages[0]["engine_args"]["parallel_config"].text_encoder_tp_size == 4
    assert "text_encoder_tp_size" not in parallel


def test_conflicting_parallel_config_fails(monkeypatch):
    with pytest.raises(ValueError, match="Conflicting text_encoder_tp_size"):
        _prepare(monkeypatch, typed=4, parallel={"tensor_parallel_size": 4, "text_encoder_tp_size": 1})


@pytest.mark.asyncio
@pytest.mark.parametrize("algorithm", ["diffusion_nft", "flow_grpo"])
@pytest.mark.parametrize("typed,cli", [(4, None), (1, 4)])
async def test_run_server_preserves_etp_at_engine_boundary(monkeypatch, algorithm, typed, cli):
    monkeypatch.setattr(strategy_module, "import_external_libs", lambda _: None)
    monkeypatch.setattr(strategy_module.VllmOmniPipelineBase, "get_pipeline_path", lambda **_: None)
    monkeypatch.setattr(server_module, "get_non_ephemeral_free_port", lambda *_: 12345)
    monkeypatch.setattr(os, "environ", dict(os.environ))
    server = SimpleNamespace(
        config=DiffusionRolloutConfig(
            tensor_model_parallel_size=4,
            text_encoder_tp_size=typed,
            vae_patch_parallel_size=4,
            vae_use_tiling=True,
        ),
        model_config=SimpleNamespace(architecture="MiniMaxH3Pipeline", algorithm=algorithm),
    )
    server._generate_strategy = strategy_module.DiffusionStrategy(server)
    captured = {}

    class EngineBoundaryReached(Exception):
        pass

    def capture_engine(**kwargs):
        captured.update(kwargs)
        raise EngineBoundaryReached

    monkeypatch.setattr(server_module, "AsyncOmni", capture_engine)
    with pytest.raises(EngineBoundaryReached):
        await server_module.vLLMOmniHttpServer.run_server(
            server, Namespace(tensor_parallel_size=4, text_encoder_tp_size=cli)
        )
    stages = AsyncOmniEngine._create_default_diffusion_stage_cfg(captured)
    od_config = OmniDiffusionConfig(
        parallel_config=stages[0]["engine_args"]["parallel_config"],
        vae_use_tiling=stages[0]["engine_args"]["vae_use_tiling"],
    )
    assert od_config.parallel_config.tensor_parallel_size == 4
    assert od_config.parallel_config.text_encoder_tp_size == 4
    assert od_config.parallel_config.vae_patch_parallel_size == 4
    assert od_config.vae_use_tiling is True


@pytest.mark.parametrize("algorithm", ["flow_grpo", "diffusion_nft"])
@pytest.mark.parametrize("nested", [None, {"tensor_parallel_size": 2}])
def test_vae_parallel_reaches_od_config(monkeypatch, algorithm, nested):
    config = DiffusionRolloutConfig(
        tensor_model_parallel_size=2, vae_patch_parallel_size=2, vae_parallel_mode="tile", vae_use_tiling=True
    )
    od = _prepare_parallel(monkeypatch, config, algorithm=algorithm, nested=nested)
    assert od.parallel_config.tensor_parallel_size == 2
    assert od.parallel_config.vae_patch_parallel_size == 2
    assert od.parallel_config.vae_parallel_mode == "tile"
    assert od.vae_use_tiling is True
    if nested is not None:
        assert nested == {"tensor_parallel_size": 2}


@pytest.mark.parametrize("field", ["ulysses_degree", "ring_degree", "vae_patch_parallel_size"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_parallel_degrees_reject_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        DiffusionRolloutConfig(**{field: value})


def test_vae_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="vae_parallel_mode"):
        DiffusionRolloutConfig(vae_parallel_mode="unknown")


def test_default_parallelism_is_unchanged(monkeypatch):
    od = _prepare_parallel(monkeypatch, DiffusionRolloutConfig(tensor_model_parallel_size=2))
    assert od.parallel_config.world_size == 2
    assert od.parallel_config.ulysses_degree == 1
    assert od.parallel_config.ring_degree == 1
    assert od.parallel_config.vae_patch_parallel_size == 1
    assert od.vae_use_tiling is False


def test_nested_vae_conflict_is_not_silently_ignored(monkeypatch):
    config = DiffusionRolloutConfig(tensor_model_parallel_size=2, vae_patch_parallel_size=2)
    with pytest.raises(ValueError, match="Conflicting.*vae_patch_parallel_size"):
        _prepare_parallel(monkeypatch, config, nested={"vae_patch_parallel_size": 1})


def test_legacy_sp_cannot_bypass_resource_allocation(monkeypatch):
    with pytest.raises(ValueError, match="ulysses_degree"):
        _prepare_parallel(monkeypatch, DiffusionRolloutConfig(), ulysses_degree=2)


def test_nested_sp_cannot_bypass_resource_allocation(monkeypatch):
    with pytest.raises(ValueError, match="ring_degree"):
        _prepare_parallel(monkeypatch, DiffusionRolloutConfig(), nested={"ring_degree": 2})


@pytest.mark.parametrize("algorithm", ["flow_grpo", "diffusion_nft"])
@pytest.mark.parametrize("nested", [False, True])
def test_pure_ulysses_and_shared_encoder_vae_group(monkeypatch, algorithm, nested):
    config = DiffusionRolloutConfig(
        name="vllm_omni",
        tensor_model_parallel_size=1,
        ulysses_degree=4,
        text_encoder_tp_size=4,
        vae_patch_parallel_size=4,
        vae_use_tiling=True,
    )
    od = _prepare_parallel(
        monkeypatch, config, algorithm=algorithm, nested={"tensor_parallel_size": 1} if nested else None
    )
    assert od.parallel_config.tensor_parallel_size == 1
    assert od.parallel_config.sequence_parallel_size == 4
    assert od.parallel_config.world_size == 4
    assert od.parallel_config.text_encoder_tp_size == 4
    assert od.parallel_config.vae_patch_parallel_size == 4


@pytest.mark.parametrize("key,field", [("usp", "ulysses_degree"), ("ring-degree", "ring_degree")])
def test_explicit_one_cannot_override_allocated_sp(monkeypatch, key, field):
    config = DiffusionRolloutConfig(
        name="vllm_omni",
        tensor_model_parallel_size=1,
        engine_kwargs={"vllm_omni": {key: 1}},
        **{field: 4},
    )
    with pytest.raises(ValueError, match=field):
        _prepare_parallel(monkeypatch, config, **{field: 1})


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"ulysses_degree": 2, "ring_degree": 2}, "hybrid"),
        ({"vae_patch_parallel_size": 2}, "full DiT group"),
        ({"vae_parallel_mode": "spatial_shard_height"}, "tile only"),
        ({"cfg_parallel_size": 2}, "cfg_parallel_size"),
        ({"text_encoder_tp_size": 2}, "text_encoder_tp_size"),
    ],
)
def test_h3_rejects_unsupported_groups(overrides, error):
    from vllm_omni.diffusion.data import DiffusionParallelConfig

    from verl_omni.pipelines.minimax_h3_diffusion_nft.common import validate_h3_parallel_config

    with pytest.raises(ValueError, match=error):
        validate_h3_parallel_config(DiffusionParallelConfig(tensor_parallel_size=4, **overrides))


@pytest.mark.parametrize("tp,usp,vae,etp", [(4, 1, 4, 4), (1, 4, 4, 4), (2, 2, 1, 1)])
def test_h3_accepts_group_reuse(tp, usp, vae, etp):
    from vllm_omni.diffusion.data import DiffusionParallelConfig

    from verl_omni.pipelines.minimax_h3_diffusion_nft.common import validate_h3_parallel_config

    validate_h3_parallel_config(
        DiffusionParallelConfig(
            tensor_parallel_size=tp,
            ulysses_degree=usp,
            vae_patch_parallel_size=vae,
            text_encoder_tp_size=etp,
        )
    )


@pytest.mark.parametrize("key", ["vae_use_tiling", "vae-use-tiling"])
def test_explicit_legacy_tiling_conflict(monkeypatch, key):
    config = DiffusionRolloutConfig(vae_use_tiling=True, engine_kwargs={"vllm_omni": {key: False}})
    with pytest.raises(ValueError, match="Conflicting vae_use_tiling"):
        _prepare_parallel(monkeypatch, config, vae_use_tiling=False)


@pytest.mark.parametrize("algorithm", ["flow_grpo", "diffusion_nft"])
def test_h3_parallel_validation_precedes_weight_loading(monkeypatch, algorithm):
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    import verl_omni.pipelines  # noqa: F401

    def unexpected_load(*args, **kwargs):
        pytest.fail("Invalid H3 topology reached model loading")

    monkeypatch.setattr(MiniMaxH3Pipeline, "__init__", unexpected_load)
    pipeline_cls = strategy_module.VllmOmniPipelineBase.get_class("MiniMaxH3Pipeline", algorithm)
    parallel = DiffusionParallelConfig(tensor_parallel_size=4, vae_patch_parallel_size=2)
    with pytest.raises(ValueError, match="vae_patch_parallel_size"):
        pipeline_cls(od_config=SimpleNamespace(parallel_config=parallel))


class _SPDiT(nn.Module):
    _sp_plan = {"sp_prepare": {0: SequenceParallelInput(split_dim=0, expected_dims=2, split_output=True)}}

    def __init__(self):
        super().__init__()
        self.sp_prepare = nn.Identity()


class _ParallelVAE(DistributedVaeMixin):
    def __init__(self):
        self.use_tiling = False
        self.set_parallel_size = Mock()


class _SPPipeline(nn.Module):
    _dit_modules = ["transformer", "transformer_2", "transformers_ref"]

    def __init__(self):
        super().__init__()
        self.transformer = _SPDiT()
        self.transformer_2 = None
        self.transformers_ref = _SPDiT()
        self.vae = _ParallelVAE()


def _parallel_setup(pipeline, **parallel):
    from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context

    from verl_omni.pipelines.model_base import apply_rollout_parallel_setup

    od_config = OmniDiffusionConfig(parallel_config=DiffusionParallelConfig(**parallel))
    with set_forward_context(omni_diffusion_config=od_config):
        apply_rollout_parallel_setup(pipeline, od_config)
        return od_config, get_forward_context().sp_plan_hooks_applied


def test_parallel_setup_installs_sp_hooks_and_vae_parallel():
    pipeline = _SPPipeline()
    od_config, hooks_applied = _parallel_setup(pipeline, ulysses_degree=2, vae_patch_parallel_size=2)
    assert hooks_applied is True
    for dit in (pipeline.transformer, pipeline.transformers_ref):
        assert dit.sp_prepare._hook_registry.get_hook("sp_input---sp_prepare") is not None
    pipeline.vae.set_parallel_size.assert_called_once_with(2, mode="tile")
    assert od_config.vae_use_tiling is True and pipeline.vae.use_tiling is True


def test_parallel_setup_keeps_defaults_unchanged():
    pipeline = _SPPipeline()
    _, hooks_applied = _parallel_setup(pipeline, tensor_parallel_size=2)
    assert hooks_applied is False
    assert getattr(pipeline.transformer.sp_prepare, "_hook_registry", None) is None
    pipeline.vae.set_parallel_size.assert_not_called()
    assert pipeline.vae.use_tiling is False


@pytest.mark.parametrize(
    ("component", "replacement", "parallel", "error"),
    [
        ("transformers_ref", nn.Linear(1, 1), {"ulysses_degree": 2}, "sequence parallelism"),
        ("vae", SimpleNamespace(use_tiling=False), {"vae_patch_parallel_size": 2}, "vae_patch_parallel_size"),
    ],
)
def test_parallel_setup_rejects_unsupported_components(component, replacement, parallel, error):
    pipeline = _SPPipeline()
    setattr(pipeline, component, replacement)
    with pytest.raises(ValueError, match=error):
        _parallel_setup(pipeline, **parallel)
    assert getattr(pipeline.transformer.sp_prepare, "_hook_registry", None) is None


def test_parallel_setup_fails_closed_without_sp_hooks(monkeypatch):
    from vllm_omni.diffusion import registry

    monkeypatch.setattr(registry, "_apply_sequence_parallel_if_enabled", lambda *_: None)
    with pytest.raises(RuntimeError, match="hooks were not applied to _SPPipeline.transformer"):
        _parallel_setup(_SPPipeline(), ulysses_degree=2)


def test_registered_pipelines_apply_parallel_setup_once_after_construction(monkeypatch):
    import verl_omni.pipelines  # noqa: F401
    from verl_omni.pipelines import model_base

    assert all(
        "__wrapped__" in vars(vars(cls)["__init__"]) for cls in model_base.VllmOmniPipelineBase._registry.values()
    )
    monkeypatch.setattr(model_base.VllmOmniPipelineBase, "_registry", {})
    events = []
    monkeypatch.setattr(model_base, "apply_rollout_parallel_setup", lambda pipeline, _: events.append("setup"))

    @model_base.VllmOmniPipelineBase.register("Toy", algorithm="flow_grpo")
    class Parent:
        def __init__(self, *, od_config):
            events.append("parent")

    @model_base.VllmOmniPipelineBase.register("Toy", algorithm="dual_grpo")
    class Child(Parent):
        def __init__(self, *, od_config):
            super().__init__(od_config=od_config)
            events.append("child")

    Parent(od_config=None)
    assert events == ["parent", "setup"]
    events.clear()
    Child(od_config=None)
    assert events == ["parent", "child", "setup"]


@pytest.mark.parametrize("algorithm", ["flow_grpo", "diffusion_nft"])
def test_h3_adapters_receive_parallel_setup(monkeypatch, algorithm):
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    import verl_omni.pipelines  # noqa: F401

    def fake_init(self, **_):
        nn.Module.__init__(self)
        self._dit_modules = ["transformer"]
        self.transformer = _SPDiT()
        self.vae = _ParallelVAE()

    monkeypatch.setattr(MiniMaxH3Pipeline, "__init__", fake_init)
    pipeline_cls = strategy_module.VllmOmniPipelineBase.get_class("MiniMaxH3Pipeline", algorithm)
    for method in ("_install_lora_layout", "install_h3_lora_layout"):
        if hasattr(pipeline_cls, method):
            monkeypatch.setattr(pipeline_cls, method, lambda self: None)
    parallel = DiffusionParallelConfig(ulysses_degree=2, vae_patch_parallel_size=2)
    od_config = OmniDiffusionConfig(parallel_config=parallel)
    with set_forward_context(omni_diffusion_config=od_config):
        pipeline = pipeline_cls(od_config=od_config)
    assert pipeline.transformer.sp_prepare._hook_registry.get_hook("sp_input---sp_prepare") is not None
    pipeline.vae.set_parallel_size.assert_called_once_with(2, mode="tile")


def test_hydra_exposes_parallel_fields_without_plus():
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from verl.utils.config import omega_conf_to_dataclass

    config_dir = Path(__file__).resolve().parents[4] / "verl_omni/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="diffusion_trainer",
            overrides=[
                "actor_rollout_ref.rollout.name=vllm_omni",
                "actor_rollout_ref.rollout.tensor_model_parallel_size=4",
                "actor_rollout_ref.rollout.vae_patch_parallel_size=4",
                "actor_rollout_ref.rollout.vae_use_tiling=true",
            ],
        )
    rollout = omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout)
    assert rollout.vae_patch_parallel_size == 4
    assert rollout.vae_use_tiling is True
    assert rollout.ulysses_degree == rollout.ring_degree == 1
