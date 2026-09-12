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
"""CPU transition tests; distributed shard semantics are exercised separately on GPUs."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from verl.workers.config import FSDPEngineConfig

from verl_omni.workers import detach_actor_worker as snapshots


class _ShardParameter(torch.nn.Parameter):
    device_mesh = "mesh"
    placements = ("shard-0",)

    def to_local(self):
        return self.detach()


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setattr(snapshots, "DTensor", _ShardParameter)
    monkeypatch.setattr(snapshots, "get_device_name", lambda: "cpu")
    model = torch.nn.Module()
    model.base = torch.nn.Parameter(torch.arange(32.0), requires_grad=False)
    model.adapter = _ShardParameter(torch.tensor([1.0, 2.0]))
    model.extra = torch.nn.Parameter(torch.tensor([3.0]))
    model.tied = model.adapter
    model.peft_config = {"default": object()}
    transitions = []
    calls = []

    def move(device, *, model, optimizer, grad):
        transitions.append((device, model, optimizer, grad))

    def save(view):
        calls.append(tuple(id(param) for param in view.parameters()))
        return {name: param.detach() for name, param in view.named_parameters()}, "spec"

    def restore(view, state, _spec=None):
        if _spec is None:
            state, _spec = state
        with torch.no_grad():
            for name, param in view.named_parameters():
                param.copy_(state[name])

    instance = object.__new__(snapshots.DiffusionDetachActorWorker)
    instance.config = OmegaConf.create({"actor": {"strategy": "fsdp2"}})
    instance.actor = SimpleNamespace(
        engine=SimpleNamespace(
            module=model,
            model_config=SimpleNamespace(
                architecture="QwenImagePipeline", lora_rank=2, policy_state_adapters=("default",)
            ),
            engine_config=FSDPEngineConfig(strategy="fsdp2", ulysses_sequence_parallel_size=1),
            is_param_offload_enabled=False,
            _uses_fsdp2_cpu_offload_policy=False,
            to=move,
        )
    )
    instance._strategy_handlers = (save, restore)
    instance.cpu_saved_models = {}
    instance.test_calls = calls
    instance.test_transitions = transitions
    return instance


@pytest.mark.parametrize("offload", [False, True])
def test_snapshot_is_trainable_only_independent_and_preserves_parameter_identity(worker, offload):
    engine = worker.actor.engine
    engine.is_param_offload_enabled = offload
    model = engine.module
    parameters = tuple(model.parameters())
    optimizer = torch.optim.SGD(parameters, lr=0.1)
    rng = torch.get_rng_state().clone()
    worker.save_model_to_cpu(0)
    snapshot = worker.cpu_saved_models[0]
    assert isinstance(snapshot, snapshots._TrainableSnapshot)
    assert snapshot.names == ("adapter", "extra")
    assert worker.test_calls == [(id(model.adapter), id(model.extra))]
    assert sum(t.numel() for t in snapshot.state[0].values()) == 3
    with torch.no_grad():
        model.adapter.add_(10)
        model.extra.add_(20)
        model.base.add_(30)
    model.adapter.grad = torch.ones_like(model.adapter)
    worker.restore_model_from_cpu(0)
    torch.testing.assert_close(model.adapter, torch.tensor([1.0, 2.0]), rtol=0, atol=0)
    torch.testing.assert_close(model.extra, torch.tensor([3.0]), rtol=0, atol=0)
    torch.testing.assert_close(model.base, torch.arange(32.0) + 30, rtol=0, atol=0)
    torch.testing.assert_close(model.adapter.grad, torch.ones(2), rtol=0, atol=0)
    assert all(a is b for a, b in zip(parameters, model.parameters(), strict=True))
    assert all(a is b for a, b in zip(parameters, optimizer.param_groups[0]["params"], strict=True))
    assert model.tied is model.adapter
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert worker.test_transitions == [("cpu", True, False, False)] * (4 if offload else 0)
    worker.clear_cpu_model(0)
    worker.clear_cpu_model(0)
    assert worker.cpu_saved_models == {}


@pytest.mark.parametrize(
    "unsupported",
    [
        "fsdp",
        "veomni",
        "architecture",
        "no_lora",
        "multi_adapter",
        "module_adapter",
        "sp",
        "native_offload",
        "full",
        "empty",
        "unsharded",
    ],
)
def test_unsupported_combinations_keep_full_snapshot(worker, unsupported):
    engine = worker.actor.engine
    model = engine.module
    if unsupported in ("fsdp", "veomni"):
        worker.config.actor.strategy = unsupported
    elif unsupported == "architecture":
        engine.model_config.architecture = "OtherPipeline"
    elif unsupported == "no_lora":
        engine.model_config.lora_rank = 0
    elif unsupported == "multi_adapter":
        engine.model_config.policy_state_adapters = ("default", "old")
    elif unsupported == "module_adapter":
        model.peft_config["old"] = object()
    elif unsupported == "sp":
        engine.engine_config.ulysses_sequence_parallel_size = 2
    elif unsupported == "native_offload":
        engine._uses_fsdp2_cpu_offload_policy = True
    elif unsupported == "full":
        model.requires_grad_(True)
    elif unsupported == "empty":
        model.requires_grad_(False)
    else:
        model.adapter = torch.nn.Parameter(model.adapter.detach().clone())
        model.tied = model.adapter

    worker.save_model_to_cpu(0)
    assert not isinstance(worker.cpu_saved_models[0], snapshots._TrainableSnapshot)
    assert worker.test_calls == [tuple(id(param) for param in model.parameters())]
    original = model.base.detach().clone()
    with torch.no_grad():
        model.base.add_(1)
    worker.restore_model_from_cpu(0)
    torch.testing.assert_close(model.base, original, rtol=0, atol=0)


@pytest.mark.parametrize(
    "drift", ["new", "freeze", "dtype", "shape", "placement", "module", "config", "replace", "alias"]
)
def test_mapping_drift_rejected_before_any_restore(worker, drift):
    model = worker.actor.engine.module
    worker.save_model_to_cpu(0)
    with torch.no_grad():
        model.extra.add_(10)
    if drift == "new":
        model.new = torch.nn.Parameter(torch.ones(1))
    elif drift == "freeze":
        model.adapter.requires_grad_(False)
    elif drift == "dtype":
        model.adapter.data = model.adapter.data.double()
    elif drift == "shape":
        model.adapter.data = torch.ones(3)
    elif drift == "placement":
        model.adapter.placements = ("replicate",)
    elif drift == "replace":
        model.adapter = _ShardParameter(model.adapter.detach().clone())
        model.tied = model.adapter
    elif drift == "alias":
        model.tied = model.base
    elif drift == "module":
        replacement = torch.nn.Module()
        replacement.base = model.base
        replacement.adapter = model.adapter
        replacement.extra = model.extra
        replacement.peft_config = model.peft_config
        worker.actor.engine.module = replacement
    else:
        worker.actor.engine.model_config.lora_rank = 0
    with pytest.raises(RuntimeError, match="parameter mapping changed"):
        worker.restore_model_from_cpu(0)
    torch.testing.assert_close(model.extra, torch.tensor([13.0]), rtol=0, atol=0)


def test_remote_rank_ineligibility_uses_full_snapshot_on_every_rank(worker, monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def remote_ineligible(flag, op):
        assert op == torch.distributed.ReduceOp.MIN
        assert flag.item() == 1
        flag.fill_(0)

    monkeypatch.setattr(torch.distributed, "all_reduce", remote_ineligible)
    worker.save_model_to_cpu(0)
    assert not isinstance(worker.cpu_saved_models[0], snapshots._TrainableSnapshot)
    assert worker.test_calls == [tuple(id(param) for param in worker.actor.engine.module.parameters())]
    worker.restore_model_from_cpu(0)


def test_remote_rank_mapping_failure_is_collective(worker, monkeypatch):
    worker.save_model_to_cpu(0)
    model = worker.actor.engine.module
    with torch.no_grad():
        model.extra.add_(10)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def remote_failure(flag, op):
        assert op == torch.distributed.ReduceOp.MAX
        flag.fill_(1)

    monkeypatch.setattr(torch.distributed, "all_reduce", remote_failure)
    with pytest.raises(RuntimeError, match="at least one actor rank"):
        worker.restore_model_from_cpu(0)
    torch.testing.assert_close(model.extra, torch.tensor([13.0]), rtol=0, atol=0)


def test_snapshot_save_failure_does_not_publish_and_reoffloads(worker):
    worker.actor.engine.is_param_offload_enabled = True

    def fail(_view):
        raise RuntimeError("injected copy failure")

    worker._strategy_handlers = (fail, worker.restore_handler)
    with pytest.raises(RuntimeError, match="injected copy failure"):
        worker.save_model_to_cpu(0)
    assert worker.cpu_saved_models == {}
    assert worker.test_transitions == [("cpu", True, False, False)] * 2


def test_repeated_snapshots_and_missing_id(worker):
    model = worker.actor.engine.module
    worker.save_model_to_cpu(0)
    with torch.no_grad():
        model.extra.add_(1)
    worker.save_model_to_cpu(1)
    worker.restore_model_from_cpu(0)
    torch.testing.assert_close(model.extra, torch.tensor([3.0]), rtol=0, atol=0)
    worker.restore_model_from_cpu(1)
    torch.testing.assert_close(model.extra, torch.tensor([4.0]), rtol=0, atol=0)
    worker.clear_cpu_model(0)
    worker.clear_cpu_model(1)
    with pytest.raises(KeyError, match="Unknown actor CPU snapshot"):
        worker.restore_model_from_cpu(1)


def test_old_log_prob_failure_restores_current_actor_and_clears_cycle(worker, monkeypatch):
    from verl_omni.trainer.diffusion.v1.trainer_base import PolicyGradientDiffusionTrainerV1
    from verl_omni.trainer.diffusion.v1.trainer_separate_async import PolicyGradientDiffusionTrainerV1SeparateAsync

    model = worker.actor.engine.module
    trainer = object.__new__(PolicyGradientDiffusionTrainerV1SeparateAsync)
    trainer.parameter_sync_step = 2
    trainer.actor_rollout_wg = worker
    trainer.local_trigger_step = 0

    def compute_old(_trainer, _data):
        torch.testing.assert_close(model.extra, torch.tensor([3.0]), rtol=0, atol=0)
        if trainer.local_trigger_step == 1:
            raise RuntimeError("injected inference failure")

    monkeypatch.setattr(PolicyGradientDiffusionTrainerV1, "_compute_old_log_prob", compute_old)
    trainer._compute_old_log_prob(None)
    with torch.no_grad():
        model.extra.add_(1)
    trainer.local_trigger_step = 1
    with pytest.raises(RuntimeError, match="injected inference failure"):
        trainer._compute_old_log_prob(None)
    torch.testing.assert_close(model.extra, torch.tensor([4.0]), rtol=0, atol=0)
    assert worker.cpu_saved_models == {}
