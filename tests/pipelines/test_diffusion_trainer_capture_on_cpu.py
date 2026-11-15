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

"""CPU tensor and event tests for the opt-in diffusion trainer capture hooks.

The fake production classes below test hook installation and callback behavior,
not native trainer execution. Native DataProto metadata transport has its own test.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

SOURCE = Path(__file__).resolve().parents[2] / "verl_omni" / "pipelines" / "consistency_capture.py"


def _capture_module():
    spec = importlib.util.spec_from_file_location("_tested_diffusion_capture", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _events(root):
    events = []
    for folder in root.iterdir():
        record_path = folder / "record.json"
        payload_path = folder / "payload.pt"
        assert record_path.is_file() and payload_path.is_file()
        record = json.loads(record_path.read_text())
        assert record["payload_sha256"] == hashlib.sha256(payload_path.read_bytes()).hexdigest()
        assert record["capture_source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        assert record["rank"] == 0 and record["version"] == 1
        events.append((record, torch.load(payload_path, map_location="cpu", weights_only=True)))
    return events


@pytest.fixture
def hooks(monkeypatch, tmp_path):
    """Install the observer against minimal class modules, with actual torch math and event I/O."""
    capture = _capture_module()
    root = tmp_path / "capture"
    monkeypatch.setenv("DIFFUSION_CAPTURE_DIR", str(root))
    monkeypatch.setenv("DIFFUSION_CAPTURE_STEPS", "1")

    class FlowMatchSDEDiscreteScheduler:
        def __init__(self):
            self.timesteps = torch.tensor([900.0, 600.0, 300.0])
            self.sigmas = torch.tensor([0.9, 0.6, 0.3, 0.0])
            self.config = SimpleNamespace(num_train_timesteps=1000)

        def index_for_timestep(self, timestep):
            return int((self.timesteps == timestep).nonzero()[0, 0])

    class PPODiffusersFSDPEngine:
        def __init__(self, source=True):
            self.scheduler = FlowMatchSDEDiscreteScheduler()
            self.model_config = SimpleNamespace(adapter=SimpleNamespace(consistency_timestep_unit="training_step"))
            self.source = source
            self.parameter = torch.nn.Parameter(torch.tensor(2.0))
            self.fail = False

        def is_mp_src_rank_with_outputs(self):
            return self.source

        def forward_step(self, micro_batch, loss_function, forward_only, step):
            if self.fail:
                raise RuntimeError("engine failed")
            noise = torch.randn(micro_batch["all_timesteps"].shape[0], 1)
            sample = noise + self.parameter
            log_probs = sample.square()
            loss = log_probs.sum() if not forward_only else None
            return loss, {"model_output": {"log_probs": log_probs}, "noise": noise}

    class DiffusersFSDPEngine:
        def _run_forward_backward_batch(self, data, loss_function, forward_only, *, timesteps_key):
            if self.fail:
                raise RuntimeError("engine failed")
            outputs = []
            for step in range(data[timesteps_key].shape[1]):
                loss, output = self.forward_step(data, loss_function, forward_only, step)
                if loss is not None:
                    loss.backward()
                outputs.append(output["model_output"]["log_probs"])
            return torch.cat(outputs, dim=1)

    class PolicyGradientDiffusionTrainerV1:
        def __init__(self, engine):
            self.global_steps = 1
            self.engine = engine

        def _compute_old_log_prob(self, data):
            # This stand-in models the metadata hop to the engine, not DataProto internals.
            worker_data = dict(data.batch)
            worker_data.update(data.meta_info)
            values = self.engine._run_forward_backward_batch(worker_data, None, True, timesteps_key="all_timesteps")
            return SimpleNamespace(batch={"old_log_probs": values})

    def get_non_tensor_data(data, key, default=None):
        return data.get(key, default)

    modules = {
        "verl": ModuleType("verl"),
        "verl.utils": ModuleType("verl.utils"),
        "verl.utils.tensordict_utils": ModuleType("verl.utils.tensordict_utils"),
        "verl_omni": ModuleType("verl_omni"),
        "verl_omni.trainer": ModuleType("verl_omni.trainer"),
        "verl_omni.trainer.diffusion": ModuleType("verl_omni.trainer.diffusion"),
        "verl_omni.trainer.diffusion.v1": ModuleType("verl_omni.trainer.diffusion.v1"),
        "verl_omni.trainer.diffusion.v1.trainer_base": ModuleType("verl_omni.trainer.diffusion.v1.trainer_base"),
        "verl_omni.workers": ModuleType("verl_omni.workers"),
        "verl_omni.workers.engine": ModuleType("verl_omni.workers.engine"),
        "verl_omni.workers.engine.fsdp": ModuleType("verl_omni.workers.engine.fsdp"),
        "verl_omni.workers.engine.fsdp.diffusers_impl": ModuleType("verl_omni.workers.engine.fsdp.diffusers_impl"),
        "verl_omni.pipelines": ModuleType("verl_omni.pipelines"),
        "verl_omni.pipelines.model_base": ModuleType("verl_omni.pipelines.model_base"),
        "verl_omni.pipelines.schedulers": ModuleType("verl_omni.pipelines.schedulers"),
    }
    package_names = {
        "verl",
        "verl.utils",
        "verl_omni",
        "verl_omni.trainer",
        "verl_omni.trainer.diffusion",
        "verl_omni.trainer.diffusion.v1",
        "verl_omni.workers",
        "verl_omni.workers.engine",
        "verl_omni.workers.engine.fsdp",
        "verl_omni.pipelines",
    }
    for name, module in modules.items():
        if name in package_names:
            module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    modules["verl.utils"].tensordict_utils = modules["verl.utils.tensordict_utils"]
    modules["verl.utils.tensordict_utils"].get_non_tensor_data = get_non_tensor_data
    trainer_module = modules["verl_omni.trainer.diffusion.v1.trainer_base"]
    trainer_module.PolicyGradientDiffusionTrainerV1 = PolicyGradientDiffusionTrainerV1
    engine_module = modules["verl_omni.workers.engine.fsdp.diffusers_impl"]
    engine_module.DiffusersFSDPEngine = DiffusersFSDPEngine
    engine_module.PPODiffusersFSDPEngine = PPODiffusersFSDPEngine
    modules["verl_omni.pipelines.schedulers"].FlowMatchSDEDiscreteScheduler = FlowMatchSDEDiscreteScheduler
    modules["verl_omni.pipelines.model_base"].DiffusionModelBase = SimpleNamespace(
        get_class=lambda config: config.adapter
    )
    capture.install_trainer_capture()
    return capture, root, PolicyGradientDiffusionTrainerV1, DiffusersFSDPEngine, PPODiffusersFSDPEngine


def _data(rows=2):
    return SimpleNamespace(
        batch={
            "rollout_log_probs": torch.arange(rows * 2, dtype=torch.float64).reshape(rows, 2),
            "all_timesteps": torch.tensor([[900.0, 600.0]]).repeat(rows, 1),
        },
        non_tensor_batch={"uid": [f"uid-{i}" for i in range(rows)]},
        meta_info={},
    )


@pytest.mark.parametrize("rows", [1, 3])
def test_full_driver_and_engine_events_preserve_result_and_schedule(hooks, rows):
    capture, root, Trainer, EngineBase, PPO = hooks
    engine_class = type("CPUCombinedEngine", (PPO, EngineBase), {})
    trainer = Trainer(engine_class())
    data = _data(rows)
    torch.manual_seed(123)
    expected_first = (torch.randn(rows, 1) + 2).square()
    expected_second = (torch.randn(rows, 1) + 2).square()
    torch.manual_seed(123)
    result = trainer._compute_old_log_prob(data)
    torch.testing.assert_close(result.batch["old_log_probs"], torch.cat([expected_first, expected_second], 1))
    assert data.meta_info == {}
    events = _events(root)
    assert len(events) == 3
    drivers = [(r, p) for r, p in events if r["kind"] == "driver_pair"]
    engine_events = [(r, p) for r, p in events if r["kind"] == "engine_forward"]
    assert len(drivers) == 1 and len(engine_events) == 2
    record, payload = drivers[0]
    assert record["run_id"] == root.name and record["global_step"] == 1
    assert len(record["batch_id"]) == 32
    assert payload["uids"] == [f"uid-{i}" for i in range(rows)]
    torch.testing.assert_close(payload["actor_log_probs"], result.batch["old_log_probs"])
    torch.testing.assert_close(payload["rollout_log_probs"], data.batch["rollout_log_probs"].float())
    torch.testing.assert_close(payload["timesteps"], data.batch["all_timesteps"].float())
    assert {r["batch_id"] for r, _ in engine_events} == {record["batch_id"]}
    for step, expected in enumerate((expected_first, expected_second)):
        matches = [
            (r, p)
            for r, p in engine_events
            if torch.equal(p["timesteps"], data.batch["all_timesteps"][:, step : step + 1])
        ]
        assert len(matches) == 1
        event_record, event_payload = matches[0]
        assert event_record["sigma_timestep_contract"] is True
        torch.testing.assert_close(event_payload["actor_log_probs"], expected)
        torch.testing.assert_close(
            event_payload["rollout_log_probs"], data.batch["rollout_log_probs"][:, step : step + 1].float()
        )
        assert event_payload["sigma_indices"].tolist() == [step] * rows
        assert event_payload["num_train_timesteps"] == 1000
        torch.testing.assert_close(event_payload["sigma"], torch.full((rows, 1), [0.9, 0.6][step]))
        torch.testing.assert_close(event_payload["sigma_next"], torch.full((rows, 1), [0.6, 0.3][step]))
        torch.testing.assert_close(event_payload["next_timesteps"], torch.full((rows, 1), [600.0, 300.0][step]))
    assert capture._ENGINE_CONTEXT.get() is None


def test_metadata_restored_and_context_reset_after_engine_failure(hooks):
    capture, root, Trainer, EngineBase, PPO = hooks
    engine = type("CPUCombinedEngine", (PPO, EngineBase), {})()
    engine.fail = True
    data = _data()
    sentinel = {"old": "metadata"}
    data.meta_info[capture._CONTEXT_KEY] = sentinel
    with pytest.raises(RuntimeError, match="engine failed"):
        Trainer(engine)._compute_old_log_prob(data)
    assert data.meta_info[capture._CONTEXT_KEY] is sentinel
    assert capture._ENGINE_CONTEXT.get() is None
    assert not root.exists()


@pytest.mark.parametrize("unit", [None, "sigma"])
def test_sigma_units_are_not_inferred_from_scheduler_class(hooks, monkeypatch, unit):
    capture, root, Trainer, EngineBase, PPO = hooks
    engine = type("CPUCombinedEngine", (PPO, EngineBase), {})()
    engine.model_config.adapter.consistency_timestep_unit = unit
    engine.scheduler.timesteps = torch.tensor([0.75, 0.5, 0.25])
    engine.scheduler.sigmas = torch.tensor([0.75, 0.5, 0.25, 0.0])
    data = _data()
    data.batch["all_timesteps"] = torch.tensor([[0.75, 0.5]]).repeat(2, 1)
    torch.manual_seed(123)
    data.batch["rollout_log_probs"] = torch.cat([(torch.randn(2, 1) + 2).square() for _ in range(2)], 1)
    torch.manual_seed(123)
    Trainer(engine)._compute_old_log_prob(data)
    for record, payload in _events(root):
        if record["kind"] == "engine_forward":
            assert record["sigma_timestep_contract"] is False
            assert "sigma" not in payload and "sigma_next" not in payload
    monkeypatch.syspath_prepend(str(SOURCE.parents[2] / "scripts"))
    import replay_diffusion_trainer_capture as reader

    report = reader.replay(root, root.parent / "report.json")
    assert report["status"] == "admitted"
    assert report["sigma_diagnosis_available"] is False


def test_registered_sd3_and_bagel_timestep_contracts():
    if importlib.util.find_spec("verl") is None:
        pytest.skip("optional verl package is unavailable")
    from verl_omni.pipelines.bagel_flow_grpo.common import setup_bagel_sigmas
    from verl_omni.pipelines.model_base import DiffusionModelBase
    from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

    capture = _capture_module()
    assert DiffusionModelBase.consistency_timestep_unit is None
    scheduler = FlowMatchSDEDiscreteScheduler(num_train_timesteps=1000)
    setup_bagel_sigmas(scheduler, num_steps=4, device="cpu")
    config = SimpleNamespace(architecture="OmniBagelForConditionalGeneration", algorithm="flow_grpo", external_lib=None)
    assert DiffusionModelBase.get_class(config).consistency_timestep_unit is None
    batch = {"all_timesteps": scheduler.timesteps[1:2].reshape(1, 1), "rollout_log_probs": torch.zeros(1, 1)}
    engine = SimpleNamespace(scheduler=scheduler, model_config=config)
    payload, contract = capture._schedule_inputs(engine, batch, 0)
    assert contract is False and "sigma" not in payload
    index = scheduler.index_for_timestep(batch["all_timesteps"][0, 0])
    assert scheduler.sigmas[index] == batch["all_timesteps"][0, 0]
    assert not torch.isclose(scheduler.sigmas[index], batch["all_timesteps"][0, 0] / 1000)

    config.architecture = "StableDiffusion3Pipeline"
    assert DiffusionModelBase.get_class(config).consistency_timestep_unit == "training_step"
    scheduler = FlowMatchSDEDiscreteScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(4, device="cpu")
    engine.scheduler = scheduler
    batch["all_timesteps"] = scheduler.timesteps[1:2].reshape(1, 1)
    payload, contract = capture._schedule_inputs(engine, batch, 0)
    assert contract is True
    torch.testing.assert_close(payload["sigma"], payload["timesteps"] / 1000)


def test_default_step_selection_and_unset_directory_are_no_ops(hooks, monkeypatch):
    capture, root, Trainer, EngineBase, PPO = hooks
    engine = type("CPUCombinedEngine", (PPO, EngineBase), {})()
    trainer = Trainer(engine)
    trainer.global_steps = 2
    monkeypatch.delenv("DIFFUSION_CAPTURE_STEPS")
    result = trainer._compute_old_log_prob(_data())
    assert result.batch["old_log_probs"].shape == (2, 2)
    assert not root.exists()
    monkeypatch.delenv("DIFFUSION_CAPTURE_DIR")
    capture.install_trainer_capture()
    trainer.global_steps = 1
    trainer._compute_old_log_prob(_data())
    assert not root.exists()


def test_explicit_global_step_selection(hooks, monkeypatch):
    _, root, Trainer, EngineBase, PPO = hooks
    trainer = Trainer(type("CPUCombinedEngine", (PPO, EngineBase), {})())
    monkeypatch.setenv("DIFFUSION_CAPTURE_STEPS", "2, 4")
    trainer._compute_old_log_prob(_data())
    assert not root.exists()
    trainer.global_steps = 2
    trainer._compute_old_log_prob(_data())
    assert len(_events(root)) == 3


def test_install_is_idempotent_and_forward_only_source_rank_filter(hooks):
    capture, root, Trainer, EngineBase, PPO = hooks
    original = (Trainer._compute_old_log_prob, EngineBase._run_forward_backward_batch, PPO.forward_step)
    capture.install_trainer_capture()
    assert original == (Trainer._compute_old_log_prob, EngineBase._run_forward_backward_batch, PPO.forward_step)
    engine = type("CPUCombinedEngine", (PPO, EngineBase), {})(source=False)
    data = _data()
    context = {"run_id": "r", "batch_id": "b", "global_step": 1}
    data.meta_info[capture._CONTEXT_KEY] = context
    engine._run_forward_backward_batch({**data.batch, **data.meta_info}, None, True, timesteps_key="all_timesteps")
    assert not root.exists()
    engine.source = True
    engine._run_forward_backward_batch({**data.batch, **data.meta_info}, None, False, timesteps_key="all_timesteps")
    assert engine.parameter.grad is not None and engine.parameter.grad.abs().item() > 0
    assert not root.exists()
    assert capture._ENGINE_CONTEXT.get() is None


def test_step_return_and_rng_state_are_unchanged(hooks):
    capture, root, _, _, PPO = hooks
    engine = PPO()
    batch = _data().batch
    original_step = PPO.forward_step.__wrapped__
    context = {"run_id": "capture", "batch_id": "a" * 32, "global_step": 1}
    torch.manual_seed(87)
    expected_loss, expected_output = original_step(engine, batch, None, True, 0)
    expected_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(87)
    token = capture._ENGINE_CONTEXT.set(context)
    try:
        actual_loss, actual_output = engine.forward_step(batch, None, True, 0)
    finally:
        capture._ENGINE_CONTEXT.reset(token)
    assert actual_loss is expected_loss is None
    torch.testing.assert_close(actual_output["model_output"]["log_probs"], expected_output["model_output"]["log_probs"])
    torch.testing.assert_close(actual_output["noise"], expected_output["noise"])
    assert torch.equal(torch.random.get_rng_state(), expected_rng)
    assert len(_events(root)) == 1 and capture._ENGINE_CONTEXT.get() is None


def test_training_loss_and_gradient_are_unchanged(hooks):
    capture, root, _, _, PPO = hooks
    engine = PPO()
    batch = _data().batch
    torch.manual_seed(37)
    baseline_loss, baseline_output = PPO.forward_step.__wrapped__(engine, batch, None, False, 0)
    baseline_loss.backward()
    expected_grad = engine.parameter.grad.detach().clone()
    engine.parameter.grad = None
    torch.manual_seed(37)
    loss, output = engine.forward_step(batch, None, False, 0)
    loss.backward()
    torch.testing.assert_close(loss, baseline_loss)
    torch.testing.assert_close(output["model_output"]["log_probs"], baseline_output["model_output"]["log_probs"])
    torch.testing.assert_close(engine.parameter.grad, expected_grad)
    assert not root.exists() and capture._ENGINE_CONTEXT.get() is None


def test_native_dataproto_to_tensordict_metadata_transport():
    if importlib.util.find_spec("verl") is None:
        pytest.skip("optional verl package is unavailable")
    from verl import DataProto
    from verl.utils import tensordict_utils as tu

    from verl_omni.trainer.diffusion.diffusion_trainer_utils import _to_diffusion_worker_tensordict

    data = DataProto.from_dict(tensors={"all_timesteps": torch.tensor([[900.0]])})
    key = "_diffusion_consistency_capture"
    context = {"run_id": "run", "batch_id": "batch", "global_step": 1}
    data.meta_info[key] = context
    worker_batch = _to_diffusion_worker_tensordict(data)
    assert tu.get_non_tensor_data(worker_batch, key, default=None) == context
    assert data.meta_info[key] == context
