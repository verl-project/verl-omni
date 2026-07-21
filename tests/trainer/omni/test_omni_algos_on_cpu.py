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
"""CPU tests for omni direct-preference loss functions."""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _bootstrap_omni_algos():
    sys.modules.setdefault("verl_omni", types.ModuleType("verl_omni"))
    sys.modules.setdefault("verl_omni.trainer", types.ModuleType("verl_omni.trainer"))
    sys.modules.setdefault("verl_omni.trainer.omni", types.ModuleType("verl_omni.trainer.omni"))
    sys.modules.setdefault("verl_omni.workers", types.ModuleType("verl_omni.workers"))
    sys.modules.setdefault("verl_omni.workers.config", types.ModuleType("verl_omni.workers.config"))
    sys.modules.setdefault("verl_omni.workers.config.omni", types.ModuleType("verl_omni.workers.config.omni"))

    actor_mod = _load_module(
        "verl_omni.workers.config.omni.actor",
        REPO_ROOT / "verl_omni" / "workers" / "config" / "omni" / "actor.py",
    )
    omni_pkg = sys.modules["verl_omni.workers.config.omni"]
    omni_pkg.OmniLossConfig = actor_mod.OmniLossConfig

    return _load_module(
        "verl_omni.trainer.omni.omni_algos",
        REPO_ROOT / "verl_omni" / "trainer" / "omni" / "omni_algos.py",
    )


@pytest.fixture(scope="module")
def omni_algos():
    return _bootstrap_omni_algos()


@pytest.fixture(scope="module")
def OmniLossConfig(omni_algos):
    return sys.modules["verl_omni.workers.config.omni.actor"].OmniLossConfig


@dataclass
class _FakeActorConfig:
    omni_loss: object


class TestOmniLossRegistry:
    def test_builtin_dpo_registered(self, omni_algos):
        assert "dpo" in omni_algos.OMNI_LOSS_REGISTRY

    def test_get_existing_loss_fn(self, omni_algos):
        fn = omni_algos.get_omni_loss_fn("dpo")
        assert isinstance(fn, omni_algos.OmniDPOLoss)

    def test_get_unknown_loss_fn_raises(self, omni_algos):
        with pytest.raises(ValueError, match="Unsupported omni loss mode"):
            omni_algos.get_omni_loss_fn("nonexistent_loss")


class TestOmniDPOLossComputeLoss:
    def test_sigmoid_zero_logits_matches_log_two(self, omni_algos):
        logps = torch.tensor([-1.0, -2.0, -1.1, -2.1])
        loss, _ = omni_algos.OmniDPOLoss.compute_loss(
            policy_chosen_logps=logps[0:1],
            policy_rejected_logps=logps[1:2],
            reference_chosen_logps=logps[2:3],
            reference_rejected_logps=logps[3:4],
            beta=0.1,
            loss_type="sigmoid",
        )
        assert loss.shape == ()
        assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item(), rel=1e-5)

    def test_reward_accuracy_reflects_implicit_rewards(self, omni_algos):
        _, metrics = omni_algos.OmniDPOLoss.compute_loss(
            policy_chosen_logps=torch.tensor([0.0]),
            policy_rejected_logps=torch.tensor([-1.0]),
            reference_chosen_logps=torch.tensor([-1.0]),
            reference_rejected_logps=torch.tensor([-1.0]),
            beta=0.5,
            loss_type="sigmoid",
        )
        assert metrics["reward_accuracy"].item() == pytest.approx(1.0)

    def test_policy_margin_improves_over_reference_lowers_loss(self, omni_algos):
        worse_loss, _ = omni_algos.OmniDPOLoss.compute_loss(
            policy_chosen_logps=torch.tensor([-1.0]),
            policy_rejected_logps=torch.tensor([-1.0]),
            reference_chosen_logps=torch.tensor([-1.0]),
            reference_rejected_logps=torch.tensor([-1.0]),
            beta=0.5,
            loss_type="sigmoid",
        )
        better_loss, _ = omni_algos.OmniDPOLoss.compute_loss(
            policy_chosen_logps=torch.tensor([0.0]),
            policy_rejected_logps=torch.tensor([-2.0]),
            reference_chosen_logps=torch.tensor([-1.0]),
            reference_rejected_logps=torch.tensor([-1.0]),
            beta=0.5,
            loss_type="sigmoid",
        )
        assert better_loss.item() < worse_loss.item()

    def test_ipo_loss_is_non_negative(self, omni_algos):
        loss, _ = omni_algos.OmniDPOLoss.compute_loss(
            policy_chosen_logps=torch.tensor([0.2]),
            policy_rejected_logps=torch.tensor([-0.3]),
            reference_chosen_logps=torch.tensor([0.1]),
            reference_rejected_logps=torch.tensor([-0.2]),
            beta=0.2,
            loss_type="ipo",
        )
        assert loss.item() >= 0.0

    def test_label_smoothing_changes_loss(self, omni_algos):
        kwargs = dict(
            policy_chosen_logps=torch.tensor([0.5]),
            policy_rejected_logps=torch.tensor([-0.5]),
            reference_chosen_logps=torch.tensor([0.0]),
            reference_rejected_logps=torch.tensor([0.0]),
            beta=0.1,
            loss_type="sigmoid",
        )
        plain_loss, _ = omni_algos.OmniDPOLoss.compute_loss(label_smoothing=0.0, **kwargs)
        smooth_loss, _ = omni_algos.OmniDPOLoss.compute_loss(label_smoothing=0.1, **kwargs)
        assert plain_loss.item() != smooth_loss.item()


class TestOmniDPOLossCallable:
    def test_callable_returns_scalar_loss_and_metrics(self, omni_algos, OmniLossConfig):
        loss_fn = omni_algos.get_omni_loss_fn("dpo")
        actor_config = _FakeActorConfig(omni_loss=OmniLossConfig())
        model_output = {
            "policy_chosen_logps": torch.tensor([0.0, 0.2]),
            "policy_rejected_logps": torch.tensor([-1.0, -0.8]),
            "reference_chosen_logps": torch.tensor([-0.1, 0.0]),
            "reference_rejected_logps": torch.tensor([-0.9, -0.7]),
        }
        result = loss_fn(
            config=actor_config,
            model_output=model_output,
            data=TensorDict({}, batch_size=[2]),
        )
        assert result.loss.shape == ()
        assert "dpo_loss" in result.metrics
        assert "reward_margin" in result.metrics

    def test_validate_inputs_reports_missing_keys(self, omni_algos):
        loss_fn = omni_algos.get_omni_loss_fn("dpo")
        with pytest.raises(KeyError, match="missing required model_output keys"):
            loss_fn.validate_inputs(
                model_output={"policy_chosen_logps": torch.tensor([0.0])},
                data=TensorDict({}, batch_size=[1]),
            )
