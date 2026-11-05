# Copyright 2026 verl-omni contributors
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

"""Owned trainer hook tests with explicit worker doubles, without Ray/TQ."""

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


class OrdinaryTrainer:
    def _compute_old_log_prob(self, batch, metrics):
        self.calls.append("old")
        return batch

    def _compute_advantage(self, batch, metrics):
        self.calls.append("advantage")
        return batch

    def _update_actor(self, batch, metrics):
        self.calls.append("ordinary-update")
        return batch


def hooks():
    path = ROOT / "verl_omni/trainer/omni/proximal_actor_reuse.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    cls.bases = [ast.Name(id="OrdinaryTrainer", ctx=ast.Load())]
    names = {
        "_compute_old_log_prob",
        "_compute_advantage",
        "_update_actor",
        "_select_proximal_actor",
        "_check_proximal_algorithm",
    }
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = {"OrdinaryTrainer": OrdinaryTrainer, "logger": logging.getLogger(__name__)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), "exec"), scope)
    result = scope[cls.name]()
    result.calls = []
    result.config = OmegaConf.create({"trainer": {"v1": {"reuse_proximal_actor_forward": False}, "critic_warmup": 0}})
    return result


def test_disabled_setting_delegates_all_hooks_unchanged():
    trainer = hooks()
    batch, metrics = object(), {}
    assert trainer._compute_old_log_prob(batch, metrics) is batch
    assert trainer._compute_advantage(batch, metrics) is batch
    assert trainer._update_actor(batch, metrics) is batch
    assert trainer.calls == ["old", "advantage", "ordinary-update"]
    assert metrics == {}


@pytest.mark.parametrize("supported", [True, False])
def test_worker_eligibility_is_checked_before_skipping_old(supported):
    trainer = hooks()
    trainer.config.trainer.v1.reuse_proximal_actor_forward = True
    trainer.global_steps = 1
    trainer._check_proximal_algorithm = lambda batch: None
    trainer.actor_rollout_wg = SimpleNamespace(
        qualify_proximal_actor=lambda: [{"supported": supported}, {"supported": supported}],
        _query_dispatch_info=lambda mesh: [0, 0],
    )
    ticket = object()
    trainer._prepare_proximal_actor = lambda batch: ticket

    def update(batch, metrics, prepared):
        assert prepared is ticket
        trainer.calls.append("leased-update")
        return batch

    trainer._update_proximal_actor = update
    batch, metrics = object(), {}
    trainer._compute_old_log_prob(batch, metrics)
    trainer._update_actor(batch, metrics)
    assert trainer.calls == (["leased-update"] if supported else ["old", "ordinary-update"])
    assert metrics["training/proximal_anchor_reuse"] == int(supported)


def test_partial_failure_never_falls_back_or_prepares_again():
    trainer = hooks()
    trainer._omni_prepared_actor = object()

    def fail(*args):
        raise RuntimeError("partial update")

    trainer._update_proximal_actor = fail
    with pytest.raises(RuntimeError, match="partial"):
        trainer._update_actor(object(), {})
    with pytest.raises(RuntimeError, match="has not finished"):
        trainer._compute_old_log_prob(object(), {})
    assert trainer.calls == []


@pytest.mark.parametrize("field,value", [("ppo_epochs", 2), ("shuffle", True), ("use_prefix_grouper", True)])
def test_unsupported_update_schedule_is_rejected(field, value):
    trainer = hooks()
    trainer.config = OmegaConf.create(
        {
            "trainer": {"v1": {}},
            "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False},
            "actor_rollout_ref": {
                "actor": {
                    "ppo_epochs": 1,
                    "shuffle": False,
                    "use_prefix_grouper": False,
                    "calculate_sum_pi_squared": False,
                    "policy_loss": {"loss_mode": "gspo"},
                    "ppo_mini_batch_size": 2,
                },
                "rollout": {"n": 1},
            },
        }
    )
    trainer.config.actor_rollout_ref.actor[field] = value
    trainer.parameter_sync_step = 1
    trainer.use_critic = trainer.use_teacher_policy = False
    with pytest.raises(ValueError, match="one qualified"):
        trainer._check_proximal_algorithm([1, 2])
