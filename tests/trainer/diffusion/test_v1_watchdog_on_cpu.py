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
"""Watchdog configuration and the actual V1 fit-loop reporting boundary."""

import logging
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

import verl_omni
from verl_omni.trainer.diffusion.v1 import trainer_base
from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync


def _trainer(monkeypatch, overrides=()):
    config_dir = Path(verl_omni.__file__).parent / "trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="diffusion_trainer", overrides=list(overrides))
    monkeypatch.setattr(PolicyGradientDiffusionTrainerV1Sync, "_build_replay_buffer", lambda self: None)
    return PolicyGradientDiffusionTrainerV1Sync(config)


def test_watchdog_default_does_not_construct_or_change_metrics(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("disabled watchdog must not be constructed")

    monkeypatch.setattr(trainer_base, "TrainingWatchdog", forbidden)
    trainer = _trainer(monkeypatch)
    metrics = {"actor/grad_norm": 0.0, "actor/ratio_mean": 10.0}
    trainer._report_training_watchdog(metrics)
    assert trainer._training_watchdog is None
    assert metrics == {"actor/grad_norm": 0.0, "actor/ratio_mean": 10.0}


def test_nondefault_hydra_watchdog_config_reaches_reporting(monkeypatch, caplog):
    trainer = _trainer(
        monkeypatch,
        [
            "trainer.v1.watchdog.enabled=true",
            "trainer.v1.watchdog.window_size=2",
            "trainer.v1.watchdog.warmup_steps=0",
            "trainer.v1.watchdog.cooldown_steps=0",
            "trainer.v1.watchdog.ratio_upper=1.1",
            "trainer.v1.watchdog.max_staleness=2",
        ],
    )
    with caplog.at_level(logging.WARNING, logger=trainer_base.logger.name):
        for step in (10, 11):
            trainer.global_steps = step
            metrics = {"actor/ratio_mean": 1.2, "training/off_policy/trajectory_staleness_worst/max": 3.0}
            trainer._report_training_watchdog(metrics)
    assert metrics["watchdog/alerts"] == 2.0
    assert metrics["actor/ratio_mean"] == 1.2
    assert metrics["watchdog/processing_time_s"] >= 0
    assert "window=[10,11]" in caplog.text
    assert "evidence=" in caplog.text


def test_invalid_enabled_config_rejected(monkeypatch):
    with pytest.raises(ValueError, match="enabled must be a boolean"):
        _trainer(monkeypatch, ['trainer.v1.watchdog.enabled="yes"'])


def test_production_numpy_scalar_metrics_reach_watchdog(monkeypatch):
    trainer = _trainer(
        monkeypatch,
        [
            "trainer.v1.watchdog.enabled=true",
            "trainer.v1.watchdog.window_size=1",
            "trainer.v1.watchdog.warmup_steps=0",
            "trainer.v1.watchdog.max_staleness=2",
        ],
    )
    trainer.global_steps = 4
    # Match the NumPy array reductions in _compute_metrics and reduce_metrics.
    metrics = {
        "actor/grad_norm": np.float32(0.1),
        "actor/ratio_mean": np.float64(3.0),
        "training/off_policy/trajectory_staleness_worst/max": np.array([1, 3], dtype=int).max(),
    }
    trainer._report_training_watchdog(metrics)
    assert metrics["watchdog/alerts"] == 2.0
    assert metrics["watchdog/coverage/trajectory_staleness"] == 1.0
    assert metrics["watchdog/coverage/nonfinite_grad"] == 1.0


def _run_fit(monkeypatch, *, enabled, initial_step=0):
    trainer = _trainer(
        monkeypatch,
        [
            f"trainer.v1.watchdog.enabled={str(enabled).lower()}",
            "trainer.v1.watchdog.warmup_steps=2",
            "trainer.v1.watchdog.window_size=2",
            "trainer.val_before_train=false",
            "trainer.test_freq=-1",
            "trainer.save_freq=-1",
            "trainer.logger=[console]",
            "trainer.total_epochs=1",
        ],
    )
    torch.manual_seed(31)
    parameter = torch.nn.Parameter(torch.randn(3))
    optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.9)
    rows, events = [], []
    trainer.global_steps = initial_step
    trainer.steps_per_epoch = 100
    trainer.total_training_steps = initial_step + 4
    if enabled:
        # Simulate stale diagnostics from a previous fit; fit must reset them.
        trainer._training_watchdog.observe(initial_step - 1, {"actor/ratio_mean": 10.0})

    def step(metrics, timing):
        optimizer.zero_grad()
        loss = (parameter * torch.randn(3)).square().mean()
        loss.backward()
        optimizer.step()
        metrics.update({"actor/loss": float(loss.detach()), "actor/ratio_mean": 10.0})
        events.append(("update", trainer.global_steps))
        return SimpleNamespace(keys=["sample"], partition_id="train")

    def compute(batch, metrics, timing, step, epoch):
        metrics["training/global_step"] = step
        events.append(("metrics", step))

    monkeypatch.setattr(trainer, "step", step)
    monkeypatch.setattr(trainer, "_compute_metrics", compute)
    for name in (
        "on_train_begin",
        "on_step_begin",
        "on_step_end",
        "on_train_end",
        "_reissue_inflight_prompts",
        "_shutdown_dump_executor",
        "_shutdown_dataloaders",
    ):
        monkeypatch.setattr(trainer, name, lambda: None)
    monkeypatch.setattr(trainer, "_consume_sync_metrics", lambda: {})
    monkeypatch.setattr(trainer_base, "Tracking", lambda **kw: SimpleNamespace(log=lambda **kw: rows.append(kw)))
    monkeypatch.setattr(trainer_base, "ValidationGenerationsLogger", lambda **kw: None)
    monkeypatch.setattr(trainer_base.SkipManager, "init", lambda config: None)
    monkeypatch.setattr(trainer_base.SkipManager, "set_step", lambda step: None)
    monkeypatch.setattr(trainer_base.tq, "kv_clear", lambda **kw: events.append(("clear", trainer.global_steps)))
    trainer.fit(None)
    return parameter.detach().clone(), deepcopy(optimizer.state_dict()), torch.get_rng_state(), rows, events


@pytest.mark.parametrize("initial_step", [0, 50])
def test_real_fit_loop_preserves_updates_rng_and_resets_warmup(monkeypatch, initial_step):
    baseline = _run_fit(monkeypatch, enabled=False, initial_step=initial_step)
    treatment = _run_fit(monkeypatch, enabled=True, initial_step=initial_step)
    torch.testing.assert_close(baseline[0], treatment[0], rtol=0, atol=0)
    assert torch.equal(baseline[2], treatment[2])
    assert baseline[1]["param_groups"] == treatment[1]["param_groups"]
    for key in baseline[1]["state"][0]:
        torch.testing.assert_close(baseline[1]["state"][0][key], treatment[1]["state"][0][key], rtol=0, atol=0)
    assert baseline[4] == treatment[4]
    assert len(treatment[3]) == 4
    assert [row["data"]["watchdog/alerts"] for row in treatment[3]] == [0.0, 0.0, 0.0, 1.0]
    for base_row, row in zip(baseline[3], treatment[3], strict=True):
        assert base_row == {
            "step": row["step"],
            "data": {k: v for k, v in row["data"].items() if not k.startswith("watchdog/")},
        }
