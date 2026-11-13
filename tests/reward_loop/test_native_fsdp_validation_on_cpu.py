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
"""Small regression checks for the native reward acceptance tools."""

import json
import sys
from types import SimpleNamespace

import hydra
import pytest
import torch

from tests.reward_loop.validate_native_fsdp import owned_actor_states
from tests.special_e2e import validate_native_fsdp_training_failure as failure_validation
from tests.special_e2e.verify_native_fsdp_training import optimizer_steps, validate_training_log


def test_gradient_values_and_resume_messages_are_field_bounded():
    for value in ("nan", "inf", "-inf", "", "0", "-1"):
        with pytest.raises((ValueError, AssertionError)):
            validate_training_log(f"actor/grad_norm: {value}\nactor/lr: 0.001", 1)
    assert validate_training_log("actor/grad_norm: 1.25e-3 - actor/lr: 0.001", 1) == [0.00125]
    for message in (
        "Found checkpoint: %s /run/global_step_2",
        "Resuming diffusion from /run/global_step_2, global_steps=2",
    ):
        assert validate_training_log(message + "\nactor/grad_norm: 0.5", 3, 2) == [0.5]
    with pytest.raises(AssertionError):
        validate_training_log("Resuming diffusion from /run/global_step_20\nactor/grad_norm: 0.5", 3, 2)


def test_optimizer_counters_match_the_checkpoint_step(tmp_path):
    path = tmp_path / "optimizer.pt"
    for observed in (2.0, 3.0, 2.5, float("nan")):
        torch.save(
            {"state": {0: {"step": torch.tensor(observed), "exp_avg": torch.ones(1), "exp_avg_sq": torch.ones(1)}}},
            path,
        )
        if observed == 2:
            assert optimizer_steps(path, 2) == [2.0]
        else:
            with pytest.raises(AssertionError):
                optimizer_steps(path, 2)


def test_both_resume_modes_emit_an_accepted_trainer_message(tmp_path, caplog):
    from verl_omni.trainer.diffusion.v1 import trainer_base

    checkpoint = tmp_path / "global_step_2"
    checkpoint.mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("2")
    for mode in ("auto", "resume_path"):
        calls = []
        trainer = SimpleNamespace(
            config=SimpleNamespace(
                trainer=SimpleNamespace(
                    resume_mode=mode,
                    default_local_dir=str(tmp_path),
                    resume_from_path=str(checkpoint),
                    del_local_ckpt_after_load=False,
                )
            ),
            trainer_mode="sync",
            actor_rollout_wg=SimpleNamespace(load_checkpoint=lambda path, calls=calls, **kwargs: calls.append(path)),
        )
        caplog.clear()
        with caplog.at_level("INFO", logger=trainer_base.__name__):
            trainer_base.PolicyGradientDiffusionTrainerV1._load_checkpoint(trainer)
        assert trainer.global_steps == 2 and calls == [str(checkpoint / "actor")]
        assert validate_training_log(caplog.text + "\nactor/grad_norm: 0.5", 3, 2) == [0.5]


def test_owned_actor_inventory_cannot_omit_facade_or_be_empty(monkeypatch):
    rows = {
        name: {"ActorID": name, "ActorClassName": role, "Name": "", "Pid": 1, "State": "DEAD"}
        for name, role in (("rank0", "rank"), ("rank1", "rank"), ("facade", "facade"))
    }
    monkeypatch.setattr(failure_validation.ray._private.state, "actors", lambda: rows)
    counts = {"rank": 2, "facade": 1}
    assert len(owned_actor_states(list(rows), counts)) == 3
    for ids in ([], ["rank0", "rank1"], ["rank0", "rank0", "facade"]):
        with pytest.raises(AssertionError):
            owned_actor_states(ids, counts)
    with pytest.raises(KeyError):
        owned_actor_states(["rank0", "rank1", "missing"], counts)


def test_hydra_failure_still_checks_owned_actors_and_checkpoint(tmp_path, monkeypatch):
    @hydra.main(version_base=None, config_path=None)
    def failed_training(config):
        raise RuntimeError("fit: FSDP wake_up failed: model.safetensors missing")

    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps([f"hydra.run.dir={tmp_path / 'hydra'}", "hydra.output_subdir=null"]))
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("0")
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify",
            "--overrides",
            str(overrides),
            "--checkpoint-root",
            str(tmp_path),
            "--expected-step",
            "0",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setenv("HYDRA_FULL_ERROR", "0")
    monkeypatch.setattr(failure_validation.runpy, "run_module", lambda *args, **kwargs: failed_training())
    roles = [
        "PickScoreFSDPRank",
        "PickScoreFSDPRank",
        "OmniFSDPRewardLoopWorker",
        "TransferQueueController",
        "SimpleStorageUnit",
    ]
    rows = {
        str(index): {
            "ActorID": str(index),
            "ActorClassName": role,
            "Name": "",
            "Pid": 1,
            "State": "DEAD",
            "JobID": "job",
        }
        for index, role in enumerate(roles)
    }
    rows["other"] = dict(rows["0"], ActorID="other", JobID="other", State="ALIVE")
    monkeypatch.setattr(failure_validation.ray._private.state, "actors", lambda: rows)
    monkeypatch.setattr(failure_validation.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        failure_validation.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_job_id=lambda: SimpleNamespace(hex=lambda: "job")),
    )
    monkeypatch.setattr(failure_validation.ray, "shutdown", lambda: None)
    failure_validation.main()
    report = json.loads(output.read_text())
    assert report["passed"] and report["latest_checkpoint"] == 0
    assert len(report["actors_before_ray_shutdown"]) == 5 and "expected_error" in report
