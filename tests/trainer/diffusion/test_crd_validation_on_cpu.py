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
"""Fault injection for CRD GPU evidence checks, without launching a GPU job."""

import sys

import pytest
import torch

from tests.special_e2e.check_crd_run import check_run, main
from tests.special_e2e.crd_smoke_reward import compute_score


@pytest.fixture
def evidence(tmp_path):
    log = tmp_path / "run.log"
    log.write_text(
        "step:3 - actor/crd_loss:0.2 - actor/crd_policy_loss:0.1 - actor/crd_kl_loss:0.1 - actor/grad_norm:0.5\n"
    )
    checkpoint = tmp_path / "global_step_3" / "actor"
    checkpoint.mkdir(parents=True)
    for prefix in ("model", "optim", "extra_state"):
        (checkpoint / f"{prefix}_world_size_1_rank_0.pt").write_bytes(b"fixture")
    return log, tmp_path


def test_complete_run_evidence(evidence, monkeypatch, capsys):
    log, root = evidence
    monkeypatch.setattr(sys, "argv", ["check_crd_run", "--log", str(log), "--output-dir", str(root), "--steps", "3"])
    main()
    assert capsys.readouterr().out == "CRD finite metrics and complete checkpoints verified for steps [3].\n"


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_metrics_fail(evidence, value):
    log, root = evidence
    log.write_text(log.read_text().replace("crd_loss:0.2", f"crd_loss:{value}"))
    with pytest.raises(ValueError, match="nonfinite"):
        check_run(log, root, [3])


def test_missing_checkpoint_rank_fails(evidence):
    log, root = evidence
    path = root / "global_step_3/actor/model_world_size_1_rank_0.pt"
    path.rename(path.with_name("model_world_size_2_rank_0.pt"))
    with pytest.raises(ValueError, match="Incomplete"):
        check_run(log, root, [3])


def test_resume_did_not_advance_fails(evidence):
    with pytest.raises(ValueError, match="Expected training steps"):
        check_run(*evidence, [4])


@pytest.mark.parametrize("failure", ["zero_gradient", "skipped", "missing_optimizer"])
def test_unsuccessful_update_fails(evidence, failure):
    log, root = evidence
    if failure == "zero_gradient":
        log.write_text(log.read_text().replace("grad_norm:0.5", "grad_norm:0"))
    elif failure == "skipped":
        log.write_text(log.read_text() + "grad_norm is not finite: skipping this optimizer step.\n")
    else:
        (root / "global_step_3/actor/optim_world_size_1_rank_0.pt").unlink()
    with pytest.raises(ValueError):
        check_run(log, root, [3])


def test_smoke_reward_is_bounded_and_depends_on_image():
    blank = torch.zeros(3, 32, 32, dtype=torch.uint8)
    noise = torch.randint(0, 256, (3, 32, 32), dtype=torch.uint8)
    blank_reward = compute_score(blank, {"media_kind": "image"})["score"]
    noise_reward = compute_score(noise, {"media_kind": "image"})["score"]
    assert 0 < noise_reward < blank_reward <= 1
