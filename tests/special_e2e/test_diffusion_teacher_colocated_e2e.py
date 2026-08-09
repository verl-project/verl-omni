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
"""Env-gated e2e test for the fused diffusion teacher runtime.

The teacher is fused with the actor inside ``ActorRolloutRefWorker``, the same
way the reference model is. This test drives ``run_diffusion_teacher_smoke.sh``
through the real trainer, so the fused actor + reference + teacher init and the
once-per-step teacher hook are asserted, not just described.

It needs a real SD3 checkpoint pair because the fused path also builds rollout.
The test therefore *skips* unless a GPU is present and
``MODEL_PATH``/``TEACHER_PATH`` point at real checkpoints -- the CI runners
have neither, which is why this is not in the auto-run smoke group.

Run:  MODEL_PATH=<sd3.5> TEACHER_PATH=<sd3.5-teacher> \
        pytest tests/special_e2e/test_diffusion_teacher_colocated_e2e.py
"""

import os
import subprocess
import sys

import pytest
import torch

MODEL_PATH = os.environ.get("MODEL_PATH")
TEACHER_PATH = os.environ.get("TEACHER_PATH")

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and MODEL_PATH and TEACHER_PATH),
    reason="needs a GPU and a real SD3 checkpoint pair via MODEL_PATH/TEACHER_PATH",
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SMOKE = os.path.join(REPO_ROOT, "tests/special_e2e/run_diffusion_teacher_smoke.sh")


def _run_smoke(mode):
    """One-step colocated run of the given smoke mode; returns its stdout."""
    env = {**os.environ, "SMOKE": mode, "TOTAL_TRAIN_STEPS": "1", "NUM_GPUS": "1"}
    proc = subprocess.run(
        ["bash", SMOKE],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0:  # surface the failure so the log is readable in CI output
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
    assert proc.returncode == 0, f"{mode} smoke exited {proc.returncode}"
    return proc.stdout


def test_colocated_pure_distillation():
    """Actor and teacher fused in one worker: the run completes and distill_kl actually fires."""
    out = _run_smoke("distill")
    # the target was consumed by the loss, not merely wired up
    assert "actor/distill_kl_loss" in out
    # a teacher forward ran as its own timed stage
    assert "timing_s/teacher" in out


def test_colocated_three_identities():
    """Actor + reference + teacher live at once: both penalties and both forwards appear."""
    out = _run_smoke("coexistence")
    assert "actor/distill_kl_loss" in out
    assert "actor/kl_loss" in out  # the KL-to-reference penalty coexists with distillation
    assert "timing_s/teacher" in out
    assert "timing_s/ref" in out  # reference and teacher are independent forwards, not one slot
