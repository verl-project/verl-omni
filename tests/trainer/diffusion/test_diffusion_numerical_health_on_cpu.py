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
"""CPU unit tests for the L2 diffusion numerical health verifier."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_CHECKER_PATH = Path(__file__).resolve().parents[2] / "special_e2e" / "check_diffusion_numerical_health.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_diffusion_numerical_health", _CHECKER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_checker = _load_checker()


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_healthy_jsonl_passes_and_normalizes_double_actor_prefix(tmp_path: Path) -> None:
    jsonl = _write_jsonl(
        tmp_path / "metrics.jsonl",
        [
            {
                "step": 1,
                "data": {
                    "actor/loss": 0.125,
                    "actor/actor/kl_loss": 0.002,
                    "actor/grad_norm": 1.4,
                    "actor/actor/ratio_mean": 1.00001,
                    "actor/actor/ppo_kl": 2.5e-5,
                    "rollout/latents_dtype": "torch.float32",
                },
            },
            {
                "step": 2,
                "data": {
                    "actor/loss": 0.11,
                    "actor/grad_norm": 1.1,
                },
            },
        ],
    )

    summary = _checker.check_diffusion_numerical_health(
        jsonl_path=jsonl,
        steps=[1, 2],
        step1_ratio_tol=1e-4,
        step1_kl_tol=1e-4,
        require_positive_grad_norm=True,
        require_fp32_latents=True,
    )

    assert summary["passed"] is True
    assert summary["steps"] == [1, 2]
    assert summary["checked_steps"]["1"]["actor/loss"] == pytest.approx(0.125)
    assert summary["checked_steps"]["1"]["actor/kl_loss"] == pytest.approx(0.002)
    assert summary["step1_invariants"]["actor/ratio_mean"] == pytest.approx(1.00001)
    assert summary["step1_invariants"]["actor/ppo_kl"] == pytest.approx(2.5e-5)
    assert summary["step1_invariants"]["latents_dtype"] == "torch.float32"


def test_console_log_parser_handles_ansi_and_numpy_scalar_wrappers(tmp_path: Path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "\x1b[32mstep:1\x1b[0m - actor/loss:np.float32(0.042) - actor/grad_norm:0.85 "
        "- actor/actor/ratio_mean:1.0002 - actor/actor/ppo_kl:0.0001\n",
        encoding="utf-8",
    )

    summary = _checker.check_diffusion_numerical_health(
        log_path=log,
        steps=[1],
        step1_ratio_tol=1e-3,
        step1_kl_tol=1e-3,
        require_positive_grad_norm=True,
    )

    assert summary["checked_steps"]["1"]["actor/loss"] == pytest.approx(0.042)
    assert summary["step1_invariants"]["actor/ratio_mean"] == pytest.approx(1.0002)


def test_step1_ratio_bias_is_rejected(tmp_path: Path) -> None:
    jsonl = _write_jsonl(
        tmp_path / "metrics.jsonl",
        [
            {
                "step": 1,
                "data": {
                    "actor/loss": 0.1,
                    "actor/grad_norm": 1.0,
                    "actor/ratio_mean": 0.99996,
                    "actor/ppo_kl": 1e-6,
                },
            }
        ],
    )

    with pytest.raises(ValueError, match="actor/ratio_mean"):
        _checker.check_diffusion_numerical_health(jsonl_path=jsonl, steps=[1], step1_ratio_tol=1e-5)


def test_step1_kl_spike_is_rejected(tmp_path: Path) -> None:
    jsonl = _write_jsonl(
        tmp_path / "metrics.jsonl",
        [
            {
                "step": 1,
                "data": {
                    "actor/loss": 0.1,
                    "actor/grad_norm": 1.0,
                    "actor/ratio_mean": 1.0,
                    "actor/ppo_kl": 0.12,
                },
            }
        ],
    )

    with pytest.raises(ValueError, match="actor/ppo_kl"):
        _checker.check_diffusion_numerical_health(jsonl_path=jsonl, steps=[1], step1_kl_tol=0.05)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf"), "nan"])
def test_nonfinite_loss_or_grad_norm_is_rejected(tmp_path: Path, bad_value: object) -> None:
    log = tmp_path / "train.log"
    log.write_text(f"step:1 - actor/loss:{bad_value} - actor/grad_norm:1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="actor/loss"):
        _checker.check_diffusion_numerical_health(log_path=log, steps=[1])

    log.write_text(f"step:1 - actor/loss:0.1 - actor/grad_norm:{bad_value}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="actor/grad_norm"):
        _checker.check_diffusion_numerical_health(log_path=log, steps=[1])


def test_skipped_optimizer_step_in_console_log_fails_even_with_clean_jsonl(tmp_path: Path) -> None:
    jsonl = _write_jsonl(
        tmp_path / "metrics.jsonl",
        [{"step": 1, "data": {"actor/loss": 0.1, "actor/grad_norm": 1.0}}],
    )
    log = tmp_path / "train.log"
    log.write_text("WARNING grad_norm is not finite: nan; skipping this optimizer step.\n", encoding="utf-8")

    with pytest.raises(ValueError, match="optimizer update was skipped"):
        _checker.check_diffusion_numerical_health(jsonl_path=jsonl, log_path=log, steps=[1])


def test_zero_grad_norm_and_missing_step_or_metric_fail_closed(tmp_path: Path) -> None:
    jsonl = _write_jsonl(
        tmp_path / "metrics.jsonl",
        [{"step": 1, "data": {"actor/loss": 0.1, "actor/grad_norm": 0.0}}],
    )

    with pytest.raises(ValueError, match="positive actor/grad_norm"):
        _checker.check_diffusion_numerical_health(jsonl_path=jsonl, steps=[1], require_positive_grad_norm=True)

    with pytest.raises(ValueError, match="missing \\[2\\]"):
        _checker.check_diffusion_numerical_health(jsonl_path=jsonl, steps=[1, 2])

    with pytest.raises(ValueError, match="Missing required metric 'actor/ratio_mean'"):
        _checker.check_diffusion_numerical_health(jsonl_path=jsonl, steps=[1], step1_ratio_tol=0.05)


@pytest.mark.parametrize("dtype_value", ["torch.bfloat16", "bfloat16", None])
def test_non_fp32_or_missing_latent_dtype_is_rejected(tmp_path: Path, dtype_value: str | None) -> None:
    data = {"actor/loss": 0.1, "actor/grad_norm": 1.0}
    if dtype_value is not None:
        data["rollout/latents_dtype"] = dtype_value
    jsonl = _write_jsonl(tmp_path / "metrics.jsonl", [{"step": 1, "data": data}])

    with pytest.raises(ValueError, match="float32"):
        _checker.check_diffusion_numerical_health(jsonl_path=jsonl, steps=[1], require_fp32_latents=True)


def test_cli_writes_summary_json_and_exits_nonzero_on_failure(tmp_path: Path) -> None:
    jsonl = _write_jsonl(
        tmp_path / "metrics.jsonl",
        [
            {
                "step": 1,
                "data": {
                    "actor/loss": 0.2,
                    "actor/grad_norm": 0.9,
                    "actor/ratio_mean": 1.001,
                    "actor/ppo_kl": 0.002,
                },
            }
        ],
    )
    summary_path = tmp_path / "summary.json"
    ok = subprocess.run(
        [
            sys.executable,
            str(_CHECKER_PATH),
            "--jsonl",
            str(jsonl),
            "--steps",
            "1",
            "--step1-ratio-tol",
            "0.05",
            "--step1-kl-tol",
            "0.05",
            "--summary-json",
            str(summary_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok.returncode == 0, ok.stdout + ok.stderr
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["passed"] is True
    assert saved["step1_invariants"]["actor/ratio_mean"] == pytest.approx(1.001)

    bad = subprocess.run(
        [
            sys.executable,
            str(_CHECKER_PATH),
            "--jsonl",
            str(jsonl),
            "--steps",
            "1",
            "--step1-ratio-tol",
            "0.0001",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode != 0
    assert "actor/ratio_mean" in bad.stderr
