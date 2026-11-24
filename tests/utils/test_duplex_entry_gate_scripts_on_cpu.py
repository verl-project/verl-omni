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
"""CPU tests for the duplex entry-gate scripts' shared plumbing.

Covers the deploy-config writer (the scripts hand the standalone engine
client its thinker-only duplex deploy config) and the parity check's
nucleus-cut classification. The scripts load via importlib so ``--help``
stays import-pure.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from verl_omni.utils.dataset.duplex.deploy_config import write_duplex_deploy_config
from verl_omni.utils.dataset.duplex.sampling_logprobs import DuplexSamplingContext, DuplexUnitInputs

_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Deploy-config generation (scripts/duplex_calibration_audit.py,
# scripts/duplex_m0_parity_check.py)
# ---------------------------------------------------------------------------


def test_write_duplex_deploy_config_produces_the_thinker_only_duplex_layout(tmp_path):
    path = write_duplex_deploy_config(tmp_path / "nested" / "deploy.yaml")
    assert Path(path).exists()
    deploy = yaml.safe_load(Path(path).read_text())
    assert deploy["pipeline"] == "minicpmo_4_5_duplex"
    assert deploy["session_mode"] == "duplex"
    assert [stage["stage_id"] for stage in deploy["stages"]] == [0]
    assert deploy["stages"][0]["tensor_parallel_size"] == 1
    # The engine's own parser must accept it (stage layout round-trip).
    from vllm_omni.config.stage_config import load_deploy_config

    parsed = load_deploy_config(path)
    assert parsed.session_mode == "duplex"
    assert [stage.stage_id for stage in parsed.stages] == [0]
    assert parsed.stages[0].tensor_parallel_size == 1


@pytest.mark.parametrize("script_name", ["duplex_calibration_audit", "duplex_m0_parity_check"])
def test_script_stage_init_timeout_defaults_are_env_overridable(script_name, monkeypatch, tmp_path):
    script = _load_script(script_name)
    argv = ["--model", "unused", "--episodes", "unused", "--output", str(tmp_path / "out.json")]
    args = script._parse_args(argv)
    assert args.stage_init_timeout == 1800
    assert args.tensor_parallel_size == 1  # bare top-level TP>1 is rejected by the multi-stage layout
    monkeypatch.setenv("VERL_OMNI_DUPLEX_STAGE_INIT_TIMEOUT", "900")
    assert script._parse_args(argv).stage_init_timeout == 900


@pytest.mark.parametrize("script_name", ["duplex_calibration_audit", "duplex_m0_parity_check"])
def test_script_deploy_config_path_defaults_next_to_the_output(script_name, tmp_path):
    script = _load_script(script_name)
    args = script._parse_args(["--model", "unused", "--episodes", "unused", "--output", str(tmp_path / "out.json")])
    path = script._duplex_deploy_config_path(args)
    assert path == str(tmp_path / "out.deploy.yaml")
    deploy = yaml.safe_load(Path(path).read_text())
    assert deploy["session_mode"] == "duplex"


# ---------------------------------------------------------------------------
# Parity: nucleus-cut classification (scripts/duplex_m0_parity_check.py)
# ---------------------------------------------------------------------------


def _parity_script():
    return _load_script("duplex_m0_parity_check")


def _context() -> DuplexSamplingContext:
    return DuplexSamplingContext(
        chunk_eos_token_id=10,
        tts_pad_token_id=11,
        bad_token_ids=(),
        repetition_penalty=1.0,
        repetition_history_tokens=8,
    )


def _unit(token_id: int, **overrides) -> DuplexUnitInputs:
    fields = {
        "token_id": token_id,
        "action_kind": "sampled",
        "stream": "content",
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 0.8,
    }
    fields.update(overrides)
    return DuplexUnitInputs(**fields)


def test_parity_verdict_aligns_captured_with_sampled_rows_and_uses_the_engine_history(monkeypatch):
    """The script's parity wiring: forced rows bring no captured value, nucleus

    membership comes from the shared classifier, and the recompute runs over the
    engine's own repetition history.
    """
    import math

    import torch

    from verl_omni.utils.dataset.duplex import sampling_logprobs
    from verl_omni.utils.dataset.duplex.sampling_logprobs import engine_repetition_history

    script = _parity_script()
    context = _context()
    # softmax mass: token 1 ≈ .55, token 2 ≈ .45 (cumulative crosses top_p=0.8
    # at token 2, kept), every other token ≈ 1e-9 (outside the cut).
    logits = torch.full((3, 16), -10.0)
    logits[:, 1] = 10.0
    logits[:, 2] = math.log(0.43 / 0.53) + 10.0
    inputs = [
        _unit(2),
        _unit(1, action_kind="forced", stream="boundary"),
        _unit(5),
    ]
    recompute_calls: list = []
    real_recompute = sampling_logprobs.recompute_action_logprobs

    def recording_recompute(logits_, inputs_, context_, *, action_history=None):
        recompute_calls.append(action_history)
        return real_recompute(logits_, inputs_, context_, action_history=action_history)

    monkeypatch.setattr(sampling_logprobs, "recompute_action_logprobs", recording_recompute)
    values, _ = real_recompute(logits, inputs, context, action_history=engine_repetition_history(inputs))
    # Captured equals recomputed on the sampled rows; the forced row carries none.
    logprobs = [float(values[0]), None, float(values[2])]
    parity = script._parity_verdict(
        logits, inputs, context, [u.action_kind for u in inputs], logprobs, rel_tol=1e-3, abs_tol=1e-4
    )
    assert recompute_calls == [engine_repetition_history(inputs)]
    assert parity.to_dict()["num_sampled"] == 2
    assert parity.to_dict()["num_out_of_nucleus"] == 1
    assert [row.index for row in parity.documented_loosening] == [2]
    assert parity.matches is True
    # An in-nucleus residual beyond tolerance is a gate failure, not a loosening.
    shifted = [float(values[0]) - 0.5, None, float(values[2])]
    parity = script._parity_verdict(
        logits, inputs, context, [u.action_kind for u in inputs], shifted, rel_tol=1e-3, abs_tol=1e-4
    )
    assert [row.index for row in parity.breaches] == [0]
    assert parity.matches is False
