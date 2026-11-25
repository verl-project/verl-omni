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
two-stage pre-truncation comparison (the measure
itself lives in ``verl_omni.utils.dataset.duplex.policy_measure``).
The scripts load via importlib so ``--help`` stays import-pure.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from verl_omni.utils.dataset.duplex.deploy_config import write_duplex_deploy_config
from verl_omni.utils.dataset.duplex.policy_measure import two_stage_logprob

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
# Parity: the two-stage pre-truncation measure (verl_omni/utils/dataset/duplex/policy_measure.py,
# compared against the capture by scripts/duplex_m0_parity_check.py)
# ---------------------------------------------------------------------------

# Vocab of 8 ids: 0 = chunk_eos, 1 = tts_pad, 2 = bad, 3-7 = content.
_CHUNK_EOS_ID = 0
_FORBIDDEN = [1, 2, 0]  # the resolver's composition order: tts_pad, bad_token_ids, chunk_eos


def _parity_script():
    return _load_script("duplex_m0_parity_check")


def _logits_row():
    import torch

    return torch.tensor([2.0, -5.0, -5.0, 0.5, 0.2, 0.0, -0.5, -1.0])


def test_two_stage_logprob_boundary_hits_take_the_raw_softmax_branch():
    import torch
    from torch.nn.functional import softmax

    row = _logits_row()
    value = two_stage_logprob(
        row, _CHUNK_EOS_ID, chunk_eos_id=_CHUNK_EOS_ID, forbidden_index=torch.tensor(_FORBIDDEN), temperature=0.7
    )
    assert value == pytest.approx(float(torch.log(softmax(row.float(), dim=-1)[_CHUNK_EOS_ID])))


def test_two_stage_logprob_content_rows_use_the_masked_stage2_joint():
    import math

    import torch
    from torch.nn.functional import log_softmax, softmax

    row = _logits_row()
    temperature = 0.7
    value = two_stage_logprob(
        row, 4, chunk_eos_id=_CHUNK_EOS_ID, forbidden_index=torch.tensor(_FORBIDDEN), temperature=temperature
    )
    p0 = float(softmax(row.float(), dim=-1)[_CHUNK_EOS_ID])
    masked = row.clone()
    masked[_FORBIDDEN] = float("-inf")
    expected = math.log1p(-p0) + float(log_softmax(masked / temperature, dim=-1)[4])
    assert value == pytest.approx(expected)
    # A forbidden token drawn through stage 2 is masked out: its mass is -inf.
    assert two_stage_logprob(
        row, 1, chunk_eos_id=_CHUNK_EOS_ID, forbidden_index=torch.tensor(_FORBIDDEN), temperature=temperature
    ) == float("-inf")


def test_two_stage_logprob_greedy_rows_compare_at_temperature_one():
    import torch

    row = _logits_row()
    forbidden_index = torch.tensor(_FORBIDDEN)
    greedy = two_stage_logprob(row, 5, chunk_eos_id=_CHUNK_EOS_ID, forbidden_index=forbidden_index, temperature=0.0)
    at_one = two_stage_logprob(row, 5, chunk_eos_id=_CHUNK_EOS_ID, forbidden_index=forbidden_index, temperature=1.0)
    assert greedy == pytest.approx(at_one)


def test_two_stage_logprob_log1m_stays_finite_near_saturation():
    import math

    import torch

    # chunk_eos dominates by 14 nats: p0 = 1 - 8e-7, the last fp32 steps below 1.
    row = torch.tensor([30.0, -100.0, -100.0, 16.0, -100.0, -100.0, -100.0, -100.0])
    value = two_stage_logprob(
        row, 3, chunk_eos_id=_CHUNK_EOS_ID, forbidden_index=torch.tensor(_FORBIDDEN), temperature=0.7
    )
    assert math.isfinite(value)
    assert value < -10


def test_parity_rows_compares_captured_values_and_skips_forced_or_reemit():
    """The gate's wiring: captured values replayed through the same measure give

    a zero delta, one perturbed captured value breaches the band, and rows
    without a draw (forced / reemit) are skipped, not compared.
    """
    import torch

    script = _parity_script()
    row = _logits_row()
    logits = torch.stack([row, row, row, row])
    forbidden_index = torch.tensor(_FORBIDDEN)

    def captured(token_id, temperature=0.7):
        return two_stage_logprob(
            row, token_id, chunk_eos_id=_CHUNK_EOS_ID, forbidden_index=forbidden_index, temperature=temperature
        )

    actions = [
        (4, "sampled", 0.7, captured(4)),
        (5, "forced", 0.7, None),
        (6, "sampled", 0.7, captured(6)),
        (7, "reemit", 0.7, None),
    ]
    parity = script._parity_rows(
        logits,
        actions,
        chunk_eos_id=_CHUNK_EOS_ID,
        forbidden_index=forbidden_index,
        max_abs_nats=0.05,
        mean_abs_nats=0.01,
    )
    assert parity["num_compared"] == 2
    assert parity["num_skipped_forced_or_reemit"] == 2
    assert parity["max_abs_delta"] == 0.0
    assert parity["mean_abs_delta"] == 0.0

    # One perturbed captured value breaches the max band.
    actions[0] = (4, "sampled", 0.7, captured(4) + 0.2)
    parity = script._parity_rows(
        logits,
        actions,
        chunk_eos_id=_CHUNK_EOS_ID,
        forbidden_index=forbidden_index,
        max_abs_nats=0.05,
        mean_abs_nats=0.01,
    )
    assert parity["max_abs_delta"] == pytest.approx(0.2)
    assert parity["max_abs_delta"] > 0.05


def test_audio_embedding_parity_reports_per_unit_streaming_vs_offline_gaps():
    """The diagnostic reads every clip twice through the patched audio path —

    streamed with ``audio_clip_kinds`` present, offline without — and reports
    one gap per streaming unit, the reference clip excluded.
    """
    import numpy as np
    import torch

    script = _parity_script()

    class _AudioModel:
        """Patched-shaped get_audio_embedding: streamed units carry a known gap."""

        def __init__(self):
            self.streamed = []  # per call: whether the data carried audio_clip_kinds

        def get_audio_embedding(self, data):
            self.streamed.append("audio_clip_kinds" in data)
            embeddings = []
            for index, frames in enumerate(torch.hstack(data["audio_feature_lens"]).tolist()):
                rows = (((frames - 1) // 2 + 1) - 5) // 5 + 1
                gap = (0.0, 0.25, 0.75)[index] if self.streamed[-1] else 0.0
                embeddings.append(torch.full((rows, 4), gap))
            return [embeddings]

    model = _AudioModel()
    # One episode: a 130-frame reference block, then two 104-frame units.
    model_inputs = {
        "data": {
            "input_ids": torch.ones(1, 40, dtype=torch.long),
            "position_ids": torch.arange(40)[None],
            "audio_features": [[np.zeros((80, frames), dtype=np.float32) for frames in (130, 102, 104)]],
            "audio_feature_lens": [[130, 102, 104]],
            "audio_bounds": [[[3, 16], [18, 28], [30, 40]]],
            "audio_clip_kinds": ["reference", "unit", "unit"],
        }
    }

    parity = script._audio_embedding_parity(model, model_inputs, has_reference=True)

    assert model.streamed == [True, False]
    assert parity == {"unit0_max_abs": 0.25, "units_ge1_max_abs": 0.75, "per_unit": [0.25, 0.75]}
    assert json.loads(json.dumps(parity)) == parity

    # Without a reference block every clip is a unit, the first one included.
    model_inputs["data"]["audio_clip_kinds"] = ["unit", "unit", "unit"]
    parity = script._audio_embedding_parity(model, model_inputs, has_reference=False)
    assert parity["per_unit"] == [0.0, 0.25, 0.75]
    assert parity["unit0_max_abs"] == 0.0 and parity["units_ge1_max_abs"] == 0.75

    # A single-unit episode has no units >= 1 to summarize.
    model_inputs["data"].update(audio_feature_lens=[[130, 102]], audio_clip_kinds=["reference", "unit"])
    model_inputs["data"]["audio_features"] = [[np.zeros((80, frames), dtype=np.float32) for frames in (130, 102)]]
    parity = script._audio_embedding_parity(model, model_inputs, has_reference=True)
    assert parity == {"unit0_max_abs": 0.25, "units_ge1_max_abs": None, "per_unit": [0.25]}
