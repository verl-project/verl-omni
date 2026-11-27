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
client its thinker-only duplex deploy config), the calibration gate's
ceiling argument, the episode build's stable dialogue split and per-split
axis balancing, and the parity check's two-stage pre-truncation comparison (the measure
itself lives in ``verl_omni.utils.dataset.duplex.policy_measure``).
The scripts load via importlib so ``--help`` stays import-pure.
"""

from __future__ import annotations

import importlib.util
import json
import zlib
from collections import defaultdict
from pathlib import Path

import pandas as pd
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


def test_write_duplex_deploy_config_produces_the_thinker_only_duplex_layout(tmp_path, monkeypatch):
    monkeypatch.delenv("VERL_OMNI_DUPLEX_MAX_SESSIONS", raising=False)
    path = write_duplex_deploy_config(tmp_path / "nested" / "deploy.yaml")
    assert Path(path).exists()
    deploy = yaml.safe_load(Path(path).read_text())
    assert deploy["pipeline"] == "minicpmo_4_5_duplex"
    assert deploy["session_mode"] == "duplex"
    assert deploy["model_runner"] == "v2"
    assert deploy["duplex_session"] == {
        "max_sessions": 16,
        "max_pending_output_events_per_session": 8192,
        "max_pending_output_bytes_per_session": 32 * 1024 * 1024,
    }
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
def test_build_engine_disables_prefix_caching(script_name, monkeypatch, tmp_path):
    script = _load_script(script_name)
    seen: dict = {}

    def recorder(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(
        "verl_omni.pipelines.minicpm.duplex_rollout_adapter.require_duplex_vllm_omni",
        lambda: recorder,
    )
    args = script._parse_args(
        ["--model", "unused-model", "--episodes", "unused", "--output", str(tmp_path / "out.json")]
    )
    assert script._build_engine(args) is not None
    assert seen["enable_prefix_caching"] is False
    assert seen["model"] == "unused-model"
    assert seen["enforce_eager"] is True


@pytest.mark.parametrize("script_name", ["duplex_calibration_audit", "duplex_m0_parity_check"])
def test_script_deploy_config_path_defaults_next_to_the_output(script_name, tmp_path):
    script = _load_script(script_name)
    args = script._parse_args(["--model", "unused", "--episodes", "unused", "--output", str(tmp_path / "out.json")])
    path = script._duplex_deploy_config_path(args)
    assert path == str(tmp_path / "out.deploy.yaml")
    deploy = yaml.safe_load(Path(path).read_text())
    assert deploy["session_mode"] == "duplex"


# ---------------------------------------------------------------------------
# Episode build (scripts/duplex_build_episodes.py)
# ---------------------------------------------------------------------------

_VAL_DIALOGUES = {"dd-0007", "dd-0012", "dd-0019"}  # crc32 buckets 44, 42, 26: held out at 5%
_TRAIN_DIALOGUES = {f"dd-{index:04d}" for index in range(6)}  # buckets >= 50


def _manifest(path: Path) -> Path:
    """Nine dialogues mining two pauses and two turns each, plus one interruption
    for the five with model speech overlapping the first user utterance."""
    dialogues = []
    for dialogue_id in sorted(_TRAIN_DIALOGUES | _VAL_DIALOGUES):
        with_model_speech = dialogue_id in {"dd-0000", "dd-0002", "dd-0004", "dd-0007", "dd-0019"}
        dialogues.append(
            {
                "dialogue_id": dialogue_id,
                "user_audio": f"/data/user/{dialogue_id}.wav",
                "user_words": [
                    {"text": "a", "start_s": start, "end_s": end}
                    for start, end in [(1.0, 2.0), (2.6, 3.0), (6.0, 7.0), (7.7, 8.0)]
                ],
                "model_words": [{"text": "m", "start_s": 1.5, "end_s": 2.5}] if with_model_speech else [],
            }
        )
    manifest = path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(dialogue) + "\n" for dialogue in dialogues), encoding="utf-8")
    return manifest


def _axes(frame: pd.DataFrame) -> dict[str, int]:
    return frame["episode"].map(lambda spec: spec["axis"]).value_counts().to_dict()


def test_build_split_is_stable_per_dialogue_id():
    script = _load_script("duplex_build_episodes")
    # The split is a pure function of the dialogue id — crc32 against the
    # fraction — so every process agrees regardless of PYTHONHASHSEED.
    for dialogue_id in ["dd-0000", "dd-0007", "dt-9999"]:
        expected = zlib.crc32(dialogue_id.encode("utf-8")) % 1000 < 50
        assert script._is_val(dialogue_id, 0.05) is expected
    assert script._is_val("dd-0007", 0.0) is False
    val_share = sum(script._is_val(f"dd-{index:04d}", 0.05) for index in range(2000)) / 2000
    assert 0.03 < val_share < 0.08


def test_build_rows_split_by_dialogue_and_carry_the_dialogue_id(tmp_path):
    script = _load_script("duplex_build_episodes")
    args = script._parse_args(["--manifest", "unused", "--output", "unused"])
    rows = script.build_rows(_manifest(tmp_path), args)
    flags_by_dialogue: dict[str, set[bool]] = defaultdict(set)
    for row in rows:
        flags_by_dialogue[row["dialogue_id"]].add(row["is_val"])
    assert all(len(flags) == 1 for flags in flags_by_dialogue.values())  # no dialogue straddles the split
    assert {row["dialogue_id"] for row in rows if row["is_val"]} == _VAL_DIALOGUES
    assert all(row["commit"] is True for row in rows)
    assert all(row["episode"]["start_s"] >= 0 for row in rows)
    # Miner defaults thread through: the 0.6 s and 0.7 s pauses, the 2 s clips.
    assert script._parse_args(["--manifest", "x", "--output", "y"]).per_axis_cap is None  # auto
    assert script._parse_args(["--manifest", "x", "--output", "y", "--per-axis-cap", "0"]).per_axis_cap == 0
    assert script._parse_args(["--manifest", "x", "--output", "y", "--per-axis-cap", "7"]).per_axis_cap == 7
    assert (args.min_pause_s, args.max_pause_s, args.min_interrupt_s) == (0.6, 1.8, 1.5)


def test_build_auto_cap_balances_the_axes_inside_each_split(tmp_path):
    script = _load_script("duplex_build_episodes")
    manifest = _manifest(tmp_path)
    output = tmp_path / "episodes.parquet"
    argv = ["--manifest", str(manifest), "--output", str(output)]
    assert script.main(argv) == 0
    train, val = (pd.read_parquet(output.with_name(f"episodes_{split}.parquet")) for split in ("train", "val"))
    # Train mines 12 pauses / 12 turns / 3 interruptions and val 6 / 6 / 2;
    # auto caps every axis to the split's smallest, so both are 1:1:1.
    assert _axes(train) == {"pause": 3, "turn": 3, "interruption": 3}
    assert _axes(val) == {"pause": 2, "turn": 2, "interruption": 2}
    assert set(train["dialogue_id"]).isdisjoint(_VAL_DIALOGUES)
    assert set(val["dialogue_id"]) <= _VAL_DIALOGUES
    assert "is_val" not in train.columns and "dialogue_id" in train.columns
    # 0 = no cap keeps the imbalanced mix; an integer caps each axis during
    # mining (3 interruptions exist, so that axis stays below its cap).
    uncapped = tmp_path / "uncapped.parquet"
    script.main([*argv[:-1], str(uncapped), "--per-axis-cap", "0"])
    assert _axes(pd.read_parquet(uncapped.with_name("uncapped_train.parquet"))) == {
        "pause": 12,
        "turn": 12,
        "interruption": 3,
    }
    capped = tmp_path / "capped.parquet"
    script.main([*argv[:-1], str(capped), "--per-axis-cap", "4"])
    assert _axes(pd.read_parquet(capped.with_name("capped_train.parquet"))) == {
        "pause": 4,
        "turn": 4,
        "interruption": 3,
    }


def test_build_rejects_a_pause_band_that_is_inverted_or_a_non_numeric_cap():
    script = _load_script("duplex_build_episodes")
    with pytest.raises(SystemExit):
        script._parse_args(["--manifest", "x", "--output", "y", "--min-pause-s", "2.0", "--max-pause-s", "1.0"])
    with pytest.raises(SystemExit):
        script._parse_args(["--manifest", "x", "--output", "y", "--per-axis-cap", "many"])


def test_calibration_script_exposes_the_two_sided_gate():
    script = _load_script("duplex_calibration_audit")
    argv = ["--model", "unused", "--episodes", "unused", "--output", "unused"]
    assert script._parse_args(argv).max_mean_ratio == pytest.approx(0.80)
    assert script._parse_args([*argv, "--max-mean-ratio", "0.9"]).max_mean_ratio == pytest.approx(0.9)


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


def test_actor_logits_reads_every_action_token_of_each_unit():
    """One logit row per action token, each read at its predicting position.

    The fake forward's row ``t`` carries the token after position ``t``: the
    rows the gate reads back must name the replayed action tokens, in unit
    then token order — a spoken unit's content tokens and boundary included.
    """
    from types import SimpleNamespace

    import numpy as np
    import torch

    from tests.utils.dataset.duplex._duplex_fixtures import listening_then_speaking_wires
    from tests.utils.dataset.duplex.test_replay_on_cpu import _vocabulary
    from verl_omni.utils.dataset.duplex.replay import build_replay
    from verl_omni.utils.dataset.duplex.trajectory import AXIS_TURN, DuplexEpisodeEvent, DuplexTrajectory

    trajectory = DuplexTrajectory.assemble(
        session_id="duplex-abc",
        seed=7,
        event=DuplexEpisodeEvent(axis=AXIS_TURN, user_turn_end_s=2.0),
        record_set_wires=listening_then_speaking_wires(),
        client_audio_chunks=[np.zeros(16000, dtype=np.float32) for _ in range(4)],
    )
    replay = build_replay(trajectory, _vocabulary())
    sequence = np.asarray(replay.prompt_ids + replay.response_ids)
    next_token = np.concatenate([sequence[1:], [-1]])  # the token row t predicts
    logits = torch.from_numpy(next_token).float().reshape(1, -1, 1)
    forward = lambda **model_inputs: SimpleNamespace(logits=logits)  # noqa: E731

    rows = _parity_script()._actor_logits(forward, {}, replay)
    assert rows[:, 0].tolist() == [float(token_id) for token_id in replay.action_token_ids()]
    assert rows.shape[0] == 8  # 1 + 1 listen tokens, 3 + 3 tokens of the two spoken units


def _barge_in_search(statuses_by_index, axes, **kwargs):
    """Run ``_select_and_roll`` over fake episodes; each index's first roll ends with its scripted status."""
    rolled: list[int] = []
    dataset = [{"duplex_episode": {"axis": axis}} for axis in axes]

    def roll_pair(sample, index):
        rolled.append(index)
        result = {"episode_id": f"ep-{index}", "status": statuses_by_index.get(index, "completed")}
        return result, result

    pairs = _parity_script()._select_and_roll(dataset, roll_pair, **kwargs)
    return rolled, pairs


def test_barge_in_search_rolls_turn_and_interruption_episodes_until_one_barges_in():
    axes = ["pause"] * 4 + ["pause", "turn", "pause", "interruption", "turn", "turn"]
    rolled, pairs = _barge_in_search(
        {7: "truncated_interruption"}, axes, num_episodes=4, max_episodes=64, require_barge_in=True
    )
    # The pause episodes past the first four are skipped; the search ends at the barge-in.
    assert rolled == [0, 1, 2, 3, 5, 7]
    assert pairs[-1][0]["episode_id"] == "ep-7"


def test_barge_in_search_is_bounded_and_off_without_the_flag():
    axes = ["turn"] * 10
    rolled, _ = _barge_in_search({}, axes, num_episodes=2, max_episodes=5, require_barge_in=True)
    assert rolled == [0, 1, 2, 3, 4]
    rolled, _ = _barge_in_search({}, axes, num_episodes=2, max_episodes=5, require_barge_in=False)
    assert rolled == [0, 1]
    # A barge-in inside the first episodes ends the search right after them.
    rolled, _ = _barge_in_search(
        {1: "truncated_interruption"}, axes, num_episodes=2, max_episodes=5, require_barge_in=True
    )
    assert rolled == [0, 1]


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
