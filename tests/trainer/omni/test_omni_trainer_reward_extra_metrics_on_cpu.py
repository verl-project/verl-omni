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
"""CPU tests for the omni trainer's reward-extra metrics aggregation.

The duplex scorer's numeric trip-wire columns (and any other scorer's
returned keys) must reach the step metrics: means with NaN-skip, and
``inf``/NaN shares surfaced as ``<key>_never_share`` instead of silently
averaged away.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _infos(*dicts):
    return list(dicts)


def test_numeric_columns_average_per_step():
    from verl_omni.trainer.omni.trainer_base import _reward_extra_step_metrics

    infos = _infos(
        {"duplex_speak_ratio": 0.4, "duplex_num_units": 5},
        {"duplex_speak_ratio": 0.6, "duplex_num_units": 7},
    )
    metrics = _reward_extra_step_metrics(infos)
    assert metrics["reward/duplex_speak_ratio"] == pytest.approx(0.5)
    assert metrics["reward/duplex_num_units"] == pytest.approx(6.0)


def test_nan_values_are_skipped_from_the_mean_but_shared():
    from verl_omni.trainer.omni.trainer_base import _reward_extra_step_metrics

    metrics = _reward_extra_step_metrics(_infos({"duplex_turn_latency_s": 1.0}, {"duplex_turn_latency_s": None}, {}))
    # None entries are absent from the column; nothing non-finite stays.
    assert metrics == {"reward/duplex_turn_latency_s": pytest.approx(1.0)}


def test_nan_column_values_skip_and_surface_a_share():
    import math

    from verl_omni.trainer.omni.trainer_base import _reward_extra_step_metrics

    metrics = _reward_extra_step_metrics(
        _infos({"latency": 1.0}, {"latency": math.nan}, {"latency": 2.0}, {"latency": math.nan})
    )
    assert metrics["reward/latency"] == pytest.approx(1.5)
    assert metrics["reward/latency_never_share"] == pytest.approx(0.5)


def test_inf_values_surface_as_never_share():
    from verl_omni.trainer.omni.trainer_base import _reward_extra_step_metrics

    metrics = _reward_extra_step_metrics(
        _infos(
            {"duplex_yield_latency_s": float("inf")},
            {"duplex_yield_latency_s": 0.4},
            {"duplex_yield_latency_s": float("inf")},
            {"duplex_yield_latency_s": 0.8},
        )
    )
    # A mean containing inf is meaningless; the finite values average and the
    # never-yielded share stays visible.
    assert metrics["reward/duplex_yield_latency_s"] == pytest.approx(0.6)
    assert metrics["reward/duplex_yield_latency_s_never_share"] == pytest.approx(0.5)


def test_all_inf_column_reports_only_the_share():
    from verl_omni.trainer.omni.trainer_base import _reward_extra_step_metrics

    metrics = _reward_extra_step_metrics(
        _infos({"duplex_turn_latency_s": float("inf")}, {"duplex_turn_latency_s": float("inf")})
    )
    assert metrics == {"reward/duplex_turn_latency_s_never_share": 1.0}


def test_non_numeric_entries_are_ignored():
    from verl_omni.trainer.omni.trainer_base import _reward_extra_step_metrics

    metrics = _reward_extra_step_metrics(
        _infos(
            {"duplex_axis": "turn", "ok": True, "score": 3},
            {"duplex_axis": "pause", "ok": False, "score": "bad"},
        )
    )
    # Strings and bools are not numeric columns; ints are.
    assert metrics == {"reward/score": 3.0}


def test_sparse_columns_average_over_the_samples_that_carry_them():
    from verl_omni.trainer.omni.trainer_base import _reward_extra_step_metrics

    infos = _infos(
        {"duplex_pause_violation": 1.0, "duplex_pacing_corrupted": 0.0},  # pause-axis episode
        {"duplex_yield_latency_s": float("inf"), "duplex_pacing_corrupted": 0.0},  # interruption-axis episode
        {"duplex_episode_failed": 1.0, "duplex_pacing_corrupted": 1.0},  # dropped (pacing corrupted)
    )
    metrics = _reward_extra_step_metrics(infos)
    assert metrics["reward/duplex_pause_violation"] == pytest.approx(1.0)
    assert metrics["reward/duplex_yield_latency_s_never_share"] == 1.0
    assert metrics["reward/duplex_pacing_corrupted"] == pytest.approx(1 / 3)  # the drop rate


def test_reward_extra_metrics_reads_the_batch(monkeypatch):
    import transfer_queue as tq

    captured: dict = {}

    def fake_kv_batch_get(*, keys, partition_id, select_fields):
        captured["keys"] = keys
        captured["partition_id"] = partition_id
        captured["select_fields"] = select_fields
        extra_fields = [
            {
                "reward_extra_info": {
                    "duplex_speak_ratio": 0.4,
                    "duplex_episode_failed": 1.0,
                    "duplex_pacing_corrupted": 1.0,
                }
            },
            {
                "reward_extra_info": {
                    "duplex_speak_ratio": 0.9,
                    "duplex_episode_failed": 0.0,
                    "duplex_pacing_corrupted": 0.0,
                }
            },  # padding row
            {"reward_extra_info": {}},
        ]
        return SimpleNamespace(pop=lambda key: SimpleNamespace(tolist=lambda: extra_fields))

    monkeypatch.setattr(tq, "kv_batch_get", fake_kv_batch_get)
    from verl_omni.trainer.omni import OmniPPOTrainerSync  # the concrete trainer subclass

    trainer = OmniPPOTrainerSync.__new__(OmniPPOTrainerSync)
    batch = SimpleNamespace(
        keys=["k0", "k1", "k2"],
        partition_id="train",
        tags=[{"is_padding": False}, {"is_padding": True}, {"is_padding": False}],
    )
    metrics = trainer._reward_extra_metrics(batch)
    assert captured["keys"] == ["k0", "k1", "k2"]
    assert captured["select_fields"] == ["extra_fields"]
    assert metrics["reward/duplex_speak_ratio"] == pytest.approx(0.4)  # the padding row is excluded
    assert metrics["reward/duplex_episode_failed"] == pytest.approx(1.0)
    # The pacing-corruption drop rate: one dropped episode among the non-padding rows.
    assert metrics["reward/duplex_pacing_corrupted"] == pytest.approx(1.0)


def test_compute_metrics_folds_reward_extras_around_verl(monkeypatch):
    import transfer_queue as tq
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    def fake_kv_batch_get(*, keys, partition_id, select_fields):
        extra_fields = [{"reward_extra_info": {"duplex_speak_ratio": 0.4}}]
        return SimpleNamespace(pop=lambda key: SimpleNamespace(tolist=lambda: extra_fields))

    monkeypatch.setattr(tq, "kv_batch_get", fake_kv_batch_get)
    super_calls: list[tuple[int, int]] = []

    def fake_super_compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        super_calls.append((global_steps, epoch))
        metrics["verl/metric"] = 1.0

    monkeypatch.setattr(PPOTrainer, "_compute_metrics", fake_super_compute_metrics)
    from verl_omni.trainer.omni import OmniPPOTrainerSync

    trainer = OmniPPOTrainerSync.__new__(OmniPPOTrainerSync)
    batch = SimpleNamespace(keys=["k0"], partition_id="train", tags=[{"is_padding": False}])
    metrics: dict = {}
    trainer._compute_metrics(batch, metrics, {}, 7, 3)
    assert super_calls == [(7, 3)]
    assert metrics["verl/metric"] == 1.0
    assert metrics["reward/duplex_speak_ratio"] == pytest.approx(0.4)
