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

"""CPU regression: GEN seed rows must never reach a sampled bagel Co-RL batch.

A dual-lane episode writes two kinds of row under one prompt uid: the UND episode row
(``prompts``/``responses``/``raw_prompt``) and one row per complete FlowGRPO seed group's
seed (``prompt_token_ids``/``all_latents``/``timesteps``), the latter tagged
``is_auxiliary`` because the GEN lane addresses them by the UND row's ``child_gen_keys``
column instead of by sampling.

Sibling ``verl`` 0.9.0.dev skips ``is_auxiliary`` rows in
``ReplayBuffer._materialize_batch``; the installed ``verl`` 0.10.0.dev has no such filter, so
the first run whose UND lane actually worked -- K >= 1, i.e. the first run that ever wrote a
GEN row -- died at step 0 (``bagel_corl_prfix_t07_20261015_183002_a1``, 18:57:03) with

    KeyError: 'key "prompts" not found in TensorDict with keys []'

in ``_compute_reward_colocate``: ``_materialize_batch`` had put the UND row and its GEN rows
in one batch, and ``BatchMeta.select_fields(["prompts", "responses", "raw_prompt"])`` keeps
only fields present across the whole batch, so ``async_get_data`` returned an empty
TensorDict. The GEN lane's own symptom, one frame later, is the sibling's
``assert len(output) == len(batch)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from verl_omni.trainer.omni.bagel_corl_replay_buffer import (
    drop_auxiliary_sampled_rows,
    install_auxiliary_row_filter,
    is_auxiliary_row,
)
from verl_omni.trainer.omni.bagel_corl_trainer import OmniBagelCoRLTrainerSync


@dataclass
class _Meta:
    """Minimal stand-in for ``transfer_queue.KVBatchMeta`` (same field names)."""

    partition_id: str = "train"
    keys: list[str] = field(default_factory=list)
    tags: list[dict] = field(default_factory=list)


UND_TAG = {"status": "success", "bagel_role": "und"}
GEN_TAG = {"status": "success", "bagel_role": "gen", "is_auxiliary": True}


def _mixed_batch() -> _Meta:
    """One UND episode row followed by its two GEN seed rows, as the writer emits them."""
    return _Meta(
        keys=["uid_0_0", "uid_0_0::gen::call1::0", "uid_0_0::gen::call1::1"],
        tags=[UND_TAG, GEN_TAG, GEN_TAG],
    )


def test_only_the_writer_flag_counts():
    """``is_auxiliary`` is the contract; a bare ``bagel_role`` must not be treated as one."""
    assert is_auxiliary_row(GEN_TAG)
    assert not is_auxiliary_row(UND_TAG)
    assert not is_auxiliary_row({"bagel_role": "gen"})
    # Padding rows built by ``upsample_batch_to_divisible_size`` copy their template's tag,
    # and a non-dict tag must not raise.
    assert not is_auxiliary_row(None)
    assert not is_auxiliary_row("gen")


def test_gen_seed_rows_are_dropped_and_the_und_row_kept():
    """The regression itself: a mixed batch must materialize as UND-only, in order.

    Reading UND fields off a batch that still contained the GEN rows is what raised
    ``KeyError: 'prompts'`` -- the batch is the input to the reward step, so this ordering and
    membership is the whole bug.
    """
    filtered = drop_auxiliary_sampled_rows(_mixed_batch())
    assert filtered.keys == ["uid_0_0"]
    assert filtered.tags == [UND_TAG]
    assert filtered.partition_id == "train"


def test_a_lone_und_batch_is_returned_untouched():
    """K = 0 (every run before the AR weights were fixed) must not pay for this filter."""
    meta = _Meta(keys=["uid_0_0"], tags=[UND_TAG])
    assert drop_auxiliary_sampled_rows(meta) is meta


def test_a_batch_that_lost_its_und_row_fails_loud():
    """An all-auxiliary selection must name itself here, not as an empty batch downstream."""
    meta = _Meta(keys=["uid_0_0::gen::call1::0"], tags=[GEN_TAG])
    with pytest.raises(RuntimeError, match="only auxiliary rows"):
        drop_auxiliary_sampled_rows(meta)


class _FakeBuffer:
    """A replay buffer shaped like the installed ``verl``: auxiliary rows are materialized."""

    def __init__(self):
        self.materialize_calls = []
        self.poll_interval = 0.05

    def _materialize_batch(self, partition_id, selected_prompt_uids, partition_snapshot):
        self.materialize_calls.append((partition_id, list(selected_prompt_uids), dict(partition_snapshot)))
        return _mixed_batch()


def test_installing_the_filter_fixes_every_buffer_regardless_of_verl():
    """The compatibility shim, exercised against a buffer that has no aux handling of its own."""
    buffer = install_auxiliary_row_filter(_FakeBuffer())
    filtered = buffer._materialize_batch("train", ["uid_0_0"], {})
    assert filtered.keys == ["uid_0_0"]
    # The base implementation still ran, unchanged, and was not bypassed.
    assert buffer.materialize_calls == [("train", ["uid_0_0"], {})]
    assert buffer.poll_interval == 0.05, "the shim must not replace the buffer"


def test_installing_twice_does_not_stack_filters():
    """``_build_replay_buffer`` may be re-entered; one layer, one drop."""
    buffer = install_auxiliary_row_filter(install_auxiliary_row_filter(_FakeBuffer()))
    assert buffer._materialize_batch("train", ["uid_0_0"], {}).keys == ["uid_0_0"]
    assert len(buffer.materialize_calls) == 1


def test_a_buffer_without_the_hook_is_left_alone(caplog):
    """A custom sampler owns its own semantics: warn, never guess."""

    class _OpaqueBuffer:
        pass

    opaque = _OpaqueBuffer()
    with caplog.at_level("WARNING"):
        assert install_auxiliary_row_filter(opaque) is opaque
    assert "no _materialize_batch" in caplog.text


def _base_defining_build_replay_buffer():
    """The class ``super()._build_replay_buffer()`` inside the hook resolves to."""
    for cls in OmniBagelCoRLTrainerSync.__mro__[1:]:
        if "_build_replay_buffer" in vars(cls):
            return cls
    raise AssertionError("no base class defines _build_replay_buffer")


def test_the_trainer_hook_installs_the_filter(monkeypatch):
    """The wiring, not just the helper: ``_build_replay_buffer`` returns a filtered buffer."""
    base = _base_defining_build_replay_buffer()
    buffer = _FakeBuffer()
    monkeypatch.setattr(base, "_build_replay_buffer", lambda self: buffer)

    trainer = OmniBagelCoRLTrainerSync.__new__(OmniBagelCoRLTrainerSync)
    trainer.trainer_mode = "bagel_corl_sync"
    built = trainer._build_replay_buffer()

    assert built is buffer
    assert trainer.trainer_mode == "bagel_corl_sync", "the sync-mode override must be restored"
    assert built._materialize_batch("train", ["uid_0_0"], {}).keys == ["uid_0_0"]
