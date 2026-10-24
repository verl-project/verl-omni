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
"""CPU tests for the fixed validation-visualization holdout.

``val-core`` is a scalar mean over the whole val parquet, so a lane can collapse into a
degenerate-but-scoring mode and barely move it. The holdout is the artifact that makes
that visible, which makes two things load-bearing: the four cases are *fixed* (so two runs
are comparable), and the TransferQueue batch names each case by its ``sample_index`` (so
the dump lands at ``sample_9001`` and not at an opaque uid).
"""

from __future__ import annotations

import numpy as np
import pytest
from tensordict import NonTensorData, NonTensorStack

from verl_omni.utils.agentic_val_viz import (
    VAL_VIZ_DATA_SOURCE,
    VAL_VIZ_PROMPT_KEY,
    AgenticValVizProvider,
    resolve_agentic_val_viz_provider,
)

# The holdout trailer the Mode (2a) recipe dumps, in order.
EXPECTED_SAMPLE_INDICES = [9001, 9002, 9003, 9004]


def _provider() -> AgenticValVizProvider:
    return resolve_agentic_val_viz_provider()


def _cell(batch, key: str, i: int):
    """Read one row of a ``TensorDict`` column the way ``AgentLoopWorkerTQ`` does.

    The worker walks ``batch.items()`` and unwraps ``NonTensorStack[i].data`` for a
    column and ``NonTensorData.data`` for a scalar. ``batch[key]`` is *not* equivalent:
    tensordict unwraps a non-tensor column to its internal ``LinkedList`` (no
    ``.tolist()``/indexing of rows), so the worker's indexing is the contract to pin.
    """
    column = batch.get(key)
    if isinstance(column, NonTensorStack):
        return column[i].data
    return getattr(column, "data", column)


def _scalar(batch, key: str):
    """Read a ``NonTensorData`` scalar off the batch (``batch[key]`` already unwraps)."""
    return getattr(batch.get(key), "data", batch.get(key))


def test_resolver_is_off_unless_asked(monkeypatch):
    monkeypatch.delenv("AGENTIC_VAL_VIZ", raising=False)

    assert resolve_agentic_val_viz_provider() is None


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_resolver_accepts_the_usual_truthy_spellings(monkeypatch, value):
    monkeypatch.setenv("AGENTIC_VAL_VIZ", value)

    assert resolve_agentic_val_viz_provider() is not None


def test_default_holdout_is_the_cafe_pair_plus_the_cn_pair(monkeypatch):
    """9001/9002 cafe + 9003/9004 CN poster, each a reflect then a plan case."""
    monkeypatch.setenv("AGENTIC_VAL_VIZ", "1")

    provider = _provider()

    assert [case.sample_index for case in provider.cases] == EXPECTED_SAMPLE_INDICES
    assert [case.task_type for case in provider.cases] == ["reflect", "plan", "reflect", "plan"]
    # The two CN cases share one poster task, and only the plan case carries it as a
    # reference subtask -- that is what makes the pair a reflection/plan comparison.
    cn_tasks = {case.user_request for case in provider.cases[2:]}
    assert len(cn_tasks) == 1
    assert provider.cases[1].reference_subtasks and provider.cases[3].reference_subtasks
    assert provider.cases[0].reference_subtasks is None
    assert all(case.expected_num_images == 1 for case in provider.cases)


def test_provider_requires_at_least_one_case():
    with pytest.raises(ValueError, match="at least one ValVizCase"):
        AgenticValVizProvider([])


def test_build_batch_is_a_validate_batch_carrying_the_case_identity(monkeypatch):
    monkeypatch.setenv("AGENTIC_VAL_VIZ", "1")

    batch = _provider().build_batch(30, eos_token_id=2, pad_token_id=0)

    assert batch.meta_info["validate"] is True
    assert batch.meta_info["global_steps"] == 30
    assert list(batch.non_tensor_batch["index"]) == EXPECTED_SAMPLE_INDICES
    assert set(batch.non_tensor_batch["data_source"]) == {VAL_VIZ_DATA_SOURCE}
    assert [messages[-1]["content"] for messages in batch.non_tensor_batch["raw_prompt"]]


def test_tensordict_batch_names_the_dump_folder_after_the_sample_index(monkeypatch):
    """``dataset_task_uid`` decides the artifact folder, so it must be ``9001``, not the uid.

    ``episode_artifact_keys`` resolves ``dataset_task_uid`` before ``uid`` and feeds it to
    ``build_trajectory_relpath`` as ``sample_index``. That is what turns the Mode (2a)
    reference path into ``rollout_images/val/step_000030/sample_9001.01``; keying the
    folder on the (unique per step) uid would give every step a fresh opaque folder and
    make two runs incomparable.
    """
    monkeypatch.setenv("AGENTIC_VAL_VIZ", "1")

    batch = _provider().build_tensordict_batch(30)

    assert len(batch) == len(EXPECTED_SAMPLE_INDICES)
    assert [_cell(batch, "dataset_task_uid", i) for i in range(len(batch))] == [str(i) for i in EXPECTED_SAMPLE_INDICES]
    assert all(_cell(batch, "raw_prompt", i)[-1]["role"] == "user" for i in range(len(batch)))

    from verl_omni.agent_loop.bagel_corl import episode_artifact_keys

    relpath, seed_base = episode_artifact_keys(
        dataset_task_uid=str(_cell(batch, "dataset_task_uid", 0)),
        session_id=0,
        global_steps=30,
    )
    assert relpath == "step_000030/sample_9001.01"
    # Deterministic per episode: a replay of one step must redraw the same noise.
    assert seed_base == episode_artifact_keys(dataset_task_uid="9001", session_id=0, global_steps=30)[1]


def test_tensordict_batch_uid_is_unique_per_step(monkeypatch):
    """A repeated uid would overwrite the previous step's row in TransferQueue."""
    monkeypatch.setenv("AGENTIC_VAL_VIZ", "1")
    provider = _provider()

    first = [_cell(provider.build_tensordict_batch(0), "uid", i) for i in range(len(EXPECTED_SAMPLE_INDICES))]
    again = [_cell(provider.build_tensordict_batch(0), "uid", i) for i in range(len(EXPECTED_SAMPLE_INDICES))]
    later = [_cell(provider.build_tensordict_batch(30), "uid", i) for i in range(len(EXPECTED_SAMPLE_INDICES))]
    first = [str(uid) for uid in first]
    again = [str(uid) for uid in again]
    later = [str(uid) for uid in later]

    assert first == again, "the same step must write the same keys, not duplicates"
    assert set(first).isdisjoint(later)
    assert len(set(first)) == len(first)


def test_tensordict_batch_carries_the_marker_and_the_validation_metadata(monkeypatch):
    """What the worker keys on: ``validate`` (kernel + ``val/`` prefix) and the marker.

    ``global_steps``/``validate`` are ``NonTensorData`` scalars, so they survive
    ``TensorDict.chunk`` and reach every worker's per-row ``prompt`` dict; the marker is
    what makes the worker run the episode without writing a TransferQueue row.
    """
    monkeypatch.setenv("AGENTIC_VAL_VIZ", "1")

    batch = _provider().build_tensordict_batch(12)
    chunks = batch.chunk(2)

    for chunk in chunks:
        assert len(chunk) == 2
        assert bool(_scalar(chunk, "validate")) is True
        assert int(_scalar(chunk, "global_steps")) == 12
        assert str(_scalar(chunk, VAL_VIZ_PROMPT_KEY)) == "1"
        for key, value in chunk.items():
            if isinstance(value, np.ndarray):  # pragma: no cover - guard against a bare ndarray
                raise AssertionError(f"{key} stayed a numpy array; the TQ worker cannot index it")


def test_tensordict_batch_columns_are_what_the_tq_worker_extracts(monkeypatch):
    """Every column must be indexable the way ``AgentLoopWorkerTQ`` indexes it.

    The worker walks ``batch.items()`` and, for non-tensors, unwraps ``v[i].data``. A
    column left as a bare nested list would not be chunkable, and one left as an
    ``(N, k)`` object array would hand the agent loop an array where it expects one row's
    value -- so each column is pinned to the payload the loop reads.
    """
    monkeypatch.setenv("AGENTIC_VAL_VIZ", "1")

    batch = _provider().build_tensordict_batch(7)

    rows: dict[str, list] = {}
    for key, value in batch.items():
        if isinstance(value, NonTensorStack):
            # Unwrapped exactly like the base worker does it.
            rows[key] = [value[i].data for i in range(len(value))]
        else:
            assert isinstance(value, NonTensorData), f"{key} is neither a column nor a scalar non-tensor"

    assert len(rows["raw_prompt"]) == len(EXPECTED_SAMPLE_INDICES)
    assert [messages[-1]["content"] for messages in rows["raw_prompt"]]
    assert [int(index) for index in rows["index"]] == EXPECTED_SAMPLE_INDICES
    assert rows["data_source"] == [VAL_VIZ_DATA_SOURCE] * len(EXPECTED_SAMPLE_INDICES)
    assert [str(uid) for uid in rows["uid"]] == [f"val-viz-7-{i}" for i in EXPECTED_SAMPLE_INDICES]
    assert [str(uid) for uid in rows["dataset_task_uid"]] == [str(i) for i in EXPECTED_SAMPLE_INDICES]
    assert [info["task_type"] for info in rows["extra_info"]] == ["reflect", "plan", "reflect", "plan"]
    assert all("ground_truth" in reward_model for reward_model in rows["reward_model"])


def test_holdout_raw_prompt_reaches_the_agent_loop_as_a_message_list():
    """Each row's ``raw_prompt`` must be a ``list`` of messages after the worker unpacks it.

    ``np.array`` turned the four equal-length message lists into a ``(4, 2)`` matrix, so a
    row arrived as an ``ndarray`` and ``_messages_are_text_only`` rejected it. The Hermes
    ``<tools>`` block was then never injected, the UND model never called ``generate_image``
    (``num_gen_calls=0`` for samples 9001..9004), and no holdout PNG was written.
    """
    import os

    from tensordict.tensorclass import NonTensorData, NonTensorStack

    from verl_omni.agent_loop.bagel_corl import _messages_are_text_only
    from verl_omni.utils.agentic_val_viz import resolve_agentic_val_viz_provider

    os.environ["AGENTIC_VAL_VIZ"] = "1"
    batch = resolve_agentic_val_viz_provider().build_tensordict_batch(0)
    column = batch["raw_prompt"]
    # Mirror ``AgentLoopWorkerTQ.generate_sequences``' per-row extraction.
    rows = [column[i].data if isinstance(column, NonTensorStack) else column[i] for i in range(len(batch))]
    rows = [row.data if isinstance(row, NonTensorData) else row for row in rows]

    assert len(rows) == 4
    for row in rows:
        assert isinstance(row, list), type(row)
        assert [m["role"] for m in row] == ["system", "user"]
        assert _messages_are_text_only(row)
