# Copyright 2026 verl-omni contributors
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

"""Worker extension dispatch contracts on synthetic CPU metadata."""

import pytest
from transfer_queue import KVBatchMeta

from verl_omni.pipelines import step_extension
from verl_omni.pipelines.step_extension import SELECTOR_KEY, invoke_worker_extension, with_worker_extension_selector
from verl_omni.workers.omni_engine_workers import OmniDetachActorWorker


def test_worker_control_dispatch_preserves_rejected_preparation():
    worker = object.__new__(OmniDetachActorWorker)
    observation = {"request": "synthetic", "receipt": "synthetic"}
    worker._local_proximal_preparation = ("cohort-1", observation)

    assert worker.control_step_extension("proximal_actor_reuse", "get_preparation") is observation
    with pytest.raises(ValueError, match="differs"):
        worker.control_step_extension("proximal_actor_reuse", "discard_preparation", "wrong")
    assert worker._local_proximal_preparation == ("cohort-1", observation)
    assert worker.control_step_extension("proximal_actor_reuse", "discard_preparation", "cohort-1") is True
    with pytest.raises(RuntimeError, match="no staged preparation"):
        worker.control_step_extension("proximal_actor_reuse", "get_preparation")


def test_unknown_and_wrong_mode_do_not_change_worker_state():
    worker = object.__new__(OmniDetachActorWorker)
    worker._local_proximal_preparation = ("cohort-1", {"receipt": "synthetic"})
    with pytest.raises(ValueError, match="Unknown control operation"):
        worker.control_step_extension("proximal_actor_reuse", "update", "cohort-1")
    with pytest.raises(ValueError, match="Unknown worker extension"):
        worker.control_step_extension("unknown", "discard_preparation", "cohort-1")
    with pytest.raises(ValueError, match="Invalid worker extension dispatch mode"):
        invoke_worker_extension("proximal_actor_reuse", "unknown", "update", worker, "cohort-1")
    assert worker._local_proximal_preparation == ("cohort-1", {"receipt": "synthetic"})


def test_selector_preserves_metadata_and_rejects_collisions():
    metadata = KVBatchMeta(keys=["row-1"], tags=[{}], partition_id="train", extra_info={"temperature": 1.0})
    selected = with_worker_extension_selector(metadata, "proximal_actor_reuse", "prepare")
    assert selected is not metadata
    assert metadata.extra_info == {"temperature": 1.0}
    assert selected.extra_info == {"temperature": 1.0, SELECTOR_KEY: ("proximal_actor_reuse", "prepare")}
    with pytest.raises(ValueError, match="Caller-supplied"):
        with_worker_extension_selector(selected, "proximal_actor_reuse", "update")


def test_conflicting_step_extensions_fail_before_constructing_any(monkeypatch):
    calls = []
    factory = lambda trainer: calls.append(trainer)
    monkeypatch.setattr(
        step_extension,
        "_STEP_EXTENSIONS",
        {"one": (lambda config: True, factory), "two": (lambda config: True, factory)},
    )
    config = type("Config", (), {"trainer": {"v1": {}}})()
    trainer = type("Trainer", (), {"config": config})()
    with pytest.raises(ValueError, match="Conflicting training-step extensions"):
        step_extension.resolve_step_extension(trainer)
    assert calls == []


def test_named_selection_conflicting_with_another_opt_in_fails_before_construction(monkeypatch):
    calls = []
    factory = lambda trainer: calls.append(trainer)
    monkeypatch.setattr(
        step_extension,
        "_STEP_EXTENSIONS",
        {"legacy": (lambda config: True, factory), "named": (lambda config: False, factory)},
    )
    config = type("Config", (), {"trainer": {"v1": {"step_extension": "named"}}})()
    trainer = type("Trainer", (), {"config": config})()
    with pytest.raises(ValueError, match="Conflicting training-step extensions"):
        step_extension.resolve_step_extension(trainer)
    assert calls == []


def test_named_and_legacy_selection_of_same_extension_constructs_once(monkeypatch):
    calls = []
    factory = lambda trainer: calls.append(trainer) or "constructed"
    monkeypatch.setattr(step_extension, "_STEP_EXTENSIONS", {"selected": (lambda config: True, factory)})
    config = type("Config", (), {"trainer": {"v1": {"step_extension": "selected"}}})()
    trainer = type("Trainer", (), {"config": config})()
    assert step_extension.resolve_step_extension(trainer) == "constructed"
    assert calls == [trainer]
