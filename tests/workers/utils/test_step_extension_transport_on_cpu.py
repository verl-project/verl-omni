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

"""Reject reserved stored fields before metadata is merged into real data."""

from types import SimpleNamespace

import pytest
from transfer_queue import KVBatchMeta

from verl_omni.workers.utils import step_extension_transport as transport


def test_all_fields_selection_checks_resolved_storage_metadata(monkeypatch):
    metadata = KVBatchMeta(keys=["row"], tags=[{}], partition_id="train", fields=None)
    calls = []
    monkeypatch.setattr(
        transport,
        "kv_batch_meta2batch_meta",
        lambda _: SimpleNamespace(field_names=["input_ids", transport.SELECTOR_KEY]),
    )

    @transport.step_extension_transport
    def stage(worker, batch):
        calls.append(batch)

    with pytest.raises(ValueError, match="Stored field collides"):
        stage(object(), metadata)
    assert calls == []


def test_non_metadata_is_rejected_before_dispatch():
    @transport.step_extension_transport
    def stage(worker, batch):
        pytest.fail("Non-metadata reached the worker")

    with pytest.raises(TypeError, match="batch metadata"):
        stage(object(), object())
