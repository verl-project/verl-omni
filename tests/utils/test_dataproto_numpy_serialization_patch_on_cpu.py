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
from types import SimpleNamespace

import verl.protocol as verl_protocol


def test_numpy_dataproto_serializes_original_batch(monkeypatch):
    sentinel = object()
    batch = object()
    calls = []

    monkeypatch.setenv("VERL_DATAPROTO_SERIALIZATION_METHOD", "numpy")
    monkeypatch.setattr(
        verl_protocol,
        "serialize_tensordict",
        lambda value: calls.append(value) or sentinel,
    )
    data = SimpleNamespace(batch=batch, non_tensor_batch={"label": [1]}, meta_info={"step": 1})

    state = verl_protocol.DataProto.__getstate__(data)

    assert calls == [batch]
    assert state == (sentinel, {"label": [1]}, {"step": 1})
