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

"""Validate extension metadata before the TransferQueue bridge merges fields."""

from functools import wraps

from verl.utils.transferqueue_utils import BatchMeta, KVBatchMeta, kv_batch_meta2batch_meta

SELECTOR_KEY = "_step_extension_selector"


def step_extension_transport(method):
    """Reject stored fields that would be shadowed by the extension selector."""

    @wraps(method)
    def wrapped(self, data):
        if not isinstance(data, BatchMeta | KVBatchMeta):
            raise TypeError("Worker extension requires dispatched batch metadata")
        batch = kv_batch_meta2batch_meta(data) if isinstance(data, KVBatchMeta) else data
        if SELECTOR_KEY in batch.field_names:
            raise ValueError("Stored field collides with the worker extension selector")
        return method(self, data)

    return wrapped
