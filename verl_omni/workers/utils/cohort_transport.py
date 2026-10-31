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

"""Bind cohort identity at the dispatched Omni RPC boundary."""

from copy import copy
from functools import wraps

from verl.utils.transferqueue_utils import BatchMeta, KVBatchMeta, batch_meta2kv_batch_meta


def cohort_transport(method):
    @wraps(method)
    def wrapped(self, data):
        if not isinstance(data, BatchMeta | KVBatchMeta):
            raise TypeError("Cohort RPC requires actual dispatched batch metadata")
        key = "_proximal_transport_identity"
        if key in data.extra_info:
            raise ValueError("Caller-supplied transport identity is forbidden")
        # Native dispatch resolves keys to storage indexes before chunking.
        # Recover this rank's actual keys from those indexes while preserving
        # the original carrier for the registered TransferQueue bridge.
        if isinstance(data, BatchMeta):
            if data.size <= 0 or len(set(data.partition_ids)) != 1:
                raise ValueError("Cohort RPC requires a nonempty single-partition shard")
            identity = batch_meta2kv_batch_meta(data)
            if len(identity.keys) != data.size or identity.partition_id != data.partition_ids[0]:
                raise ValueError("Storage keys do not match the dispatched shard")
        else:
            identity = data
        if identity.fields is not None and key in identity.fields:
            raise ValueError("Caller-supplied transport identity is forbidden")
        if not identity.partition_id or not identity.keys or len(set(identity.keys)) != len(identity.keys):
            raise ValueError("Cohort RPC requires unique keys and a single partition")
        local = copy(data)
        local.extra_info = {
            **data.extra_info,
            key: {"keys": tuple(identity.keys), "partition_id": identity.partition_id},
        }
        return method(self, local)

    return wrapped
