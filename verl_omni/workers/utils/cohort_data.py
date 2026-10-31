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

"""Value identities for actual cohort rows, independent of model and modality.

Only materialized CPU values are accepted. No file is opened, CUDA transfer is
hidden, or object repr/pickle used as an identity. Unsupported media objects
must take the ordinary path before a cohort is reserved.
"""

import hashlib
import json
import math
from dataclasses import dataclass, replace

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.tensorclass import NonTensorData, NonTensorStack

from verl_omni.workers.utils.cohort_lease import ShardIdentity


def value_digest(value) -> str:
    """Hash typed values, including jagged tensors and materialized media."""
    digest = hashlib.sha256()

    def emit(tag, payload=b""):
        digest.update(tag.encode() + b":" + str(len(payload)).encode() + b":" + payload)

    def visit(item):
        if isinstance(item, NonTensorStack):
            visit(item.tolist())
        elif isinstance(item, NonTensorData):
            visit(item.data)
        elif isinstance(item, TensorDictBase):
            visit(dict(item.items()))
        elif isinstance(item, torch.Tensor):
            if item.device.type != "cpu":
                raise ValueError("Cohort identity requires materialized CPU tensors")
            if item.is_nested:
                emit("nested", str(item.dtype).encode())
                visit(tuple(item.unbind()))
            else:
                if item.layout != torch.strided:
                    raise TypeError("Unsupported tensor layout for cohort identity")
                tensor = item.detach().contiguous()
                emit("tensor", str(tensor.dtype).encode())
                visit(tuple(tensor.shape))
                emit("bytes", tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            emit("ndarray", str(item.dtype).encode())
            visit(tuple(item.shape))
            if item.dtype.hasobject:
                visit(item.tolist())
            else:
                emit("bytes", np.ascontiguousarray(item).tobytes())
        elif isinstance(item, np.generic):
            visit(item.item())
        elif isinstance(item, dict):
            if not all(isinstance(key, str) for key in item):
                raise TypeError("Cohort mappings require string keys")
            emit("dict", str(len(item)).encode())
            for key in sorted(item):
                visit(key)
                visit(item[key])
        elif isinstance(item, tuple | list):
            emit(type(item).__name__, str(len(item)).encode())
            for element in item:
                visit(element)
        elif isinstance(item, bytes):
            emit("bytes", item)
        elif item is None or isinstance(item, str | bool | int | float):
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError("Nonfinite cohort metadata")
            emit(type(item).__name__, json.dumps(item).encode())
        else:
            raise TypeError(f"Unsupported cohort value: {type(item)}")

    visit(value)
    return digest.hexdigest()


def row_fields(data: TensorDict) -> tuple[str, ...]:
    """Return every actual row field, excluding scalar execution metadata."""
    return tuple(sorted(key for key, value in data.items() if not isinstance(value, NonTensorData)))


def _row_digests(data, fields):
    if not fields or len(set(fields)) != len(fields) or any(field not in data for field in fields):
        raise ValueError("Missing or duplicate cohort fields")
    # Unbind actual columns before hashing; this gives the same logical row
    # whether native DP dispatch selected a shard or reordered complete rows.
    columns = {}
    for field in fields:
        value = data.get(field)
        if isinstance(value, NonTensorStack):
            rows = value.tolist()
        elif isinstance(value, torch.Tensor | TensorDictBase):
            rows = value.unbind()
        else:
            raise TypeError(f"Not a materialized row field: {field}")
        if len(rows) != len(data):
            raise ValueError("Cohort column has the wrong number of rows")
        columns[field] = rows
    return tuple(value_digest({field: columns[field][i] for field in fields}) for i in range(len(data)))


@dataclass(frozen=True)
class CohortData:
    """Immutable expected data, prepared before dispatch and checked afterwards.

    ``keys`` must come from the actual transport metadata. ``execution`` is a
    resolved config/shape description independently produced at both sides;
    a trainer step counter is not an execution or parameter identity.
    """

    keys: tuple[str, ...]
    raw_fields: tuple[str, ...]
    raw_rows: tuple[str, ...]
    execution_digest: str
    derived_fields: tuple[str, ...] = ()
    derived_rows: tuple[str, ...] = ()

    @classmethod
    def prepare(cls, data: TensorDict, keys, execution):
        keys = tuple(keys)
        if len(keys) != len(data) or not keys or len(set(keys)) != len(keys):
            raise ValueError("Require one distinct transport key per actual row")
        if any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("Invalid transport key")
        fields = row_fields(data)
        return cls(keys, fields, _row_digests(data, fields), value_digest(execution))

    def bind_derived(self, data: TensorDict, keys, execution, fields):
        """Bind advantages/returns without blessing a change to the raw input."""
        fields = tuple(sorted(fields))
        if self.derived_fields or not fields or set(fields) & set(self.raw_fields):
            raise ValueError("Derived fields must be new and bound exactly once")
        self._check_raw(data, keys, execution)
        if set(row_fields(data)) != set(self.raw_fields) | set(fields):
            raise ValueError("Unexpected or missing row fields while binding derived data")
        return replace(self, derived_fields=fields, derived_rows=_row_digests(data, fields))

    def shard(self, indices):
        """Follow native DP dispatch using its actual row indices, not a guessed split."""
        indices = tuple(indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("Empty or duplicate shard indices")
        if any(type(index) is not int or index < 0 or index >= len(self.keys) for index in indices):
            raise ValueError("Invalid cohort shard index")
        return replace(
            self,
            keys=tuple(self.keys[i] for i in indices),
            raw_rows=tuple(self.raw_rows[i] for i in indices),
            derived_rows=tuple(self.derived_rows[i] for i in indices) if self.derived_rows else (),
        )

    def _check_raw(self, data, keys, execution):
        if tuple(keys) != self.keys or len(data) != len(self.keys):
            raise ValueError("Actual ordered transport keys changed")
        if value_digest(execution) != self.execution_digest:
            raise ValueError("Actual execution configuration changed")
        if _row_digests(data, self.raw_fields) != self.raw_rows:
            raise ValueError("Actual raw cohort data changed")

    def identity(self) -> ShardIdentity:
        """Expected identity after derived rows have been bound and verified."""
        if not self.derived_fields:
            raise ValueError("Derived fields have not been bound")
        return ShardIdentity(
            self.keys,
            value_digest((self.raw_fields, self.keys, self.raw_rows)),
            value_digest((self.derived_fields, self.keys, self.derived_rows)),
            self.execution_digest,
        )

    def observe(self, data: TensorDict, keys, execution) -> ShardIdentity:
        """Build the lease identity from actual data after checking all expected rows."""
        self._check_raw(data, keys, execution)
        if not self.derived_fields:
            raise ValueError("Derived fields have not been bound")
        if set(row_fields(data)) != set(self.raw_fields) | set(self.derived_fields):
            raise ValueError("Unexpected or missing row fields at the worker")
        derived = _row_digests(data, self.derived_fields)
        if derived != self.derived_rows:
            raise ValueError("Actual derived cohort data changed")
        return ShardIdentity(
            self.keys,
            value_digest((self.raw_fields, self.keys, self.raw_rows)),
            value_digest((self.derived_fields, self.keys, derived)),
            self.execution_digest,
        )
