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
"""Compatibility patches for the pinned ``verl`` dependency."""

import os
from functools import wraps


def apply_numpy_dataproto_serialization_fix() -> None:
    """Avoid an unused TensorDict consolidation in NumPy DataProto serialization.

    This compatibility workaround is used by NPU workloads that enable NumPy
    serialization and was validated on Ascend 910C. Remove it after
    https://github.com/verl-project/verl/pull/7539 is merged and the pinned
    ``verl`` revision includes that upstream fix.

    The pinned ``verl`` version consolidates ``self.batch`` before checking the
    serialization method. The NumPy serializer works from the original batch,
    so that consolidation creates an unused full-batch copy for every RPC.
    Keep the original implementation for torch serialization and bypass it
    only for the NumPy path.
    """
    import verl.protocol as verl_protocol

    original_getstate = verl_protocol.DataProto.__getstate__
    if getattr(original_getstate, "_verl_omni_numpy_serialization_patched", False):
        return

    @wraps(original_getstate)
    def patched_getstate(self):
        if os.getenv("VERL_DATAPROTO_SERIALIZATION_METHOD") == "numpy":
            return (
                verl_protocol.serialize_tensordict(self.batch) if self.batch is not None else None,
                self.non_tensor_batch,
                self.meta_info,
            )
        return original_getstate(self)

    patched_getstate._verl_omni_numpy_serialization_patched = True
    verl_protocol.DataProto.__getstate__ = patched_getstate
