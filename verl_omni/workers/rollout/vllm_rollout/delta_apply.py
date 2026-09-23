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
"""Reassembly of ``delta_sharded`` flushes on the vllm-omni rollout worker.

The delta checkpoint engine streams per-flush sparse payloads; the server adapter
forwards each flush's sentinel tensors (``__delta_spec__`` / ``__positions__`` /
``__values__``) through the existing bucketed ZMQ channel. This module stitches them
back into whole flushes and applies each in place with verl's delta loader
(``verl.workers.rollout.sglang_rollout.delta_loader.apply_delta``), reused unchanged:
same wire contract, same per-flush checksum, same NaN-masked ``load_weights`` apply.

Pure torch; importable without vllm/vllm_omni so CPU tests can drive it.
"""

from __future__ import annotations

import json
import logging
import os

import torch

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SPEC_NAME = "__delta_spec__"
POSITIONS_NAME = "__positions__"
VALUES_NAME = "__values__"


class DeltaFlushReceiver:
    """Accumulate bucketed sentinel tensors into whole delta flushes and apply them.

    The bucketed IPC channel never splits one tensor across buckets, but one flush's
    tensors may land in different buckets and the bucket buffer is reused once the
    callback returns, so the spec and positions are copied out on arrival; values are
    consumed within the callback that receives them.

    Args:
        load_target: Anything with a ``load_weights`` method -- the vllm model on AR
            workers, the vllm-omni pipeline (or the worker itself) on diffusion workers.
    """

    def __init__(self, load_target):
        self.load_target = load_target
        self._spec_tensor: torch.Tensor | None = None
        self._positions: torch.Tensor | None = None
        self.saw_seed = False
        self.applied_flushes = 0

    def on_bucket(self, weights: list, is_last: bool = False) -> None:
        """Consume one bucket of the delta stream; applies every flush it completes."""
        for name, tensor in weights:
            if name == SPEC_NAME:
                if self._spec_tensor is not None:
                    raise RuntimeError("delta stream: a new flush spec arrived before the previous flush's values")
                self._spec_tensor = tensor.clone()
            elif name == POSITIONS_NAME:
                if self._positions is not None:
                    raise RuntimeError("delta stream: duplicate positions blob within one flush")
                self._positions = tensor.clone()
            elif name == VALUES_NAME:
                self._apply(tensor)
            else:
                raise ValueError(f"delta stream: unexpected tensor {name!r}")
        if is_last:
            self.finish()

    def _apply(self, values: torch.Tensor) -> None:
        from verl.workers.rollout.sglang_rollout.delta_loader import apply_delta

        if self._spec_tensor is None:
            raise RuntimeError("delta stream: values arrived before the flush spec")
        spec = json.loads(bytes(self._spec_tensor.cpu().numpy().tobytes()).decode())
        if spec["encoding"] == "dense" and not spec.get("verify"):
            self.saw_seed = True
        named = [(SPEC_NAME, self._spec_tensor), (VALUES_NAME, values)]
        if self._positions is not None:
            named.insert(1, (POSITIONS_NAME, self._positions))
        apply_delta(self.load_target, named)
        self.applied_flushes += 1
        self._spec_tensor = None
        self._positions = None

    def finish(self) -> None:
        """Assert the stream ended on a flush boundary; called on the last bucket."""
        if self._spec_tensor is not None or self._positions is not None:
            raise RuntimeError("delta stream ended with a partial flush")
        logger.info("delta apply done: %d flushes (seed=%s)", self.applied_flushes, self.saw_seed)
