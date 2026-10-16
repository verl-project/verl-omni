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
"""Protocol for stateful reward scorers (the layered scorer interface).

Free-function scorers — the ``compute_score`` contract in
``.agents/rules/reward.md`` — are unchanged and remain the recommended shape
for rule functions and HTTP clients. A scorer becomes a class only when it
holds state across requests (a local model, a preference model, a judge
session).

A class-based scorer writes **scoring semantics only**. Batching, threading,
placement, and lifecycle choreography are framework-owned: the executor-side
microbatcher groups requests and calls :meth:`RewardScorer.score_batch`, and
only the executor core ever invokes the lifecycle hooks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["RewardScorer"]


class RewardScorer(ABC):
    """Stateful scorer protocol: write scoring semantics, nothing else.

    The framework does the rest: the executor's microbatcher groups requests,
    runs :meth:`score_batch` (sync in a worker thread, async awaited), and is
    the only caller of the lifecycle hooks.

    Declarations are plain data readable from the class:

    - ``max_batch_size``: requests packed per ``score_batch`` call; ``None``
      means unbatched (one request per call). Chunking internal to the forward
      (e.g. frame budgets) stays inside ``score_batch`` and composes
      with this request-level bound.
    - ``supports_cpu_offload``: when true, the executor parks the weights on
      the host between scoring phases instead of disposing the scorer and
      rebuilding it on the next wake.

    ``score_batch`` contract:

    - One entry per request, in input order: ``out[i]`` is the result of
      ``batch[i]``, whatever internal padding or reordering the forward did.
    - An ``Exception`` in a request's slot means only that request's
      preprocessing failed; a raised exception means the whole microbatch
      failed — the executor fails every request in the batch with it.

    A scorer never manages a queue, consumer task, thread, or event loop, and
    never calls another scorer's or the executor's lifecycle methods. The
    constructor takes configuration only (paths, dtypes, knobs); weights are
    untouched before :meth:`load`, where device placement happens.

    Lifecycle defaults: assigning ``self.model`` (the weights module) and
    ``self.device`` in :meth:`load` inherits the ``wake_up``/``sleep``/
    ``close`` implementations; a hook is overridden only for extra
    bookkeeping (e.g. syncing a cached device flag). gc and accelerator-cache
    clearing stay with the executor.
    """

    max_batch_size: int | None = None
    supports_cpu_offload: bool = False
    # Assign both in load() to inherit the default lifecycle hooks below.
    model: Any = None
    device: Any = None

    @abstractmethod
    def score_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any] | BaseException]:
        """Score one microbatch of per-sample kwarg dicts.

        Args:
            batch: one kwarg dict per sample, matching the free-function ``compute_score``
                keyword contract (``data_source``, ``solution_image``,
                ``ground_truth``, ``extra_info``, plus configured kwargs).

        Returns:
            one entry per input, in input order: a score dict (``{"score": float,
                ...metrics}``) on success, or an ``Exception`` instance when only
            that request's preprocessing failed.

        Raises:
            Exception: the forward failed — the executor fails every request in
                this microbatch with the raised error.
        """

    def load(self, device: Any) -> None:  # noqa: B027 — optional hook: scorers without weights keep the no-op
        """Construct weights and place them on ``device`` (no-op default)."""

    def wake_up(self) -> None:
        """Default: restore :attr:`model` to :attr:`device` after :meth:`sleep`."""
        self._require_weights().to(self.device)

    def sleep(self) -> None:
        """Default: park :attr:`model` on the host CPU."""
        self._require_weights().to("cpu")

    def close(self) -> None:
        """Default: drop the :attr:`model` reference.

        The executor owns ``gc`` and accelerator-cache clearing after
        sleep/close; a scorer's hook only releases what the scorer itself holds.
        """
        self.model = None

    def _require_weights(self) -> Any:
        if self.model is None:
            raise RuntimeError(
                f"{type(self).__name__} declares supports_cpu_offload but has no model; "
                "assign self.model and self.device in load()"
            )
        return self.model
