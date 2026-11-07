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
"""Base class for stateful reward scorers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["RewardScorer"]


class RewardScorer(ABC):
    """Base class for stateful reward scorers — local models, preference models, judge sessions.

    Subclass, implement :meth:`score_batch`, and register with
    ``@register_scorer("name")``. The executor owns batching, threading, device
    placement, and lifecycle; a scorer never manages a queue, consumer task,
    thread, or event loop. For when to write a class versus a plain
    ``compute_score`` function, see ``docs/contributing/integrating_a_reward_scorer.md``.

    Class options (override as class attributes):

    - ``max_batch_size``: requests per :meth:`score_batch` call; ``None`` (the
      default) scores one request per call. Chunking internal to the forward
      stays inside :meth:`score_batch` and composes with this bound.
    - ``supports_cpu_offload``: when ``True``, the executor parks the weights
      on CPU between scoring phases instead of disposing the scorer and
      rebuilding it on the next wake.

    Instance state (assign both in :meth:`load` to inherit the lifecycle
    defaults below; scorers without weights skip both):

    - ``self.model``: the weights module.
    - ``self.device``: the device the weights live on.

    Example::

        @register_scorer("my_scorer")
        class MyScorer(RewardScorer):
            max_batch_size = 8
            supports_cpu_offload = True

            def load(self, device):
                self.model = build_model().to(device)
                self.device = device

            def score_batch(self, batch):
                ...
    """

    max_batch_size: int | None = None
    supports_cpu_offload: bool = False
    model: Any = None
    device: Any = None

    @abstractmethod
    def score_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any] | Exception]:
        """Score one microbatch.

        Args:
            batch: one dict per sample, carrying the per-sample keys a
                ``compute_score`` function receives (``data_source``,
                ``solution_image``, ``ground_truth``, ``extra_info``) plus any
                configured ``executor.kwargs``.

        Returns:
            one entry per input, in input order: a score dict
            (``{"score": float, ...metrics}``) on success, or an ``Exception``
            in that sample's slot when only its preprocessing failed.

        Raises:
            Exception: the forward failed; every sample in this microbatch is
                failed with this error.
        """

    def load(self, device: Any) -> None:  # noqa: B027 — intentional optional hook
        """Build weights and place them on ``device``; no-op when the scorer holds no weights."""

    def wake_up(self) -> None:
        """Move ``self.model`` back to ``self.device`` after :meth:`sleep`."""
        self._require_weights().to(self.device)

    def sleep(self) -> None:
        """Park ``self.model`` on CPU."""
        self._require_weights().to("cpu")

    def close(self) -> None:
        """Drop the ``self.model`` reference; gc and accelerator-cache clearing stay with the executor."""
        self.model = None

    def _require_weights(self) -> Any:
        if self.model is None or self.device is None:
            raise RuntimeError(
                f"{type(self).__name__} declares supports_cpu_offload but has no weights; "
                "assign self.model and self.device in load()"
            )
        return self.model
