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
"""Asyncio primitives shared across the reward loop."""

from __future__ import annotations

import asyncio
from typing import Any

__all__ = [
    "await_owned",
]


async def await_owned(awaitable: Any) -> Any:
    """Await ``awaitable`` to settlement, absorbing repeated cancellation of the awaiting task.

    Use when the operation must run to completion once started, even if the task
    awaiting it is cancelled: cancellation is absorbed until settlement, the
    operation's outcome is then consumed, and ``CancelledError`` is re-raised.
    """
    task = asyncio.ensure_future(awaitable)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling():
                cancelled = True
        except BaseException:
            if not task.done():
                raise
    if cancelled:
        try:
            task.result()
        except BaseException:
            pass
        raise asyncio.CancelledError
    return task.result()
