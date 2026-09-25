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

"""Registry of per-agent-loop config binders.

A concrete agent loop sometimes needs its recipe's Hydra knobs filled before the
worker builds anything — which function tools to load, which ``multi_turn.format``
to speak. That knowledge belongs to the pipeline that owns the loop, not to the
generic Omni worker, so the pipeline registers a binder here at import time and the
worker looks it up by the loop's ``default_agent_loop`` name.

This module deliberately imports nothing from ``verl_omni.agent_loop`` or from any
pipeline: it is the leaf that lets a pipeline import the registry without creating
the cycle that an import back into ``omni_agent_loop`` would.
"""

from __future__ import annotations

from typing import Any, Callable

__all__ = ["AGENT_LOOP_CONFIG_BINDERS", "register_agent_loop_config_binder"]

#: ``default_agent_loop`` name -> ``binder(config)``. Populated at import time by the
#: pipeline modules that own a concrete loop.
AGENT_LOOP_CONFIG_BINDERS: dict[str, Callable[[Any], None]] = {}


def register_agent_loop_config_binder(name: str):
    """Register ``fn(config)`` as the config binder for agent loop ``name``.

    Args:
        name: The ``default_agent_loop`` registration name to bind.

    Returns:
        A decorator that registers ``fn`` on the module-level dict and returns it
        unchanged, so the decorated function stays directly callable.
    """

    def decorate(fn: Callable[[Any], None]) -> Callable[[Any], None]:
        AGENT_LOOP_CONFIG_BINDERS[name] = fn
        return fn

    return decorate
