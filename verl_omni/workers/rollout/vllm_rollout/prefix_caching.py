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
"""Honor an explicit ``rollout.enable_prefix_caching=False`` on the omni server.

verl's ``build_cli_args_from_config`` serializes booleans as ``if v:
append("--{k}")``, so an explicit ``False`` emits no flag at all. vLLM's parser
default for ``enable_prefix_caching`` is ``None``, which resolves to ``True``
for generative decoders — the engine silently runs with prefix caching ON.

With vllm-omni's Orchestrator this is catastrophic in colocated RL training:
sleep/wake is dispatched worker-direct (``collective_rpc("handle_sleep_task")``
→ ``CuMemAllocator``), so the scheduler's CPU-side block-hash table survives
while the KV pool is discarded. The next step's rollouts then attach the
previous step's cached prefix blocks — whose KV is now zeroed — for every
shared, text-only prompt prefix, silently corrupting generation quality
(observed as an AVQA step-2 reward collapse: content-correct but
format-broken answers). Even without sleep mode, prefix-cache reuse
across weight updates is invalid for RL rollouts, so honoring the configured
``False`` is the semantically correct behavior.

This module intentionally depends only on the stdlib so CPU tests can import
it directly; the target module (verl's ``vllm_async_server``) is imported
lazily inside the installer.

TODO(verl-upstream): fix ``build_cli_args_from_config`` to emit ``--no-<flag>``
for explicit-``False`` booleans whose CLI action is a ``BooleanOptionalAction``
(``bool | None`` fields). Once that ships, this shim can be removed. It must
stay scoped to ``enable_prefix_caching``: plain-``bool`` vLLM flags are
``store_true`` and have no ``--no-`` form, so a generic re-emission would
crash the parser.

TODO(vllm-omni-upstream): route Orchestrator sleep/wake through
``EngineCore.sleep()/wake_up()`` (which reset scheduler caches before
discarding KV, as golden vLLM does) and implement the currently no-op
``AsyncOmni.reset_prefix_cache``. After that lands, a stale hash table over
discarded KV can no longer form even when prefix caching is enabled.
"""

import importlib
import logging
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

_PREFIX_CACHING_KEY = "enable_prefix_caching"
_PREFIX_CACHING_NO_FLAG = "--no-enable-prefix-caching"
_PATCH_MARK = "_verl_omni_prefix_caching_cli_fix"
_DEFAULT_TARGET_MODULE = "verl.workers.rollout.vllm_rollout.vllm_async_server"


def install_prefix_caching_cli_fix(target_module: ModuleType | None = None) -> None:
    """Patch verl's CLI serializer to honor an explicit prefix-caching ``False``.

    Wraps ``build_cli_args_from_config`` on verl's ``vllm_async_server`` module
    so that ``{"enable_prefix_caching": False}`` re-emits
    ``--no-enable-prefix-caching`` (accepted by vLLM's ``BooleanOptionalAction``
    parser) instead of being silently dropped. Idempotent; no-op if verl's
    serializer already honors explicit-False values.

    Args:
        target_module: Module exposing ``build_cli_args_from_config``. Defaults
            to verl's ``vllm_async_server``; overridable for tests.
    """
    if target_module is None:
        target_module = importlib.import_module(_DEFAULT_TARGET_MODULE)

    orig = getattr(target_module, "build_cli_args_from_config", None)
    if orig is None:
        raise AttributeError(
            f"{target_module!r} has no 'build_cli_args_from_config'; verl's CLI "
            "serializer moved? The prefix-caching fix in verl_omni needs updating."
        )
    if getattr(orig, _PATCH_MARK, False):
        return  # Already patched (ours) or already explicit-False-aware.

    def _build_cli_args_with_explicit_false(config: dict[str, Any]) -> list[str]:
        config = dict(config)
        extra: list[str] = []
        if config.get(_PREFIX_CACHING_KEY) is False:
            config.pop(_PREFIX_CACHING_KEY)
            extra.append(_PREFIX_CACHING_NO_FLAG)
        return orig(config) + extra

    setattr(_build_cli_args_with_explicit_false, _PATCH_MARK, True)
    target_module.build_cli_args_from_config = _build_cli_args_with_explicit_false


def warn_if_prefix_caching_mismatch(engine: Any, config: Any) -> None:
    """Log a warning when the engine's effective prefix caching != configured.

    Permanent tripwire for the serializer bug above: a mismatch means the
    configured value was dropped somewhere between trainer config and engine
    construction, and the scheduler will run a block-hash cache that survives
    sleep/wake. Best-effort only — never raises.
    """
    configured = getattr(config, _PREFIX_CACHING_KEY, None)
    if not isinstance(configured, bool):
        return
    try:
        effective = engine.vllm_config.cache_config.enable_prefix_caching
    except AttributeError:
        return
    if effective != configured:
        logger.warning(
            "Engine-effective enable_prefix_caching=%s differs from the rollout "
            "config value=%s. An explicit False may have been dropped on the way "
            "to the engine; stale prefix-cache blocks can corrupt rollouts after "
            "sleep/wake cycles.",
            effective,
            configured,
        )
