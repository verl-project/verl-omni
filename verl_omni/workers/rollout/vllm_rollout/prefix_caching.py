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
"""Temporary shim: honor an explicit `rollout.enable_prefix_caching=False`.

verl's `build_cli_args_from_config` drops explicit-False booleans, so the
engine's None parser default resolves prefix caching back to ON — which
corrupts rollouts after sleep/wake cycles on vllm-omni (stale block-hash
table over discarded KV). Until upstream ships, wrap the serializer so an
explicit False re-emits `--no-enable-prefix-caching`.

TODO(verl-upstream): drop this shim once verl serializes explicit False on
Optional[bool] engine args as `--no-<flag>` and verl-omni pins that release.
"""

import importlib
from types import ModuleType
from typing import Any

_PATCH_MARK = "_verl_omni_prefix_caching_cli_fix"


def install_prefix_caching_cli_fix(target_module: ModuleType | None = None) -> None:
    """Wrap verl's `build_cli_args_from_config` to honor an explicit False.

    Idempotent; `target_module` defaults to verl's vllm_async_server and is
    overridable for tests.
    """
    if target_module is None:
        target_module = importlib.import_module("verl.workers.rollout.vllm_rollout.vllm_async_server")

    orig = getattr(target_module, "build_cli_args_from_config", None)
    if orig is None:
        raise AttributeError(
            f"{target_module!r} has no 'build_cli_args_from_config'; verl's CLI "
            "serializer moved? Update the prefix-caching shim."
        )
    if getattr(orig, _PATCH_MARK, False):
        return

    def _build_cli_args_with_explicit_false(config: dict[str, Any]) -> list[str]:
        config = dict(config)
        extra: list[str] = []
        if config.get("enable_prefix_caching") is False:
            config.pop("enable_prefix_caching")
            extra.append("--no-enable-prefix-caching")
        return orig(config) + extra

    setattr(_build_cli_args_with_explicit_false, _PATCH_MARK, True)
    target_module.build_cli_args_from_config = _build_cli_args_with_explicit_false
