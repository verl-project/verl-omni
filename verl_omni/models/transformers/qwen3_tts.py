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
"""Runtime shims for the pinned qwen-tts package's global rope registration.

Applied at qwen3-tts pipeline import — the earliest point the guarded
registration can occur (see verl_omni/pipelines/qwen3_tts/__init__.py).
"""

import functools


class _RopeFallbackRegistry(dict):
    """Wrap registered default-rope initializers so a bare rope_theta read falls
    back to rope_parameters, where Transformers 5.16+ moved the value."""

    def __setitem__(self, key, value):
        if key == "default":
            value = _rope_theta_from_parameters_on_miss(value)
        super().__setitem__(key, value)

    def setdefault(self, key, default=None):
        # dict.setdefault bypasses __setitem__; qwen-tts registers through it.
        if key not in self:
            self[key] = default
        return dict.__getitem__(self, key)


def _rope_theta_from_parameters_on_miss(rope_fn):
    if rope_fn is None or getattr(rope_fn, "_rope_fallback_wrapped", False):
        return rope_fn

    @functools.wraps(rope_fn)
    def wrapper(config, *args, **kwargs):
        if not hasattr(config, "rope_theta"):
            theta = (getattr(config, "rope_parameters", None) or {}).get("rope_theta")
            if theta is not None:
                config.rope_theta = theta
        return rope_fn(config, *args, **kwargs)

    wrapper._rope_fallback_wrapped = True
    return wrapper


def patch_qwen3_tts_default_rope_read() -> None:
    """Intercept default-rope registrations in Transformers' rope registry.

    The pinned qwen-tts installs a process-global default-rope initializer that
    reads the bare rope_theta attribute; Transformers 5.16+ moved that value
    into rope_parameters on several config classes, so an unwrapped read kills
    every default-rope model build in a process that imported qwen_tts.

    TODO (mike): remove once qwen-tts reads rope_parameters on default-rope
    configs upstream (its registered hook reads a bare rope_theta).
    """
    import sys

    from transformers import modeling_rope_utils

    registry = modeling_rope_utils.ROPE_INIT_FUNCTIONS
    if isinstance(registry, _RopeFallbackRegistry):
        return
    wrapped = _RopeFallbackRegistry()
    for key, value in registry.items():
        wrapped[key] = value
    modeling_rope_utils.ROPE_INIT_FUNCTIONS = wrapped
    # Modules that bound the original dict by reference keep reading it;
    # rebind every holder so later registrations wrap too.
    for module in tuple(sys.modules.values()):
        if module is None:
            continue
        for name, value in vars(module).items():
            if value is registry:
                setattr(module, name, wrapped)
