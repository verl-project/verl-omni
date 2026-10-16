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
"""Registry and selection for stateful (class-based) reward scorers.

Selection follows exactly this order (RFC §5.3):

1. a name registered via :func:`register_scorer`;
2. an import spec ``pkg.module:Attr`` (a ``pkg://`` prefix is tolerated);
3. today's file-path form ``path/to/module.py:Attr``.

Every step missed fails with a ``ValueError`` listing the registered names —
the same fail-fast posture as ``DiffusionModelBase.get_class_by_name``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from verl_omni.utils.reward_score.scorers.base import RewardScorer

__all__ = [
    "get_scorer_cls",
    "import_from_spec",
    "peek_registered_scorers",
    "register_scorer",
]


@dataclass(frozen=True)
class _ScorerRegistration:
    cls: type[RewardScorer]
    modality: str | None


_REGISTERED: dict[str, _ScorerRegistration] = {}


def register_scorer(name: str, *, modality: str | None = None) -> Callable[[type[RewardScorer]], type[RewardScorer]]:
    """Class decorator registering a stateful scorer class under ``name``.

    Args:
        name: the config-facing selection name; must be a non-empty string
            without ``:`` (which would make it an import spec).
        modality: optional tag (e.g. ``image``) reserved for docs/validation.

    Raises:
        ValueError: the name is malformed or registered to a different class,
            the class is not a :class:`RewardScorer`, or its declarations are
            invalid. Re-registering the identical class is a tolerated no-op
            (the rule of upstream verl's reward-manager registry).
    """

    def decorator(cls: type[RewardScorer]) -> type[RewardScorer]:
        from verl_omni.utils.reward_score.scorers.base import RewardScorer

        if not isinstance(name, str) or not name or ":" in name:
            raise ValueError(f"Scorer name must be a non-empty string without ':', got {name!r}")
        if name in _REGISTERED and _REGISTERED[name].cls is not cls:
            registered = _REGISTERED[name].cls
            raise ValueError(
                f"Reward scorer {name!r} has already been registered: "
                f"{registered.__module__}.{registered.__qualname__} vs "
                f"{cls.__module__}.{cls.__qualname__}"
            )
        if name in _REGISTERED:
            return cls
        if not (isinstance(cls, type) and issubclass(cls, RewardScorer)):
            raise ValueError(f"register_scorer({name!r}) requires a RewardScorer subclass, got {cls!r}")
        if cls.max_batch_size is not None and (
            isinstance(cls.max_batch_size, bool) or not isinstance(cls.max_batch_size, int) or cls.max_batch_size <= 0
        ):
            raise ValueError(f"Reward scorer {name!r} max_batch_size must be a positive integer or None")
        if type(cls.supports_cpu_offload) is not bool:
            raise ValueError(f"Reward scorer {name!r} supports_cpu_offload must be a bool")
        _REGISTERED[name] = _ScorerRegistration(cls=cls, modality=modality)
        return cls

    return decorator


def peek_registered_scorers() -> dict[str, type[RewardScorer]]:
    """Return a snapshot of registered scorer classes keyed by name."""
    return {name: registration.cls for name, registration in _REGISTERED.items()}


def get_scorer_cls(ref: str) -> type[RewardScorer]:
    """Resolve a scorer class by registered name or import spec.

    Args:
        ref: a registered name, ``pkg.module:Attr``, or ``path/file.py:Attr``.

    Raises:
        ValueError: nothing resolvable matched ``ref``, or the resolved class
            is not a :class:`RewardScorer` subclass.
    """
    registered = peek_registered_scorers()
    if ref in registered:
        return registered[ref]
    if not isinstance(ref, str) or ":" not in ref:
        raise _unknown_scorer_error(ref, registered)
    cls = _load_ref(ref)
    from verl_omni.utils.reward_score.scorers.base import RewardScorer

    if not (isinstance(cls, type) and issubclass(cls, RewardScorer)):
        raise ValueError(f"Scorer reference {ref!r} resolved to {cls!r}, which does not subclass RewardScorer")
    return cls


def import_from_spec(ref: str) -> Any:
    """Resolve a ``module:Attr`` reference to the object.

    Accepts a dotted module (a ``pkg://`` prefix is tolerated — upstream's
    :func:`load_extern_object` handles prefixed, file, and ``file://`` forms);
    a bare dotted module is prefixed before delegating.
    """
    from verl.utils.import_utils import load_extern_object

    module_path, object_name = ref.rsplit(":", 1)
    if not module_path.startswith("pkg://") and "/" not in module_path and not module_path.endswith(".py"):
        module_path = f"pkg://{module_path}"
    return load_extern_object(module_path=module_path, object_name=object_name)


def _load_ref(ref: str) -> Any:
    try:
        return import_from_spec(ref)
    except (ImportError, AttributeError, FileNotFoundError, ModuleNotFoundError) as error:
        raise _unknown_scorer_error(ref, peek_registered_scorers()) from error


def _unknown_scorer_error(ref: str, registered: dict[str, type]) -> ValueError:
    names = ", ".join(sorted(registered)) or "(none)"
    return ValueError(
        f"Unknown reward scorer {ref!r}. Registered names: {names}. "
        'Out-of-tree scorers resolve via "pkg.module:Attr" or "path/to/file.py:Attr" import specs.'
    )
