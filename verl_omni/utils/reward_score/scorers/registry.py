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

Selection follows this order:

1. a name registered via :func:`register_scorer`;
2. an import spec ``pkg.module:Attr`` (a ``pkg://`` prefix is tolerated);
3. a file path ``path/to/module.py:Attr``.

Every step missed fails with a ``ValueError`` listing the registered names —
the same fail-fast posture as ``DiffusionModelBase.get_class_by_name``.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from verl_omni.utils.reward_score.scorers.base import RewardScorer

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "get_scorer_cls",
    "import_from_spec",
    "load_reward_object",
    "peek_registered_scorers",
    "register_scorer",
]

_REGISTERED: dict[str, type[RewardScorer]] = {}
_FILE_MODULES: dict[str, ModuleType] = {}
_FILE_IMPORT_LOCK = threading.RLock()


def register_scorer(name: str) -> Callable[[type[RewardScorer]], type[RewardScorer]]:
    """Class decorator registering a stateful scorer class under ``name``.

    Registering the identical class again is a tolerated no-op, keeping the
    decorator idempotent when a module is re-imported.

    Args:
        name: the config-facing selection name; must be a non-empty string
            without ``:`` (which would make it an import spec).

    Raises:
        ValueError: the name is malformed or registered to a different class,
            the class is not a :class:`RewardScorer`, or its declarations are
            invalid.
    """

    def decorator(cls: type[RewardScorer]) -> type[RewardScorer]:
        if not isinstance(name, str) or not name or ":" in name:
            raise ValueError(f"Scorer name must be a non-empty string without ':', got {name!r}")
        if name in _REGISTERED and _REGISTERED[name] is not cls:
            registered = _REGISTERED[name]
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
        _REGISTERED[name] = cls
        return cls

    return decorator


def peek_registered_scorers() -> dict[str, type[RewardScorer]]:
    """Return a snapshot of registered scorer classes keyed by name."""
    return dict(_REGISTERED)


def get_scorer_cls(ref: str) -> type[RewardScorer]:
    """Resolve a scorer class by registered name or import spec.

    Args:
        ref: a registered name, ``pkg.module:Attr``, or ``path/to/file.py:Attr``.

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
    if not (isinstance(cls, type) and issubclass(cls, RewardScorer)):
        raise ValueError(f"Scorer reference {ref!r} resolved to {cls!r}, which does not subclass RewardScorer")
    return cls


def import_from_spec(ref: str) -> Any:
    """Resolve a ``module:Attr`` reference to the object.

    The module part may be a dotted module path (``pkg.module``, optionally
    ``pkg://``-prefixed) or a file path (``path/to/module.py``, optionally
    ``file://``-prefixed); a bare dotted module is ``pkg://``-prefixed before
    delegating to verl's :func:`load_extern_object`.
    """
    module_path, object_name = ref.rsplit(":", 1)
    if not module_path.startswith("pkg://") and "/" not in module_path and not module_path.endswith(".py"):
        module_path = f"pkg://{module_path}"
    return load_reward_object(module_path, object_name)


def load_reward_object(module_path: str, object_name: str) -> Any:
    """Share a file module between reward functions and registered scorer classes.

    File imports execute once per canonical path in this worker process, just
    like package imports. Importable files retain their package context.
    In particular, resolving two attributes must not rerun
    class-registration decorators or recreate module-owned model state.
    """
    from verl.utils.import_utils import load_extern_object, load_module

    if module_path.startswith("pkg://"):
        return load_extern_object(module_path=module_path, object_name=object_name)
    path = Path(module_path.removeprefix("file://")).resolve()
    key = str(path)
    with _FILE_IMPORT_LOCK:
        module = _FILE_MODULES.get(key)
    if module is None:
        # Package imports keep their own context and import lock. Never wait for
        # another thread's package initialization while holding our file lock.
        module = _source_module(path)
        spec = getattr(module, "__spec__", None)
        if getattr(spec, "_initializing", False):
            return getattr(importlib.import_module(spec.name), object_name)
        with _FILE_IMPORT_LOCK:
            if key not in _FILE_MODULES:
                if module is None:
                    module_name = "_verl_omni_reward_" + hashlib.sha256(key.encode()).hexdigest()
                    module = load_module(key, module_name=module_name)
                _FILE_MODULES[key] = module
            module = _FILE_MODULES[key]
    return getattr(module, object_name)


def _source_module(path: Path) -> ModuleType | None:
    """Reuse a loaded source or import it under its normal package name."""
    sources = [
        (module, vars(module).get("__file__"))
        for module in sys.modules.copy().values()
        if isinstance(module, ModuleType)
    ]
    # The common package-loaded path needs no filesystem walk over other modules.
    for module, source in sources:
        if source == str(path):
            return module
    for module, source in sources:
        if isinstance(source, str) and source:
            try:
                if Path(source).resolve() == path:
                    return module
            except (OSError, RuntimeError):
                continue
    for entry in sys.path:
        if not isinstance(entry, str):
            continue
        try:
            relative = path.relative_to(Path(entry or ".").resolve())
        except (ValueError, OSError, RuntimeError):
            continue
        if relative.suffix != ".py":
            continue
        parts = relative.with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts or not all(part.isidentifier() for part in parts):
            continue
        search = None
        spec = None
        # Resolve filesystem locations before importing: an unrelated package
        # with the same name must not execute during an explicit file lookup.
        for length in range(1, len(parts) + 1):
            name = ".".join(parts[:length])
            loaded = sys.modules.get(name)
            if isinstance(loaded, ModuleType):
                spec = vars(loaded).get("__spec__")
                search = vars(loaded).get("__path__")
            elif loaded is not None:
                break
            else:
                spec = importlib.machinery.PathFinder.find_spec(name, search)
                search = spec.submodule_search_locations if spec is not None else None
            if length < len(parts) and search is None:
                break
        else:
            if spec is not None and spec.origin and Path(spec.origin).resolve() == path:
                return importlib.import_module(name)
    return None


def _load_ref(ref: str) -> Any:
    try:
        return import_from_spec(ref)
    except (ImportError, AttributeError, FileNotFoundError) as error:
        raise _unknown_scorer_error(ref, peek_registered_scorers()) from error


def _unknown_scorer_error(ref: str, registered: dict[str, type]) -> ValueError:
    names = ", ".join(sorted(registered)) or "(none)"
    return ValueError(
        f"Unknown reward scorer {ref!r}. Registered names: {names}. "
        'Out-of-tree scorers resolve via "pkg.module:Attr" or "path/to/file.py:Attr" import specs.'
    )
