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
"""Compatibility shims for Hugging Face remote-code models on transformers >= 5."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_PATCHED_ATTR = "_verl_omni_post_init_patched"


def _needs_transformers5_compat() -> bool:
    try:
        import transformers

        return int(transformers.__version__.split(".", 1)[0]) >= 5
    except Exception:
        return False


def patch_remote_auto_model_init(
    model_path: str,
    *,
    trust_remote_code: bool,
    config: Any = None,
    auto_class_name: str = "AutoModel",
) -> None:
    """Wrap a remote auto-model class so ``post_init()`` runs when missing.

    Transformers >= 5 expects every ``PreTrainedModel`` to call ``post_init()`` at
    the end of ``__init__``, which sets ``all_tied_weights_keys``.  Several remote-code
    models (including MiniCPM-o) only declare ``_tied_weights_keys`` and omit
    ``post_init()``, which makes ``from_pretrained`` fail during weight loading.
    """
    if not _needs_transformers5_compat() or not trust_remote_code:
        return

    from transformers import AutoConfig
    from transformers.models.auto.auto_factory import get_class_from_dynamic_module

    resolved_config = config
    if resolved_config is None:
        resolved_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    auto_map = getattr(resolved_config, "auto_map", None)
    if not auto_map or auto_class_name not in auto_map:
        return

    model_cls = get_class_from_dynamic_module(
        auto_map[auto_class_name],
        model_path,
        trust_remote_code=trust_remote_code,
    )
    wrap_model_init_with_post_init(model_cls)


def wrap_model_init_with_post_init(model_cls: type) -> None:
    """Ensure ``model_cls.__init__`` ends with ``post_init()`` when needed."""
    if getattr(model_cls, _PATCHED_ATTR, False):
        return

    original_init = model_cls.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, "all_tied_weights_keys") and hasattr(self, "post_init"):
            self.post_init()

    model_cls.__init__ = patched_init
    setattr(model_cls, _PATCHED_ATTR, True)
    logger.debug(
        "Patched %s.__init__ to call post_init() for transformers >= 5 compatibility.",
        model_cls.__name__,
    )
