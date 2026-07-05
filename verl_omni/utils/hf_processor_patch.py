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

"""Auto-patch Hugging Face processor configs that ship without ``model_type``.

Some multimodal models (e.g. Qwen-Image-Edit-2511, tiny-random/qwen-image-edit-plus)
store their processor in a ``processor/`` subdirectory that lacks a ``config.json``
with ``model_type``, causing ``transformers.AutoConfig`` (and therefore
``verl.utils.hf_processor``) to fail. :func:`install_auto_patch` wraps
``verl.utils.tokenizer.hf_processor`` so that the missing config is written
idempotently before the original loader runs. The patch is installed
automatically when ``verl_omni`` is imported.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Default model_type written into the patched processor config. All known
# Qwen-Image-Edit variants use the Qwen2-VL processor family.
_DEFAULT_PROCESSOR_MODEL_TYPE = "qwen2_vl"

_installed = False


def _resolve_snapshot_dir(model_id: str) -> str | None:
    """Resolve the local snapshot directory for *model_id* from the HF cache."""
    # Local path: use directly.
    if os.path.isdir(model_id):
        return model_id

    # HF model id: resolve from cache.
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_id, local_files_only=True)
    except Exception:
        cache = os.path.expanduser("~/.cache/huggingface/hub")
        slug = "models--" + model_id.replace("/", "--")
        refs = Path(cache) / slug / "refs" / "main"
        if refs.exists():
            sha = refs.read_text().strip()
            return str(Path(cache) / slug / "snapshots" / sha)
        return None


def _ensure_processor_config(model_id: str, model_type: str = _DEFAULT_PROCESSOR_MODEL_TYPE) -> None:
    """Write a minimal ``config.json`` into the processor dir if missing."""
    local = _resolve_snapshot_dir(model_id)
    if local is None:
        return

    proc_dir = Path(local) / "processor"
    cfg_file = proc_dir / "config.json"
    if not proc_dir.exists() or cfg_file.exists():
        return

    try:
        cfg_file.write_text(json.dumps({"model_type": model_type}))
        logger.info("Patched processor config: %s", cfg_file)
    except Exception as e:
        # Read-only cache (shared cluster / container) — warn but do not crash.
        logger.warning(
            "Failed to patch processor config for %s: %s. "
            "If the cache is read-only, patch manually or ignore if not needed.",
            model_id,
            e,
        )


def install_auto_patch() -> None:
    """Wrap ``verl.utils.tokenizer.hf_processor`` to auto-patch missing configs.

    Before delegating to the original ``hf_processor``, resolve the model path
    and ensure its ``processor/config.json`` exists. Idempotent: safe to call
    multiple times. Installed automatically when ``verl_omni`` is imported.
    """
    global _installed
    if _installed:
        return

    try:
        import verl.utils.tokenizer as _vt
    except ImportError:
        return

    _original = _vt.hf_processor

    def _patched_hf_processor(name_or_path, **kwargs):
        _ensure_processor_config(str(name_or_path))
        return _original(name_or_path, **kwargs)

    _vt.hf_processor = _patched_hf_processor
    # Refresh re-exports in modules that did `from verl.utils import hf_processor`.
    import sys as _sys

    for _mod_name in ("verl.utils", "verl.workers.config.model"):
        _mod = _sys.modules.get(_mod_name)
        if _mod is not None and hasattr(_mod, "hf_processor"):
            _mod.hf_processor = _patched_hf_processor

    _installed = True
