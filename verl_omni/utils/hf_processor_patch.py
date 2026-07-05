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

"""Patch Hugging Face processor configs that ship without ``model_type``.

Some multimodal models (e.g. Qwen-Image-Edit-2511) store their processor in a
``processor/`` subdirectory that lacks a ``config.json`` with ``model_type``,
causing ``transformers.AutoConfig`` (and therefore ``verl.utils.hf_processor``)
to fail. :func:`ensure_processor_config` writes a minimal ``config.json`` into
the processor directory so AutoConfig can resolve the model type. The patch is
idempotent and does not affect model weights.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Maps model_id -> model_type used for the processor config patch.
_KNOWN_PROCESSOR_MODEL_TYPES: dict[str, str] = {
    "Qwen/Qwen-Image-Edit-2511": "qwen2_vl",
}


def _resolve_snapshot_dir(model_id: str) -> str | None:
    """Resolve the local snapshot directory for *model_id* from the HF cache."""
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


def ensure_processor_config(model_id: str, model_type: str | None = None) -> None:
    """Ensure the processor directory of *model_id* has a usable ``config.json``.

    Args:
        model_id: Hugging Face model id (e.g. ``"Qwen/Qwen-Image-Edit-2511"``).
        model_type: Override ``model_type`` written into the processor config.
            If ``None``, looks up :data:`_KNOWN_PROCESSOR_MODEL_TYPES`.
    """
    if model_type is None:
        model_type = _KNOWN_PROCESSOR_MODEL_TYPES.get(model_id)
    if model_type is None:
        logger.debug("No known processor model_type for %s; skipping patch", model_id)
        return

    local = _resolve_snapshot_dir(model_id)
    if local is None:
        logger.warning("Could not locate %s in HF cache; skipping processor patch", model_id)
        return

    proc_dir = Path(local) / "processor"
    cfg_file = proc_dir / "config.json"
    if not proc_dir.exists() or cfg_file.exists():
        logger.debug("Processor config already present or processor dir missing: %s", cfg_file)
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
