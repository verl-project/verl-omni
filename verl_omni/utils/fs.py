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

import hashlib
import os
from pathlib import Path

from verl.utils.fs import copy_to_local

__all__ = ["resolve_model_local_dir", "diffusion_model_provenance"]


def diffusion_model_provenance(local_path: str) -> dict:
    """Record a resolved snapshot's revision when available and its transformer config hash."""
    root = Path(local_path)
    revision = root.name if root.parent.name == "snapshots" else None
    metadata = root / ".cache/huggingface/download/model_index.json.metadata"
    if metadata.is_file():
        with metadata.open() as file:
            revision = file.readline().strip()
    if not revision or len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        revision = None
    with (root / "transformer/config.json").open("rb") as file:
        config_hash = hashlib.file_digest(file, "sha256").hexdigest()
    return {"base_model_revision": revision, "base_transformer_config_sha256": config_hash}


def resolve_model_local_dir(path: str, use_shm: bool = False) -> str:
    """Resolve ``path`` to an on-disk directory."""
    local_path = copy_to_local(path, use_shm=use_shm)
    if not os.path.isdir(local_path):
        from huggingface_hub import snapshot_download

        local_path = snapshot_download(path)
    return local_path
