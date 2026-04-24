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

import os

from verl.utils.fs import copy_to_local

__all__ = ["resolve_model_local_dir"]


def resolve_model_local_dir(path: str, use_shm: bool = False) -> str:
    """Resolve ``path`` to an on-disk directory."""
    local_path = copy_to_local(path, use_shm=use_shm)
    local_path_expanded = os.path.expanduser(local_path)
    if not os.path.isdir(local_path_expanded):
        from huggingface_hub import snapshot_download

        if os.path.isabs(local_path_expanded):
            normalized_path = os.path.normpath(local_path_expanded)
            head, repo = os.path.split(normalized_path)
            _, owner = os.path.split(head)
            if owner and repo:
                repo_id = f"{owner}/{repo}"
                return snapshot_download(repo_id, local_dir=local_path_expanded)

        local_path = snapshot_download(path)
    else:
        local_path = local_path_expanded
    return local_path
