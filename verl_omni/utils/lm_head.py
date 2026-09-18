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
"""Load a teacher's unembedding matrix for hidden-state (nitrobrew) OPD.

The Nitrobrew loss reconstructs teacher logits on the actor as
``z_t = h @ W.T``, where ``h`` is the teacher's last-layer hidden states and
``W`` is the teacher unembedding.  ``W`` is read once from safetensors on the
driver (no full HF model) and pushed to actor ranks via
``set_teacher_unembeds``.

Qwen3-Omni keys are verified against the checkpoint layout (thinker is a
Qwen3-MoE, so the thinker text head is ``thinker.lm_head.weight`` and the
thinker text embeddings live under ``thinker.model.embed_tokens.weight``).
"""

import json
import logging
import os
from typing import Iterator

import torch
from verl.utils.fs import copy_to_local

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Ordered candidate keys, most specific first. `tie_word_embeddings` set => the
# head row is the embedding matrix; otherwise the separate lm_head.
DEFAULT_LM_HEAD_KEYS = (
    "thinker.lm_head.weight",
    "lm_head.weight",
    "thinker.model.embed_tokens.weight",
    "model.embed_tokens.weight",
    "embed_tokens.weight",
)


def _iter_lm_head_candidates(folder: str, keys: tuple[str, ...]) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (key, tensor) for each candidate key present in the checkpoint.

    Handles sharded safetensors (``model.safetensors.index.json``) and
    single-file safetensors.
    """
    from safetensors.torch import load_file

    index_path = os.path.join(folder, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        present = [k for k in keys if k in weight_map]
        # Dedupe shard reads: group present keys by their shard file.
        shard_to_keys: dict[str, list[str]] = {}
        for k in present:
            shard_to_keys.setdefault(weight_map[k], []).append(k)
        for shard, shard_keys in shard_to_keys.items():
            tensors = load_file(os.path.join(folder, shard))
            for k in shard_keys:
                yield k, tensors[k]
    else:
        tensors = load_file(os.path.join(folder, "model.safetensors"))
        for k in keys:
            if k in tensors:
                yield k, tensors[k]


def load_lm_head_weight(
    model_path: str,
    keys: tuple[str, ...] = DEFAULT_LM_HEAD_KEYS,
) -> torch.Tensor:
    """Read the teacher's lm_head (or tied embed) weight.

    Returns a CPU fp32 tensor of shape ``[V, D]``.  ``model_path`` may be a
    local directory or an HF Hub id; ``copy_to_local`` resolves FS/HDFS paths.
    """
    from huggingface_hub import snapshot_download

    resolved = copy_to_local(model_path)
    folder = (
        resolved
        if os.path.isdir(resolved)
        else snapshot_download(resolved, allow_patterns=["*.safetensors*", "*.json"])
    )

    for key, tensor in _iter_lm_head_candidates(folder, keys):
        logger.info("loaded teacher unembedding from %s (%s)", key, tuple(tensor.shape))
        return tensor.float()

    raise ValueError(f"Could not find lm_head/embed_tokens weight under {model_path}. Searched keys: {list(keys)}")
