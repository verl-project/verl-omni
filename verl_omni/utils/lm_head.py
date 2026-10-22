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

The nitrobrew loss reconstructs teacher logits on the actor as
``z_t = h @ W.T`` (h = teacher last-layer hidden states, W = teacher
unembedding), read once from safetensors on the driver and pushed to actor
ranks via ``set_teacher_unembeds``.

Key selection is driven by the model's own config: ``AutoConfig``'s
``get_text_config()`` resolves the text component's vocab_size, hidden_size
and tie_word_embeddings (nesting-aware, so multimodal thinkers work); the
matching suffix group's candidates (tied -> embed_tokens, else lm_head) are
then filtered by an exact (vocab_size, hidden_size) shape match — the
component that produced the hidden states is the only one whose unembedding
matches both dims.
``keys`` overrides the inference for exotic checkpoints.
"""

import glob
import json
import logging
import os
from typing import Iterator

import torch
from verl.utils.fs import copy_to_local

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _discover_weight_map(folder: str) -> dict[str, str]:
    """Map every checkpoint key to its shard file, reading metadata only."""
    from safetensors import safe_open

    for index_path in sorted(glob.glob(os.path.join(folder, "*.safetensors.index.json"))):
        with open(index_path) as f:
            return json.load(f)["weight_map"]

    weight_map: dict[str, str] = {}
    for shard in sorted(glob.glob(os.path.join(folder, "*.safetensors"))):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = os.path.basename(shard)
    return weight_map


def _filter_head_keys(weight_map: dict[str, str], config: dict) -> list[str]:
    """Candidate keys for the text unembedding: tied -> embed_tokens, else lm_head."""
    suffix = "embed_tokens.weight" if config.get("tie_word_embeddings") else "lm_head.weight"
    return [k for k in weight_map if k.endswith(suffix)]


def _iter_lm_head_candidates(
    folder: str,
    weight_map: dict[str, str],
    keys: tuple[str, ...],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield (key, tensor) for each candidate key present in the checkpoint.

    Reads only the candidate tensors via ``safe_open`` — ``load_file`` would
    materialize the whole shard in CPU memory.
    """
    from safetensors import safe_open

    for k in keys:
        shard = weight_map.get(k)
        if shard is None:
            continue
        with safe_open(os.path.join(folder, shard), framework="pt", device="cpu") as f:
            yield k, f.get_tensor(k)


def load_lm_head_weight(
    model_path: str,
    keys: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Read the teacher's lm_head (or tied embed) weight.

    Returns a CPU fp32 tensor of shape ``[V, D]``. ``model_path`` may be a
    local directory or an HF Hub id; ``copy_to_local`` resolves FS/HDFS paths.

    Explicit ``keys`` are tried in order (first hit wins). Otherwise the key
    is resolved from the model's own config via ``AutoConfig``; a candidate
    that does not exactly match the text config's (vocab_size, hidden_size)
    raises instead of being silently accepted.
    """
    from huggingface_hub import snapshot_download

    resolved = copy_to_local(model_path)
    folder = (
        resolved
        if os.path.isdir(resolved)
        else snapshot_download(resolved, allow_patterns=["*.safetensors*", "*.json"])
    )

    weight_map = _discover_weight_map(folder)
    if keys is not None:
        for key, tensor in _iter_lm_head_candidates(folder, weight_map, keys):
            logger.info("loaded teacher unembedding from %s (%s)", key, tuple(tensor.shape))
            return tensor.float()
        raise ValueError(f"Could not find lm_head/embed_tokens weight under {model_path}. Searched keys: {list(keys)}")

    from transformers import AutoConfig

    text_cfg = AutoConfig.from_pretrained(folder).get_text_config()
    keys = tuple(
        _filter_head_keys(weight_map, {"tie_word_embeddings": getattr(text_cfg, "tie_word_embeddings", False)})
    )
    target_shape = (text_cfg.vocab_size, text_cfg.hidden_size)

    attempted: list[str] = []
    for key, tensor in _iter_lm_head_candidates(folder, weight_map, keys):
        attempted.append(f"{key}{tuple(tensor.shape)}")
        if tuple(tensor.shape) == target_shape:
            logger.info("loaded teacher unembedding from %s (%s)", key, tuple(tensor.shape))
            return tensor.float()

    raise ValueError(
        f"No lm_head/embed_tokens candidate under {model_path} matches the text config "
        f"shape {target_shape} (vocab_size, hidden_size). Tried: {attempted}. "
        "Pass explicit `keys` for non-standard checkpoints."
    )
