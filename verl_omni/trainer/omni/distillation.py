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
"""Tokenizer compatibility checks for the existing omni OPD trainer."""

from transformers import AutoTokenizer
from verl.utils.config import omega_conf_to_dataclass

from verl_omni.utils.fs import resolve_model_local_dir


def validate_teacher_tokenizers(tokenizer, config):
    """Require compatible vocabulary and special-token mappings before allocating workers."""
    distillation = omega_conf_to_dataclass(config.distillation)
    for teacher in distillation.teacher_models.values():
        kwargs = teacher.inference.engine_kwargs.get("vllm_omni", {})
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            resolve_model_local_dir(teacher.model_path), trust_remote_code=kwargs.get("trust_remote_code", False)
        )
        if tokenizer.get_vocab() != teacher_tokenizer.get_vocab() or any(
            getattr(tokenizer, field) != getattr(teacher_tokenizer, field)
            for field in ("bos_token_id", "eos_token_id", "pad_token_id", "all_special_ids")
        ):
            raise ValueError(
                f"Omni OPD teacher {teacher.key!r} must share the student's tokenizer and special-token IDs."
            )
