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

"""CPU coverage for the Boogu-Image adapters.

Scoped to the contracts that silently corrupt training when broken: registration,
the processor-config hook released checkpoints omit, Boogu's text CFG, and the
condition-latent mapping that makes the T2I and Edit (TI2I) paths share one
adapter. The time-shift/sigma conventions live in
``test_boogu_image_common_on_cpu.py``.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict

from verl_omni.pipelines.boogu_image_flow_grpo.common import (
    apply_boogu_text_cfg,
    resolve_text_guidance_scale,
)
from verl_omni.pipelines.boogu_image_flow_grpo.diffusers_training_adapter import BooguImage
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config.diffusion.model import DiffusionModelConfig


def _model_config(local_path: str = "/nonexistent") -> DiffusionModelConfig:
    config = object.__new__(DiffusionModelConfig)
    object.__setattr__(config, "architecture", "BooguImagePipeline")
    object.__setattr__(config, "external_lib", None)
    object.__setattr__(config, "algorithm", "flow_grpo")
    object.__setattr__(config, "local_path", local_path)
    return config


class _FakeTransformerConfig:
    axes_dim_rope = (4, 2, 2)
    axes_lens = (16, 16, 16)


class _FakeModule:
    config = _FakeTransformerConfig()


def _prepare_inputs(micro_batch: TensorDict, *, negative: bool = True, batch: int = 2):
    """Drive prepare_model_inputs with one denoising step of tiny tensors."""
    latents = torch.zeros(batch, 1, 3, 4, 4)
    timesteps = torch.full((batch, 1), 250.0)
    embeds = torch.zeros(batch, 5, 8)
    mask = torch.ones(batch, 5, dtype=torch.long)
    with patch(
        "verl_omni.pipelines.boogu_image_flow_grpo.diffusers_training_adapter.get_boogu_freqs_cis",
        return_value="freqs",
    ):
        return BooguImage.prepare_model_inputs(
            module=_FakeModule(),
            model_config=_model_config(),
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=embeds,
            prompt_embeds_mask=mask,
            negative_prompt_embeds=embeds if negative else None,
            negative_prompt_embeds_mask=mask if negative else None,
            micro_batch=micro_batch,
            step=0,
        )


# --------------------------------------------------------------- registration


def test_get_class_resolves_boogu_registration():
    assert DiffusionModelBase.get_class(_model_config()) is BooguImage


# ----------------------------------------------------------- processor config


def test_processor_hook_materialises_the_processor_config(tmp_path):
    """Released checkpoints ship processor/ without config.json; mllm/ has the real one."""
    checkpoint = tmp_path / "with_mllm"
    (checkpoint / "processor").mkdir(parents=True)
    (checkpoint / "mllm").mkdir()
    (checkpoint / "mllm" / "config.json").write_text('{"model_type": "qwen3_vl_moe"}', encoding="utf-8")

    prepared = BooguImage.prepare_processor_files(str(checkpoint))

    assert prepared == str(checkpoint / "processor")
    assert json.loads((Path(prepared) / "config.json").read_text(encoding="utf-8")) == {"model_type": "qwen3_vl_moe"}

    bare = tmp_path / "without_mllm"
    (bare / "processor").mkdir(parents=True)
    prepared = BooguImage.prepare_processor_files(str(bare))
    assert json.loads((Path(prepared) / "config.json").read_text(encoding="utf-8")) == {"model_type": "qwen3_vl"}

    # Unlike the Qwen-Image-Edit adapter, a missing processor/ is not an error.
    assert BooguImage.prepare_processor_files(str(tmp_path / "empty")) is None


# -------------------------------------------------------------------- text CFG


def test_text_cfg_follows_the_standard_formula():
    positive = torch.tensor([2.0, 4.0])
    negative = torch.tensor([1.0, 1.0])

    out = apply_boogu_text_cfg(positive, negative, 4.0)

    # noise + (scale - 1) * (noise - negative); no renormalisation.
    torch.testing.assert_close(out, torch.tensor([5.0, 13.0]))
    # Boogu's default scale is 4.0, not the diffusers-wide 1.0.
    assert resolve_text_guidance_scale(None) == 4.0
    assert resolve_text_guidance_scale(2.5) == 2.5


# ------------------------------------------------- condition latents (T2I/Edit)


def test_prepare_model_inputs_degenerates_to_t2i_without_condition_latents():
    model_inputs, negative_inputs = _prepare_inputs(TensorDict({}, batch_size=[2]))

    assert model_inputs["ref_image_hidden_states"] is None
    assert negative_inputs["ref_image_hidden_states"] is None
    # Boogu consumes prompts as instruction_*, not encoder_hidden_states.
    assert "instruction_hidden_states" in model_inputs
    assert "instruction_attention_mask" in model_inputs

    _, negative_inputs = _prepare_inputs(TensorDict({}, batch_size=[2]), negative=False)
    assert negative_inputs is None


def test_prepare_model_inputs_wraps_condition_latents_for_the_refiner_path():
    """Edit path: (B, C, H, W) latents become the per-sample [[image]] nesting."""
    condition = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)

    model_inputs, negative_inputs = _prepare_inputs(TensorDict({"condition_image_latents": condition}, batch_size=[2]))

    ref = model_inputs["ref_image_hidden_states"]
    assert isinstance(ref, list) and len(ref) == 2
    assert all(isinstance(entry, list) and len(entry) == 1 for entry in ref)
    torch.testing.assert_close(ref[0][0], condition[0])
    torch.testing.assert_close(ref[1][0], condition[1])
    # Rollout feeds the reference latents to both CFG branches; training must match.
    assert negative_inputs["ref_image_hidden_states"] is ref


@pytest.mark.parametrize(
    ("condition", "micro_batch_size"),
    [
        (torch.zeros(2, 3, 16), 2),  # (B, C, H*W) instead of (B, C, H, W)
        (torch.zeros(3, 3, 4, 4), 3),  # wider than the micro-batch
    ],
)
def test_prepare_model_inputs_rejects_bad_condition_latents(condition, micro_batch_size):
    micro_batch = TensorDict({"condition_image_latents": condition}, batch_size=[micro_batch_size])
    with pytest.raises(ValueError):
        _prepare_inputs(micro_batch, batch=2)
