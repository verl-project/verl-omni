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


def test_processor_hook_copies_the_mllm_config(tmp_path):
    """Released checkpoints ship processor/ without config.json; mllm/ has the real one."""
    (tmp_path / "processor").mkdir()
    (tmp_path / "mllm").mkdir()
    (tmp_path / "mllm" / "config.json").write_text('{"model_type": "qwen3_vl_moe"}', encoding="utf-8")

    prepared = BooguImage.prepare_processor_files(str(tmp_path))

    assert prepared == str(tmp_path / "processor")
    assert json.loads((Path(prepared) / "config.json").read_text(encoding="utf-8")) == {
        "model_type": "qwen3_vl_moe"
    }


def test_processor_hook_falls_back_when_mllm_config_is_absent(tmp_path):
    (tmp_path / "processor").mkdir()

    prepared = BooguImage.prepare_processor_files(str(tmp_path))

    assert json.loads((Path(prepared) / "config.json").read_text(encoding="utf-8")) == {"model_type": "qwen3_vl"}


def test_processor_hook_preserves_an_existing_config(tmp_path):
    (tmp_path / "processor").mkdir()
    config_path = tmp_path / "processor" / "config.json"
    config_path.write_text('{"model_type": "custom"}', encoding="utf-8")

    BooguImage.prepare_processor_files(str(tmp_path))

    assert json.loads(config_path.read_text(encoding="utf-8")) == {"model_type": "custom"}


def test_processor_hook_returns_none_without_a_processor_dir(tmp_path):
    assert BooguImage.prepare_processor_files(str(tmp_path)) is None


# -------------------------------------------------------------------- text CFG


def test_text_cfg_follows_the_standard_formula():
    positive = torch.tensor([2.0, 4.0])
    negative = torch.tensor([1.0, 1.0])

    out = apply_boogu_text_cfg(positive, negative, 4.0)

    # noise + (scale - 1) * (noise - negative); no renormalisation.
    torch.testing.assert_close(out, torch.tensor([5.0, 13.0]))


def test_text_cfg_is_identity_at_scale_one():
    positive = torch.tensor([2.0, 4.0])

    out = apply_boogu_text_cfg(positive, torch.tensor([1.0, 1.0]), 1.0)

    torch.testing.assert_close(out, positive)


def test_guidance_scale_defaults_to_boogu_four():
    assert resolve_text_guidance_scale(None) == 4.0
    assert resolve_text_guidance_scale(2.5) == 2.5
    assert resolve_text_guidance_scale(1) == 1.0


# ------------------------------------------------- condition latents (T2I/Edit)


def test_prepare_model_inputs_degenerates_to_t2i_without_condition_latents():
    model_inputs, negative_inputs = _prepare_inputs(TensorDict({}, batch_size=[2]))

    assert model_inputs["ref_image_hidden_states"] is None
    assert negative_inputs["ref_image_hidden_states"] is None
    # Boogu consumes prompts as instruction_*, not encoder_hidden_states.
    assert "instruction_hidden_states" in model_inputs
    assert "instruction_attention_mask" in model_inputs


def test_prepare_model_inputs_wraps_condition_latents_for_the_refiner_path():
    """Edit path: (B, C, H, W) latents become the per-sample [[image]] nesting."""
    condition = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)

    model_inputs, _ = _prepare_inputs(TensorDict({"condition_image_latents": condition}, batch_size=[2]))

    ref = model_inputs["ref_image_hidden_states"]
    assert isinstance(ref, list) and len(ref) == 2
    assert all(isinstance(entry, list) and len(entry) == 1 for entry in ref)
    torch.testing.assert_close(ref[0][0], condition[0])
    torch.testing.assert_close(ref[1][0], condition[1])


def test_prepare_model_inputs_shares_condition_latents_with_the_negative_branch():
    """Rollout feeds the reference latents to both CFG branches; training must match."""
    condition = torch.zeros(2, 3, 4, 4)

    model_inputs, negative_inputs = _prepare_inputs(
        TensorDict({"condition_image_latents": condition}, batch_size=[2])
    )

    assert negative_inputs["ref_image_hidden_states"] is model_inputs["ref_image_hidden_states"]


def test_prepare_model_inputs_rejects_misshaped_condition_latents():
    # (B, C, H*W) instead of (B, C, H, W).
    with pytest.raises(ValueError, match=r"must be \(B, C, H, W\)"):
        _prepare_inputs(TensorDict({"condition_image_latents": torch.zeros(2, 3, 16)}, batch_size=[2]))


def test_prepare_model_inputs_rejects_condition_batch_mismatch():
    """A micro-batch wider than the latents must fail loudly, not broadcast."""
    micro_batch = TensorDict({"condition_image_latents": torch.zeros(3, 3, 4, 4)}, batch_size=[3])
    with pytest.raises(ValueError, match="micro-batch batch size"):
        _prepare_inputs(micro_batch, batch=2)


def test_prepare_model_inputs_skips_the_negative_branch_without_cfg():
    model_inputs, negative_inputs = _prepare_inputs(TensorDict({}, batch_size=[2]), negative=False)

    assert negative_inputs is None
    assert model_inputs["return_dict"] is False
