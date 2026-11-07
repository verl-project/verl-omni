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
"""CPU checks that the VeOmni diffusion engine honors attn_implementation for Qwen-Image."""

from types import SimpleNamespace

import pytest
import torch

# The engine imports veomni, which the CPU CI does not install.
pytest.importorskip("veomni")
import verl_omni.workers.engine.veomni.patch as veomni_patch
from tests.workers.veomni_lora_helpers import make_veomni_engine

qwen_image = pytest.importorskip(
    "veomni.models.diffusers.qwen_image.qwen_image_transformer.modeling_qwen_image_transformer"
)


def _tiny_qwen_image():
    config = qwen_image.QwenImageTransformer2DModelConfig(
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        in_channels=4,
        out_channels=4,
        axes_dims_rope=(2, 2, 4),
    )
    return qwen_image.QwenImageTransformer2DModel(config)


def _backend(model):
    return model.transformer_blocks[0].attn.processor._attention_backend.value


@pytest.fixture
def no_kernel_download(monkeypatch):
    import diffusers.models.attention_dispatch as dispatch

    monkeypatch.setattr(dispatch, "_check_attention_backend_requirements", lambda backend: None)
    monkeypatch.setattr(dispatch, "_maybe_download_kernel_for_backend", lambda backend: None)


@pytest.mark.parametrize(
    ("attn_implementation", "backend"),
    [
        ("eager", "native"),
        ("flash_attention_2_hub", "flash_varlen_hub"),
        ("flash_attention_3_hub", "_flash_3_varlen_hub"),
    ],
)
def test_qwen_image_uses_the_requested_backend(no_kernel_download, monkeypatch, attn_implementation, backend):
    from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry

    monkeypatch.setattr(_AttentionBackendRegistry, "_active_backend", AttentionBackendName.NATIVE)
    model = _tiny_qwen_image()
    veomni_patch._apply_attention_backend(model, attn_implementation)
    assert _backend(model) == backend
    # Only this model switches: diffusers' process-wide default stays as it was.
    assert _AttentionBackendRegistry._active_backend == AttentionBackendName.NATIVE


@pytest.mark.parametrize("attn_implementation", ["sdpa", "flash_attention_2", "flash_attention_3", "flex_attention"])
@pytest.mark.parametrize("make_model", [_tiny_qwen_image, lambda: torch.nn.Linear(4, 4)], ids=["qwen_image", "other"])
def test_names_outside_the_allowlist_are_rejected_for_every_model(make_model, attn_implementation):
    with pytest.raises(ValueError, match="not supported by the VeOmni diffusion engine"):
        veomni_patch._apply_attention_backend(make_model(), attn_implementation)


@pytest.mark.parametrize("attn_implementation", ["eager", "flash_attention_2_hub", "flash_attention_3_hub"])
def test_other_dits_keep_veomni_attention_selection(monkeypatch, attn_implementation):
    """Wan / MiniMax H3 / LTX read allowlisted names inside VeOmni; the engine must not touch them."""
    _set_veomni_hub_support(monkeypatch, True)
    veomni_patch._apply_attention_backend(torch.nn.Linear(4, 4), attn_implementation)


def _set_veomni_hub_support(monkeypatch, supported: bool):
    from veomni.arguments import OpsImplementationConfig

    if supported:
        monkeypatch.setattr(
            OpsImplementationConfig, "normalize_hub_attention_backend", staticmethod(lambda x: x), raising=False
        )
    else:
        monkeypatch.delattr(OpsImplementationConfig, "normalize_hub_attention_backend", raising=False)


@pytest.mark.parametrize(
    ("supported", "requested", "passed_to_veomni"),
    [
        (False, "flash_attention_3_hub", "eager"),
        (False, "flash_attention_2_hub", "eager"),
        (False, "flash_attention_2", "flash_attention_2"),
        (True, "flash_attention_3_hub", "flash_attention_3_hub"),
    ],
)
def test_hub_names_build_with_eager_only_when_veomni_cannot_parse_them(
    monkeypatch, supported, requested, passed_to_veomni
):
    _set_veomni_hub_support(monkeypatch, supported)
    assert veomni_patch._veomni_attn_implementation(requested) == passed_to_veomni


def test_ops_config_receives_the_buildable_attention_name(monkeypatch):
    _set_veomni_hub_support(monkeypatch, False)
    engine = make_veomni_engine()
    engine.engine_config = SimpleNamespace(attn_implementation="flash_attention_3_hub")
    assert engine._build_ops_config().attn_implementation == "eager"


def test_other_dits_reject_hub_names_the_installed_veomni_cannot_parse(monkeypatch):
    """Without this, Wan & co. would silently train with eager attention."""
    _set_veomni_hub_support(monkeypatch, False)
    with pytest.raises(ValueError, match="requires a VeOmni release with Hub attention support"):
        veomni_patch._apply_attention_backend(torch.nn.Linear(4, 4), "flash_attention_3_hub")
