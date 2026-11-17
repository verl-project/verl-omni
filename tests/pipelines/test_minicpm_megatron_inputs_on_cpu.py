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
"""Check MiniCPM's Megatron boundary with actual BSHD helpers on CPU."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("verl.models.mcore.util", reason="Megatron-Core is optional in CPU CI")

from verl_omni.pipelines.minicpm.megatron_inputs import minicpm_forward_model_engine
from verl_omni.pipelines.minicpm.thinker_training_adapter import MiniCPMThinkerAdapter
from verl_omni.pipelines.model_base import OmniModelBase


class Tokenizer:
    im_start_id, im_end_id, slice_start_id, slice_end_id, audio_start_id, audio_end_id = range(100, 106)

    def convert_tokens_to_ids(self, value):
        return 106 if value == "<unk>" else -1

    def decode(self, ids):
        return "".join(f"<{value}>" for value in ids)

    def encode(self, text, **kwargs):
        return [1]


def config():
    return SimpleNamespace(
        hf_config=SimpleNamespace(model_type="minicpmo", version="4.5", hidden_size=16),
        model_stage="thinker",
        trust_remote_code=True,
        mtp=SimpleNamespace(enable=False),
        processor=Tokenizer(),
    )


def engine_config():
    return SimpleNamespace(
        use_remove_padding=False,
        use_fused_kernels=False,
        sequence_parallel=False,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
    )


@pytest.fixture(autouse=True)
def cpu_groups(monkeypatch):
    from verl.models.mcore import util

    monkeypatch.setattr(util.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(util.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(util.mpu, "get_context_parallel_group", lambda: None)
    monkeypatch.setattr(util.mpu, "get_tensor_model_parallel_world_size", lambda: 2)


def nested(*rows):
    return torch.nested.nested_tensor([torch.tensor(row) for row in rows], layout=torch.jagged)


class Model(torch.nn.Module):
    pre_process = True
    post_process = True
    config = SimpleNamespace(fp8=None)

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, **kwargs):
        self.seen = kwargs
        return kwargs["input_ids"].float().unsqueeze(-1) * self.weight


def test_config_isolation_and_legacy_forward_binding():
    original = config()
    prepared = MiniCPMThinkerAdapter.prepare_megatron_config(original, engine_config())
    assert prepared is not original and not hasattr(original.hf_config, "text_config")
    assert prepared.hf_config.text_config.hidden_size == 16
    assert MiniCPMThinkerAdapter.bind_megatron_forward(prepared).keywords["model_config"] is prepared

    class ExistingAdapter(OmniModelBase):
        @classmethod
        def get_megatron_forward(cls):
            return int

    assert ExistingAdapter.bind_megatron_forward(original) is int


@pytest.mark.parametrize(
    "field,value",
    [
        ("use_remove_padding", True),
        ("use_fused_kernels", True),
        ("sequence_parallel", True),
        ("pipeline_model_parallel_size", 2),
        ("context_parallel_size", 2),
        ("vanilla_mbridge", True),
    ],
)
def test_unsupported_modes_fail_before_construction(field, value):
    engine = engine_config()
    setattr(engine, field, value)
    with pytest.raises(ValueError):
        MiniCPMThinkerAdapter.prepare_megatron_config(config(), engine)


def test_dense_positions_causal_mask_label_shift_and_gradients():
    ids = nested([1, 2, 3], [4, 5])
    model = Model()
    seen = {}

    def process(logits, label, temperature):
        seen.update(label=label, temperature=temperature)
        return {"log_probs": logits.squeeze(-1) * temperature + label}

    result = minicpm_forward_model_engine(
        model,
        ids,
        {},
        model_config=config(),
        pad_token_id=0,
        logits_processor=process,
        logits_processor_args={"label": ids, "temperature": ids * 0 + 1.5, "loss_mask": ids},
    )["log_probs"]
    torch.testing.assert_close(seen["label"], torch.tensor([[2, 3, 0, 1], [5, 0, 0, 4]]))
    assert model.seen["position_ids"].tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    mask = model.seen["attention_mask"]
    assert mask.shape == (2, 1, 4, 4) and mask.dtype == torch.bool
    assert bool(mask[0, 0, 0, 1]) and not bool(mask[0, 0, 2, 0])
    assert bool(mask[1, 0, 3, 2])
    torch.testing.assert_close(result.values(), torch.tensor([2.375, 3.75, 1.125, 6.5, 1.875]))
    result.values().sum().backward()
    torch.testing.assert_close(model.weight.grad, torch.tensor(22.5))


def test_audio_bounds_follow_each_row_after_padding():
    model = Model()
    ids = nested([104, 106, 105, 1], [1, 104, 106, 106, 105])
    media = {
        "audio_features": [torch.ones(1, 80, 2), torch.ones(1, 80, 4)],
        "audio_feature_lens": [[torch.tensor([2])], [torch.tensor([4])]],
        "image_bound": [[], []],
        "pixel_values": [[], []],
        "tgt_sizes": [[], []],
    }
    minicpm_forward_model_engine(model, ids, media, model_config=config(), pad_token_id=0)
    prepared = model.seen["media"]
    assert [row.tolist() for row in prepared["audio_bounds"]] == [[[1, 2]], [[2, 4]]]
    assert [row.tolist() for row in prepared["audio_feature_lens"]] == [[2], [4]]
    assert prepared["audio_features"].shape == (2, 80, 4)


def test_missing_media_is_rejected():
    with pytest.raises(ValueError, match="no media features"):
        minicpm_forward_model_engine(Model(), nested([104, 106, 105]), {}, model_config=config())
