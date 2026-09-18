# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""CPU wiring tests for diffusion regional ``torch.compile``."""

from types import SimpleNamespace

import pytest
import torch

from verl_omni.utils.diffusion_compile import _maybe_compile_repeated_blocks


class _RegionalCompileModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.compile_calls = []

    def compile_repeated_blocks(self, **kwargs) -> None:
        self.compile_calls.append(kwargs)


def _model_config(*, enabled: bool, options=None):
    return SimpleNamespace(
        use_regional_compile=enabled,
        regional_compile_options={"backend": "inductor", "fullgraph": True} if options is None else options,
    )


def _engine_config(*, strategy: str = "fsdp2", sp_size: int = 1):
    return SimpleNamespace(strategy=strategy, ulysses_sequence_parallel_size=sp_size)


@pytest.fixture(autouse=True)
def eager_boundary_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "verl_omni.utils.diffusion_compile._keep_varlen_attention_metadata_eager",
        lambda: calls.append(True),
    )
    return calls


def test_regional_compile_skips_disabled_model(eager_boundary_calls):
    model = _RegionalCompileModel()

    _maybe_compile_repeated_blocks(model, _model_config(enabled=False), _engine_config())

    assert model.compile_calls == []
    assert eager_boundary_calls == []


def test_regional_compile_forwards_options_without_mutating_them(eager_boundary_calls):
    model = _RegionalCompileModel()
    options = {"backend": "inductor", "mode": "default", "fullgraph": True, "dynamic": False}

    _maybe_compile_repeated_blocks(model, _model_config(enabled=True, options=options), _engine_config())

    assert model.compile_calls == [options]
    assert options == {"backend": "inductor", "mode": "default", "fullgraph": True, "dynamic": False}
    assert eager_boundary_calls == [True]


@pytest.mark.parametrize(
    ("strategy", "sp_size", "message"),
    [
        ("fsdp", 1, "FSDP1.*has not been validated.*use_orig_params=True"),
        ("fsdp2", 2, "Ulysses SP.*distributed integration has not been validated"),
    ],
)
def test_regional_compile_rejects_unsupported_parallelism(strategy, sp_size, message):
    with pytest.raises(NotImplementedError, match=message):
        _maybe_compile_repeated_blocks(
            _RegionalCompileModel(),
            _model_config(enabled=True),
            _engine_config(strategy=strategy, sp_size=sp_size),
        )
