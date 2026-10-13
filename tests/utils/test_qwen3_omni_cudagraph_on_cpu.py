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
"""Exercise the real vLLM bytecode guard without GPU kernels or model weights."""

from types import SimpleNamespace

import pytest
import torch
from torch._dynamo.convert_frame import register_bytecode_hook
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import CUDAGraphMode
from vllm.sequence import IntermediateTensors
from vllm_omni.model_executor.models.qwen3_omni import qwen3_omni_moe_thinker as thinker

from verl_omni.utils.vllm_omni import patch as compat

_ORIGINAL_FORWARD = thinker.Qwen3MoeLLMModel.forward


class _Layer(torch.nn.Module):
    def forward(self, positions, hidden_states, residual):
        return hidden_states * 2 + positions[:, None], hidden_states if residual is None else hidden_states + residual


class _Norm(torch.nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states + residual, None


class _Probe(torch.nn.Module):
    def __init__(self, start=0, end=2):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(), _Layer()])
        self.norm = _Norm()
        self.start_layer = start
        self.end_layer = end
        self.vllm_config = SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE),
            compile_debug_dump_path=lambda: None,
        )
        self.hook_calls = 0

    def embed_input_ids(self, input_ids):
        return input_ids[:, None].float().expand(-1, 4)

    def original_code_object(self):
        return self.__class__.forward.__code__

    def bytecode_hook(self, old_code, new_code):
        if old_code is self.original_code_object():
            self.hook_calls += 1
        return TorchCompileWithNoGuardsWrapper.bytecode_hook(self, old_code, new_code)


@pytest.fixture
def pp_group(monkeypatch):
    group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(thinker, "get_pp_group", lambda: group)
    monkeypatch.setattr(thinker.Qwen3MoeLLMModel, "forward", _ORIGINAL_FORWARD)
    return group


def _probe(forward, start=0, end=2):
    cls = type("ForwardProbe", (_Probe,), {"forward": forward})
    return cls(start, end)


def _args(capture=True, embeds=True):
    return dict(
        input_ids=torch.arange(3),
        positions=torch.arange(3),
        inputs_embeds=torch.ones(3, 4) if embeds else None,
        capture_layer_indices=[0, 1] if capture else None,
        return_hidden_states=capture,
    )


def _compile(model, args, monkeypatch):
    # CPU CI disables compile globally; this test must still exercise the real hook.
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "0")
    monkeypatch.setenv("TORCHINDUCTOR_DISABLE", "0")
    monkeypatch.setattr(torch._dynamo.config, "disable", False)
    torch._dynamo.reset()
    handle = register_bytecode_hook(model.bytecode_hook)
    try:
        return torch.compile(model.__class__.forward, backend="eager", fullgraph=True, dynamic=False)(model, **args)
    finally:
        handle.remove()
        torch._dynamo.reset()


def _flatten(output):
    if isinstance(output, IntermediateTensors):
        return output.tensors
    hidden_states, captured = output
    return {"output": hidden_states, **(captured["hidden_states"]["layers"] if captured else {})}


def test_pinned_forward_reproduces_guard_on_single_rank(pp_group, monkeypatch):
    assert "update" in _ORIGINAL_FORWARD.__code__.co_names
    model = _probe(_ORIGINAL_FORWARD)
    with pytest.raises(RuntimeError, match="Assigning / modifying buffers"):
        _compile(model, _args(capture=False), monkeypatch)
    assert model.hook_calls == 1


@pytest.mark.parametrize("last_rank", [True, False])
@pytest.mark.parametrize("capture", [True, False])
@pytest.mark.parametrize("embeds", [True, False])
def test_patched_forward_compiles_and_preserves_outputs(pp_group, monkeypatch, last_rank, capture, embeds):
    pp_group.is_last_rank = last_rank
    args = _args(capture, embeds)
    expected = _flatten(_probe(_ORIGINAL_FORWARD).forward(**args))
    bases = thinker.Qwen3MoeLLMModel.__bases__
    call = thinker.Qwen3MoeLLMModel.__call__
    compat.patch_qwen3_omni_thinker_forward()
    compat.patch_qwen3_omni_thinker_forward()
    assert thinker.Qwen3MoeLLMModel.__bases__ == bases
    assert thinker.Qwen3MoeLLMModel.__call__ is call
    assert "update" not in thinker.Qwen3MoeLLMModel.forward.__code__.co_names
    model = _probe(thinker.Qwen3MoeLLMModel.forward)
    actual = _flatten(_compile(model, args, monkeypatch))
    assert model.hook_calls == 1
    assert "update" not in model._compiled_bytecode.co_names
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])


def test_wrapper_delegates_all_arguments_to_upstream(pp_group, monkeypatch):
    args = _args()
    args["intermediate_tensors"] = IntermediateTensors({"hidden_states": torch.ones(3, 4)})
    args["deepstack_input_embeds"] = IntermediateTensors({"deepstack_input_embeds_0": torch.ones(3, 4)})
    model = _probe(compat._qwen3_omni_thinker_forward)
    calls = []
    result = object()

    def forward(self, **kwargs):
        calls.append((self, kwargs))
        return result

    monkeypatch.setattr(compat, "_ORIGINAL_THINKER_FORWARD", forward)
    assert model.forward(**args) is result
    assert calls[0][0] is model
    assert calls[0][1].keys() == args.keys()
    for name, value in args.items():
        assert calls[0][1][name] is value


@pytest.mark.parametrize("use_dict_update", [True, False])
def test_wrapper_still_rejects_real_buffer_mutations(pp_group, monkeypatch, use_dict_update):
    def forward(self, input_ids, **kwargs):
        if use_dict_update:
            self._buffers.update({"state": self.state + 1})
        else:
            self.state = self.state + 1
        return input_ids + self.state

    monkeypatch.setattr(compat, "_ORIGINAL_THINKER_FORWARD", forward)
    model = _probe(compat._qwen3_omni_thinker_forward)
    model.register_buffer("state", torch.tensor(0.0))
    with pytest.raises(RuntimeError, match="Assigning / modifying buffers"):
        _compile(model, _args(capture=False), monkeypatch)
    assert model.hook_calls == 1
    assert "update" in model._compiled_bytecode.co_names


def test_patch_leaves_an_already_graph_safe_forward_unchanged(pp_group, monkeypatch):
    monkeypatch.setattr(thinker.Qwen3MoeLLMModel, "forward", compat._qwen3_omni_thinker_forward)
    compat.patch_qwen3_omni_thinker_forward()
    assert thinker.Qwen3MoeLLMModel.forward is compat._qwen3_omni_thinker_forward


def test_patched_forward_preserves_deepstack_inputs(pp_group, monkeypatch):
    args = _args()
    args["deepstack_input_embeds"] = IntermediateTensors({"deepstack_input_embeds_0": torch.full((3, 4), 0.25)})
    expected = _flatten(_probe(_ORIGINAL_FORWARD).forward(**args))
    compat.patch_qwen3_omni_thinker_forward()
    actual = _flatten(_compile(_probe(thinker.Qwen3MoeLLMModel.forward), args, monkeypatch))
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])


def test_pipeline_capture_preserves_residual_and_owns_received_buffers(pp_group, monkeypatch):
    compat.patch_qwen3_omni_thinker_forward()
    forward = thinker.Qwen3MoeLLMModel.forward
    pp_group.is_last_rank = False
    first = _compile(_probe(forward, end=1), _args(), monkeypatch)
    pp_group.is_first_rank = False
    pp_group.is_last_rank = True
    args = _args()
    args["intermediate_tensors"] = first
    expected = _probe(_ORIGINAL_FORWARD, start=1).forward(**args)
    actual = _compile(_probe(forward, start=1), args, monkeypatch)
    torch.testing.assert_close(actual[0], expected[0])
    captured = actual[1]["hidden_states"]["layers"]
    torch.testing.assert_close(captured[0], torch.ones(3, 4))
    torch.testing.assert_close(captured[1], first["hidden_states"] + first["residual"])
    first[f"{thinker.PP_CAPTURE_PREFIX}0"].fill_(-999)
    torch.testing.assert_close(captured[0], torch.ones(3, 4))


@pytest.mark.parametrize(
    "architectures, should_patch",
    [
        (["Qwen3OmniMoeForConditionalGeneration"], True),
        (["Qwen3OmniMoeThinkerForConditionalGeneration"], True),
        (["Qwen3TTSTalkerForConditionalGeneration"], False),
        ([], False),
    ],
)
def test_worker_installs_patch_before_initialization(monkeypatch, architectures, should_patch):
    from verl_omni.workers.rollout.vllm_rollout import utils

    calls = []
    monkeypatch.setattr(utils, "set_death_signal", lambda: None)
    monkeypatch.setattr(utils.VLLMOmniHijack, "hijack", lambda: None)
    monkeypatch.setattr(compat, "patch_qwen3_omni_thinker_forward", lambda: calls.append("patch"))

    class Worker(utils.vLLMOmniColocateWorkerExtension):
        def __init__(self, **kwargs):
            calls.append("init")

    Worker(vllm_config=SimpleNamespace(model_config=SimpleNamespace(architectures=architectures)))
    assert calls == (["patch", "init"] if should_patch else ["init"])
