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
"""Exercise full-weight and LoRA reload with changed layouts and reused IPC storage."""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.reload import record_metadata_for_reloading

from verl_omni.workers.rollout.vllm_rollout.utils import _require_lora_modules_bound, vLLMOmniColocateWorkerExtension


class _PackedMethod(QuantizeMethodBase):
    def create_weights(self, layer, **kwargs):
        raise NotImplementedError

    def apply(self, layer, *args, **kwargs):
        raise NotImplementedError

    def process_weights_after_loading(self, layer):
        assert layer.weight.shape == (2, 2)
        layer.weight.data = layer.weight.data.reshape(1, 2, 2)


class _PackedExperts(RoutedExperts):
    def __init__(self):
        # Exercise the real expert-layer type without a process group or GPU.
        torch.nn.Module.__init__(self)
        self.weight = torch.nn.Parameter(torch.zeros(2, 2))
        self.bias = torch.nn.Parameter(torch.zeros(2))
        self.quant_method = _PackedMethod()


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = _PackedExperts()
        self.layer.weight.weight_loader = self._load_weight
        self.layer.bias.weight_loader = self._load_weight

    @staticmethod
    def _load_weight(param, loaded_weight):
        assert param.shape == loaded_weight.shape
        param.data.copy_(loaded_weight)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        for name, tensor in weights:
            params[name].weight_loader(params[name], tensor)


def test_full_weight_reload_restores_layout_and_owns_cross_bucket_tensors(monkeypatch):
    from verl.utils.vllm import patch as verl_patch
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

    from verl_omni.workers.rollout.vllm_rollout import npu_utils

    monkeypatch.setattr(verl_patch, "patch_vllm_moe_model_weight_loader", lambda model: None)
    monkeypatch.setattr(npu_utils, "_is_npu_platform", lambda: False)
    model = _Model()
    record_metadata_for_reloading(model)
    model.layer.quant_method.process_weights_after_loading(model.layer)
    storage = {name: param.data_ptr() for name, param in model.named_parameters()}
    received_values = iter(((3.0, 7.0), (5.0, 11.0)))

    class Receiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            weight_value, bias_value = next(received_values)
            buffer = torch.full((4,), weight_value)
            on_bucket_received([("layer.weight", buffer.view(2, 2))], False)
            # The first tensor is buffered until the bias arrives. Reusing its
            # storage here detects callbacks that retain IPC views without copies.
            buffer.fill_(bias_value)
            on_bucket_received([("layer.bias", buffer[:2])], True)
            buffer.fill_(-99)

    monkeypatch.setattr(bucketed_weight_transfer, "BucketedWeightReceiver", Receiver)
    worker = SimpleNamespace(
        device=torch.device("cpu"),
        _pending_lora_peft_config=None,
        _get_zmq_handle=lambda: "test-reload",
        _get_standard_weight_model_and_config=lambda: (model, SimpleNamespace(dtype=torch.float32)),
    )
    for weight_value, bias_value in ((3.0, 7.0), (5.0, 11.0)):
        vLLMOmniColocateWorkerExtension.update_weights_from_ipc(worker)
        torch.testing.assert_close(model.layer.weight, torch.full((1, 2, 2), weight_value))
        torch.testing.assert_close(model.layer.bias, torch.full((2,), bias_value))
        assert {name: param.data_ptr() for name, param in model.named_parameters()} == storage


class _DenseOmniModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.talker = torch.nn.Linear(2, 2, bias=False)
        self.encoder = torch.nn.Module()
        self.encoder.register_buffer("embed", torch.ones(2, 2))
        self.encoder.register_buffer("runtime", torch.ones(2), persistent=False)
        self.post_load_calls = 0

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        for name, tensor in weights:
            params[name].data.copy_(tensor)
        # The pinned TTS loader directly loads encoder buffers, moves the
        # encoder, and constructs runtime tensors before returning.
        self.encoder.embed.copy_(torch.full((2, 2), 7.0))
        self.encoder.to(device="cpu")
        self.codec_embed = self.talker.weight.detach().clone()

    def process_weights_after_loading(self):
        self.post_load_calls += 1


def test_dense_omni_reload_preserves_auxiliary_buffers_and_post_load_hook(monkeypatch):
    from verl.utils.vllm import patch as verl_patch
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

    from verl_omni.workers.rollout.vllm_rollout import npu_utils

    monkeypatch.setattr(verl_patch, "patch_vllm_moe_model_weight_loader", lambda model: None)
    monkeypatch.setattr(npu_utils, "_is_npu_platform", lambda: False)
    model = _DenseOmniModel()
    record_metadata_for_reloading(model)
    storage = {name: tensor.data_ptr() for name, tensor in model.state_dict().items()}
    received_values = iter((3.0, 5.0))

    class Receiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            buffer = torch.full((2, 2), next(received_values))
            on_bucket_received([("talker.weight", buffer)], True)
            buffer.fill_(-99)

    monkeypatch.setattr(bucketed_weight_transfer, "BucketedWeightReceiver", Receiver)
    worker = SimpleNamespace(
        device=torch.device("cpu"),
        _pending_lora_peft_config=None,
        _get_zmq_handle=lambda: "test-dense-reload",
        _get_standard_weight_model_and_config=lambda: (
            model,
            SimpleNamespace(
                dtype=torch.float32,
                quantization=None,
                word_embeddings_untied_by_checkpoint=False,
            ),
        ),
    )
    for step, value in enumerate((3.0, 5.0), 1):
        vLLMOmniColocateWorkerExtension.update_weights_from_ipc(worker)
        torch.testing.assert_close(model.talker.weight, torch.full((2, 2), value))
        torch.testing.assert_close(model.codec_embed, model.talker.weight)
        torch.testing.assert_close(model.encoder.embed, torch.full((2, 2), 7.0))
        torch.testing.assert_close(model.encoder.runtime, torch.ones(2))
        assert "runtime" in model.encoder._non_persistent_buffers_set
        assert {name: tensor.data_ptr() for name, tensor in model.state_dict().items()} == storage
        assert model.post_load_calls == step


def _lora_manager(loras, modules):
    adapter_manager = SimpleNamespace(get_adapter=lambda lora_id: SimpleNamespace(loras=loras), modules=modules)
    return SimpleNamespace(_adapter_manager=adapter_manager)


@pytest.mark.parametrize("ar_worker", [True, False], ids=["ar", "diffusion"])
def test_lora_sync_owns_tensors_across_reused_ipc_buckets(monkeypatch, ar_worker):
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

    class Receiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            # One reused buffer, as in BucketedWeightReceiver: an adapter larger
            # than a bucket arrives as views into the same storage.
            buffer = torch.full((4,), 3.0)
            on_bucket_received([("lora_A.weight", buffer.view(2, 2))], False)
            buffer.fill_(5.0)
            on_bucket_received([("lora_B.weight", buffer.view(2, 2))], True)
            # Stand-in for the sender cleaning up its buffer after the final ack,
            # before add_lora runs: the last bucket must be copied too.
            buffer.fill_(0.0)

    monkeypatch.setattr(bucketed_weight_transfer, "BucketedWeightReceiver", Receiver)
    added = []
    worker = SimpleNamespace(
        device=torch.device("cpu"),
        _pending_lora_peft_config=None,
        _get_zmq_handle=lambda: "test-lora-reload",
        _get_standard_weight_model_and_config=lambda: (object(), object()) if ar_worker else None,
        _move_diffusion_lora_stacks_to_device=lambda: None,
        model_runner=SimpleNamespace(lora_manager=_lora_manager(loras=set(), modules=set())),
        remove_lora=lambda lora_id: None,
        add_lora=added.append,
    )

    vLLMOmniColocateWorkerExtension.update_weights_from_ipc(worker, peft_config={"r": 2}, base_sync_done=True)

    (request,) = added
    torch.testing.assert_close(request.lora_tensors["lora_A.weight"], torch.full((2, 2), 3.0))
    torch.testing.assert_close(request.lora_tensors["lora_B.weight"], torch.full((2, 2), 5.0))


def test_lora_binding_accepts_packed_and_3d_moe_modules():
    attention = "language_model.model.layers.0.self_attn"
    experts = "language_model.model.layers.0.mlp.experts"
    modules = {f"{attention}.qkv_proj", f"{attention}.o_proj", experts}

    _require_lora_modules_bound(
        _lora_manager(
            loras={f"{attention}.qkv_proj", f"{attention}.o_proj", experts, f"{experts}.base_layer"}, modules=modules
        )
    )


def test_ar_lora_sync_rejects_adapter_module_without_rollout_layer(monkeypatch):
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

    class Receiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            on_bucket_received([("thinker.model.layers.0.self_attn.q_proj.lora_A.weight", torch.zeros(2, 2))], True)

    monkeypatch.setattr(bucketed_weight_transfer, "BucketedWeightReceiver", Receiver)
    unbound = "thinker.model.layers.0.self_attn.q_proj"
    worker = SimpleNamespace(
        device=torch.device("cpu"),
        _pending_lora_peft_config=None,
        _get_zmq_handle=lambda: "test-lora-binding",
        _get_standard_weight_model_and_config=lambda: (object(), object()),
        model_runner=SimpleNamespace(lora_manager=_lora_manager(loras={unbound}, modules={"language_model.o_proj"})),
        remove_lora=lambda lora_id: None,
        add_lora=lambda request: None,
    )

    with pytest.raises(ValueError, match=unbound):
        vLLMOmniColocateWorkerExtension.update_weights_from_ipc(worker, peft_config={"r": 2}, base_sync_done=True)
