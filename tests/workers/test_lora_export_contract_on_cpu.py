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
"""CPU tests for the shared LoRA export contract (``lora_export.export_lora_for_sync``).

Pins, for both engines behind one implementation:
- the dtype matrix (adapter tensors keep training dtype; full-weight DTensor
  gathers cast floating point to bf16; plain tensors pass through),
- merged streams materialize inside ``merged_lora_context`` and stay valid
  after the actor is restored; on FSDP2 with vanilla LoRA they are computed per
  tensor, bitwise equal to PEFT's merge, without modifying the module,
- engine memory management runs at the right point of each path,
- the colocated sync ordering rows (base before adapter; single full sync in
  merge/non-LoRA modes).
"""

import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from torch.distributed.tensor import DTensor

import verl_omni.workers.engine.lora_export as lora_export
import verl_omni.workers.engine_workers as ew
from verl_omni.workers.engine.lora_export import export_lora_for_sync

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _peft_module():
    module = torch.nn.Module()
    module.proj = torch.nn.Linear(4, 4, bias=False)
    module.peft_config = {
        "default": SimpleNamespace(to_dict=lambda: {"r": 8}),
        "old": SimpleNamespace(to_dict=lambda: {"r": 8, "adapter": "old"}),
    }
    return module


def _plain_module():
    return torch.nn.Linear(4, 4, bias=False)


def _model_config(merge: bool = False):
    return SimpleNamespace(lora={"merge": merge}, fsdp_layer_prefixes=["layers."])


def _floating_dtensor(value: float = 1.25) -> MagicMock:
    gathered = torch.tensor([value], dtype=torch.float32)
    dtensor = MagicMock(spec=DTensor)
    dtensor.to.return_value.full_tensor.return_value = gathered
    return dtensor


def _integer_dtensor() -> MagicMock:
    gathered = torch.tensor([1, 2], dtype=torch.int64)
    dtensor = MagicMock(spec=DTensor)
    dtensor.to.return_value.full_tensor.return_value = gathered
    return dtensor


def _patch_export_machinery(monkeypatch, collect=None):
    monkeypatch.setattr(lora_export, "convert_weight_keys", lambda params, module: params)
    monkeypatch.setattr(lora_export, "get_device_id", lambda: torch.device("cpu"))
    if collect is not None:
        monkeypatch.setattr(lora_export, "collect_lora_params", collect)


# ---------------------------------------------------------------------------
# Dtype matrix
# ---------------------------------------------------------------------------


def test_adapter_only_export_preserves_training_dtype(monkeypatch):
    fp32 = torch.zeros(2, dtype=torch.float32)
    _patch_export_machinery(monkeypatch, collect=MagicMock(return_value={"proj.lora_A.weight": fp32}))

    params, peft_config = export_lora_for_sync(
        _peft_module(),
        model_config=_model_config(),
        base_sync_done=True,
        is_diffusers=True,
        key_prefix="transformer.",
    )

    weights = dict(params)
    assert peft_config == {"r": 8}
    assert weights["transformer.proj.lora_A.weight"].dtype is torch.float32
    torch.testing.assert_close(weights["transformer.proj.lora_A.weight"], fp32)


@pytest.mark.parametrize("is_diffusers", [False, True])
def test_full_weight_export_casts_only_floating_dtensors(monkeypatch, is_diffusers):
    plain = torch.zeros(2, dtype=torch.float32)
    floating = _floating_dtensor()
    integer = _integer_dtensor()
    collect = MagicMock(return_value={"w": plain, "dtf": floating, "dti": integer})
    _patch_export_machinery(monkeypatch, collect=collect)
    if not is_diffusers:
        monkeypatch.setattr(lora_export, "replace_lora_wrapper", lambda name, config: name)

    params, peft_config = export_lora_for_sync(
        _peft_module(),
        model_config=_model_config(),
        base_sync_done=False,
        is_diffusers=is_diffusers,
        key_prefix="transformer." if is_diffusers else "",
    )

    weights = dict(params)
    assert peft_config == {"r": 8}
    assert weights["w" if not is_diffusers else "transformer.w"] is plain
    assert weights["dtf" if not is_diffusers else "transformer.dtf"].dtype is torch.bfloat16
    assert weights["dti" if not is_diffusers else "transformer.dti"].dtype is torch.int64


def test_non_lora_full_weights_cast_policy(monkeypatch):
    plain = torch.tensor([1.5], dtype=torch.float32)
    module = _plain_module()
    monkeypatch.setattr(module, "state_dict", lambda: {"w": plain}, raising=False)
    _patch_export_machinery(monkeypatch)

    params, peft_config = export_lora_for_sync(
        module, model_config=_model_config(), base_sync_done=True, is_diffusers=False
    )

    weights = dict(params)
    assert peft_config is None
    assert weights["w"] is plain  # plain tensors pass through untouched


# ---------------------------------------------------------------------------
# Merged stream
# ---------------------------------------------------------------------------


class _FakeModule:
    """PEFT-wrapped module whose ``state_dict`` aliases live storage, like torch's."""

    def __init__(self):
        self.weight = torch.tensor([1.0])
        # ``peft_config`` is what routes the export onto the merged-LoRA stream.
        self.peft_config = {"default": SimpleNamespace(to_dict=lambda: {"r": 8})}

    def state_dict(self):
        return {"weight": self.weight}


def _merged_context(module, merged_value=2.0, base_value=1.0):
    @contextlib.contextmanager
    def context(actor, backup_adapters):
        assert actor is module
        assert backup_adapters
        module.weight.fill_(merged_value)
        try:
            yield
        finally:
            module.weight.fill_(base_value)

    return context


def test_merged_stream_materializes_inside_context(monkeypatch):
    module = _FakeModule()
    monkeypatch.setattr(lora_export, "merged_lora_context", _merged_context(module, merged_value=2.0))
    _patch_export_machinery(monkeypatch)

    params, peft_config = export_lora_for_sync(
        module,
        model_config=_model_config(merge=True),
        base_sync_done=True,
        is_diffusers=False,
    )

    weights = dict(params)
    assert peft_config is None  # merge mode routes the rollout to a full-weight sync
    torch.testing.assert_close(weights["weight"], torch.tensor([2.0]))  # merged value, not restored base
    torch.testing.assert_close(module.weight, torch.tensor([1.0]))  # actor restored after the stream


def test_merged_stream_keeps_prefix(monkeypatch):
    module = _FakeModule()
    monkeypatch.setattr(lora_export, "merged_lora_context", _merged_context(module))
    _patch_export_machinery(monkeypatch)

    params, _ = export_lora_for_sync(
        module,
        model_config=_model_config(merge=True),
        base_sync_done=True,
        is_diffusers=True,
        key_prefix="transformer.",
    )

    assert set(dict(params)) == {"transformer.weight"}


def test_merged_stream_offload_runs_when_consumed(monkeypatch):
    module = _FakeModule()
    monkeypatch.setattr(lora_export, "merged_lora_context", _merged_context(module))
    _patch_export_machinery(monkeypatch)
    offloads = []

    params, _ = export_lora_for_sync(
        module,
        model_config=_model_config(merge=True),
        base_sync_done=True,
        is_diffusers=False,
        offload_fn=lambda: offloads.append(1),
    )
    assert offloads == []  # merged offload waits for the stream to finish

    list(params)
    assert offloads == [1]


def test_non_merged_offload_runs_before_return(monkeypatch):
    collect = MagicMock(return_value={"proj.lora_A.weight": torch.zeros(2)})
    _patch_export_machinery(monkeypatch, collect=collect)
    offloads = []

    params, _ = export_lora_for_sync(
        _peft_module(),
        model_config=_model_config(),
        base_sync_done=True,
        is_diffusers=True,
        offload_fn=lambda: offloads.append(1),
    )
    assert offloads == [1]  # params are already materialized; offload is safe now

    dict(params)
    assert offloads == [1]


def _lora_model(**lora_kwargs):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    base = torch.nn.Sequential()
    base.add_module("up", torch.nn.Linear(16, 32, bias=False))
    base.add_module("down", torch.nn.Linear(32, 16, bias=False))
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=["up", "down"], **lora_kwargs))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_()  # B starts at zero; a zero delta would hide a missing merge
    return model


def _fail_merged_context(actor, backup_adapters):
    raise AssertionError("the per-tensor path must not merge in place")


def test_fsdp2_merged_stream_matches_peft_merge_without_touching_module(monkeypatch):
    import copy

    model = _lora_model()
    expected = copy.deepcopy(model).merge_and_unload().state_dict()
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    monkeypatch.setattr(lora_export, "fsdp_version", lambda module: 2)
    monkeypatch.setattr(lora_export, "merged_lora_context", _fail_merged_context)
    _patch_export_machinery(monkeypatch)
    offloads = []

    params, peft_config = export_lora_for_sync(
        model,
        model_config=_model_config(merge=True),
        base_sync_done=True,
        is_diffusers=False,
        offload_fn=lambda: offloads.append(1),
    )
    assert offloads == []
    weights = dict(params)

    assert peft_config is None
    assert offloads == [1]
    assert set(weights) == set(expected)
    for name, tensor in expected.items():
        assert torch.equal(weights[name], tensor), name
    for name, param in model.named_parameters():
        assert torch.equal(param, before[name]), name
    assert not any(getattr(module, "merged", False) for module in model.modules())


def test_fsdp2_merged_stream_owns_merged_storage(monkeypatch):
    """A sender may flush after the stream ends; merged tensors must not alias
    live parameters that later in-place writes could overwrite."""
    import copy

    model = _lora_model()
    expected = copy.deepcopy(model).merge_and_unload().state_dict()
    monkeypatch.setattr(lora_export, "fsdp_version", lambda module: 2)
    monkeypatch.setattr(lora_export, "merged_lora_context", _fail_merged_context)
    _patch_export_machinery(monkeypatch)

    params, _ = export_lora_for_sync(
        model, model_config=_model_config(merge=True), base_sync_done=True, is_diffusers=False
    )
    held = list(params)
    live = {param.untyped_storage().data_ptr() for param in model.parameters()}
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

    assert {name for name, _ in held} == set(expected)
    for name, tensor in held:
        assert tensor.untyped_storage().data_ptr() not in live, name
        assert torch.equal(tensor, expected[name]), name


@pytest.mark.parametrize("lora_kwargs", [{"use_dora": True}, {"lora_bias": True}])
def test_fsdp2_merged_stream_falls_back_for_unsupported_lora(monkeypatch, lora_kwargs):
    model = _lora_model(**lora_kwargs)
    merged_context = MagicMock(return_value=contextlib.nullcontext())
    monkeypatch.setattr(lora_export, "fsdp_version", lambda module: 2)
    monkeypatch.setattr(lora_export, "merged_lora_context", merged_context)
    _patch_export_machinery(monkeypatch)

    params, _ = export_lora_for_sync(
        model, model_config=_model_config(merge=True), base_sync_done=True, is_diffusers=False
    )
    dict(params)

    merged_context.assert_called_once()


@pytest.mark.parametrize("target_modules", [["up", "mid"], ["up"]], ids=["lora_lora", "lora_plain"])
def test_fsdp2_merged_stream_merges_shared_base_weight_once(monkeypatch, target_modules):
    """A tied base weight receives every delta in the in-place merge, and all of
    its keys stream that one tensor; a clone per key would miss the other deltas."""
    import copy

    from peft import LoraConfig, get_peft_model

    torch.manual_seed(0)
    base = torch.nn.Sequential()
    base.add_module("up", torch.nn.Linear(16, 16, bias=False))
    base.add_module("mid", torch.nn.Linear(16, 16, bias=False))
    base.mid.weight = base.up.weight
    model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=target_modules))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_()
    expected = copy.deepcopy(model).merge_and_unload().state_dict()
    monkeypatch.setattr(lora_export, "fsdp_version", lambda module: 2)
    monkeypatch.setattr(lora_export, "merged_lora_context", _fail_merged_context)
    _patch_export_machinery(monkeypatch)

    params, _ = export_lora_for_sync(
        model, model_config=_model_config(merge=True), base_sync_done=True, is_diffusers=False
    )
    weights = dict(params)

    assert set(weights) == set(expected) == {"up.weight", "mid.weight"}
    for name, tensor in expected.items():
        assert torch.equal(weights[name], tensor), name
    assert not torch.equal(weights["up.weight"], base.up.weight)


def test_merged_export_rejects_named_adapter(monkeypatch):
    _patch_export_machinery(monkeypatch)

    with pytest.raises(ValueError, match="rollout_adapter='old'"):
        export_lora_for_sync(
            _peft_module(),
            model_config=_model_config(merge=True),
            base_sync_done=True,
            adapter_name="old",
            is_diffusers=False,
        )
    # "default" is what the weight-sync call sites pass and must stay accepted.
    _, peft_config = export_lora_for_sync(
        _peft_module(),
        model_config=_model_config(merge=True),
        base_sync_done=True,
        adapter_name="default",
        is_diffusers=False,
    )
    assert peft_config is None


# ---------------------------------------------------------------------------
# Adapter selection and engine parity
# ---------------------------------------------------------------------------


def test_peft_config_follows_requested_adapter(monkeypatch):
    collect = MagicMock(return_value={"proj.lora_A.weight": torch.zeros(2)})
    _patch_export_machinery(monkeypatch, collect=collect)

    params, peft_config = export_lora_for_sync(
        _peft_module(),
        model_config=_model_config(),
        base_sync_done=True,
        adapter_name="old",
        is_diffusers=False,
    )

    assert peft_config == {"r": 8, "adapter": "old"}
    assert collect.call_args.kwargs["adapter_name"] == "old"


def test_ar_base_export_renames_base_layer(monkeypatch):
    collect = MagicMock(return_value={"proj.weight": torch.zeros(2)})
    _patch_export_machinery(monkeypatch, collect=collect)
    monkeypatch.setattr(lora_export, "replace_lora_wrapper", lambda name, config: f"renamed.{name}")

    params, _ = export_lora_for_sync(
        _peft_module(),
        model_config=_model_config(),
        base_sync_done=False,
        is_diffusers=False,
    )

    # The rollout model is PEFT-wrapped on the AR path: plain leaves must land
    # inside the LoRA-wrapped module.
    assert set(dict(params)) == {"renamed.proj.weight"}


def test_checksum_parity_between_engines(monkeypatch):
    """Both engines export identical tensors for identical adapter state."""
    tensors = {"proj.lora_A.weight": torch.randn(2, 4), "proj.lora_B.weight": torch.randn(4, 2)}
    _patch_export_machinery(monkeypatch, collect=MagicMock(return_value=dict(tensors)))

    diffusers_params, diffusers_peft = export_lora_for_sync(
        _peft_module(), model_config=_model_config(), base_sync_done=True, is_diffusers=True, key_prefix="transformer."
    )
    omni_params, omni_peft = export_lora_for_sync(
        _peft_module(), model_config=_model_config(), base_sync_done=True, is_diffusers=False
    )

    assert diffusers_peft == omni_peft
    diffusers_weights = {name.removeprefix("transformer."): t for name, t in diffusers_params}
    omni_weights = dict(omni_params)
    assert set(diffusers_weights) == set(omni_weights) == set(tensors)
    for name, tensor in tensors.items():
        assert diffusers_weights[name].dtype is omni_weights[name].dtype is tensor.dtype
        torch.testing.assert_close(diffusers_weights[name], omni_weights[name])


def test_collect_receives_diffusers_layer_prefixes(monkeypatch):
    collect = MagicMock(return_value={"a": torch.zeros(1)})
    _patch_export_machinery(monkeypatch, collect=collect)

    export_lora_for_sync(_peft_module(), model_config=_model_config(), base_sync_done=True, is_diffusers=True)
    assert collect.call_args.kwargs["layer_prefixes"] == ["layers."]

    export_lora_for_sync(_peft_module(), model_config=_model_config(), base_sync_done=True, is_diffusers=False)
    assert collect.call_args.kwargs["layer_prefixes"] == ("transformer_blocks.",)


# ---------------------------------------------------------------------------
# Colocated sync ordering (real worker, mocked engine/rollout)
# ---------------------------------------------------------------------------


class _NoopRLInsight:
    @staticmethod
    def trace_state(*args, **kwargs):
        return contextlib.nullcontext()


def _slow_path_worker(*, has_lora: bool, peft_merge: bool, base_sync_done: bool):
    worker = object.__new__(ew.ActorRolloutRefWorker)

    engine = MagicMock(spec=["module", "get_per_tensor_param", "get_lora_peft_config"])
    engine.module = _peft_module() if has_lora else _plain_module()
    # A merged engine's export returns peft_config=None (the rollout gets a
    # plain full-weight update), so the probe cannot discover adapter mode.
    peft_config = {"r": 8} if has_lora and not peft_merge else None
    engine.get_lora_peft_config = MagicMock(return_value=peft_config)

    def _get_per_tensor_param(*, layered_summon=False, base_sync_done=True, adapter_name="default", **kwargs):
        tensors = [("adapter.lora_A.weight", torch.zeros(1))] if base_sync_done else [("base.weight", torch.zeros(1))]
        return (iter(tensors), peft_config)

    engine.get_per_tensor_param = MagicMock(side_effect=_get_per_tensor_param)
    worker.actor = SimpleNamespace(engine=engine)

    rollout = AsyncMock()
    rollout.resume = AsyncMock()
    rollout.update_weights = AsyncMock()
    rollout.sleep_level = 2
    worker.rollout = rollout

    worker.config = SimpleNamespace(rollout=SimpleNamespace(free_cache_engine=False))
    worker.peft_merge = peft_merge
    worker.base_sync_done = base_sync_done
    worker.layered_summon = False
    worker.rollout_adapter = "default"
    worker._rank = 0  # update_weights stamps timings/metrics with f"rank_{self.rank}"
    worker._offload_actor_and_empty_cache = lambda *args, **kwargs: None
    return worker


def _run_naive_update(monkeypatch, worker):
    monkeypatch.setattr(ew, "RLInsightLogger", _NoopRLInsight)
    monkeypatch.setattr(ew, "set_expandable_segments", MagicMock())
    monkeypatch.setattr(ew, "log_gpu_memory_usage", MagicMock())
    asyncio.run(ew.ActorRolloutRefWorker.update_weights(worker, mode="naive", global_steps=3))
    return worker


def test_adapter_mode_first_sync_sends_base_then_adapter(monkeypatch):
    worker = _run_naive_update(monkeypatch, _slow_path_worker(has_lora=True, peft_merge=False, base_sync_done=False))

    engine = worker.actor.engine
    assert engine.get_per_tensor_param.call_count == 2
    probe, base = engine.get_per_tensor_param.call_args_list
    assert probe.kwargs["base_sync_done"] is True
    assert base.kwargs["base_sync_done"] is False
    assert worker.rollout.update_weights.await_count == 2
    first, second = worker.rollout.update_weights.await_args_list
    assert first.kwargs["base_sync_done"] is False  # base weights first
    assert second.kwargs["base_sync_done"] is True  # then adapter deltas
    assert worker.rollout.sleep_level == 1  # adapter mode pins level-1 sleep


def test_merge_mode_sends_single_full_sync(monkeypatch):
    worker = _run_naive_update(monkeypatch, _slow_path_worker(has_lora=True, peft_merge=True, base_sync_done=False))

    engine = worker.actor.engine
    assert engine.get_per_tensor_param.call_count == 1
    worker.rollout.update_weights.assert_awaited_once()
    assert worker.rollout.update_weights.await_args.kwargs["base_sync_done"] is True
    assert worker.rollout.sleep_level == 2  # untouched: merged sync is a plain full update


def test_non_lora_sends_single_sync(monkeypatch):
    worker = _run_naive_update(monkeypatch, _slow_path_worker(has_lora=False, peft_merge=False, base_sync_done=False))

    worker.rollout.update_weights.assert_awaited_once()
    assert worker.rollout.update_weights.await_args.kwargs["base_sync_done"] is True
    assert worker.rollout.sleep_level == 2
