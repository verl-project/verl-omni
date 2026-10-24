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

"""Execute runtime methods without importing unavailable CUDA serving packages.

Ray/CUDA are replaced only at process boundaries. This is not serving parity.
"""

import ast
import asyncio
import contextlib
import importlib.util
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
ROLLOUT = ROOT / "verl_omni/workers/rollout/vllm_rollout"


def load_file(path):
    assert path.is_file(), f"missing runtime helper: {path.name}"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_function(path, name, namespace=None, class_name=None):
    nodes = ast.parse(path.read_text(encoding="utf-8")).body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    matches = [n for n in nodes if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == name]
    assert matches, f"missing runtime method: {name}"
    node = matches[0]
    node.decorator_list = []
    scope = {"Any": object, **(namespace or {})}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


def test_fingerprint_detects_cancelling_mutations_and_dtype_roundtrip():
    fp = load_file(ROLLOUT / "weight_fingerprint.py")
    tensor = torch.linspace(-0.5, 0.5, 1000)
    changed = tensor.clone()
    changed[13] += 0.25
    changed[71] -= 0.25
    assert not fp.fingerprints_match(fp.weight_fingerprint(tensor), fp.weight_fingerprint(changed))
    for dtype in (torch.float16, torch.bfloat16):
        assert fp.fingerprints_match(fp.weight_fingerprint(tensor), fp.weight_fingerprint(tensor.to(dtype)))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_fingerprints_fail_closed(value):
    fp = load_file(ROLLOUT / "weight_fingerprint.py")
    with pytest.raises(ValueError, match="finite"):
        fp.weight_fingerprint(torch.tensor([value]))
    assert not fp.fingerprints_match(value, value)


@pytest.mark.parametrize(
    "actor,engine",
    [
        ("transformer.layers.0.self_attn.o_proj.weight", "language_model.model.layers.0.self_attn.o_proj.weight"),
        ("transformer.lm_head.weight", "language_model.lm_head.weight"),
        ("transformer.time_embedder.weight", "time_embedder.weight"),
    ],
)
def test_und_ar_names_match_checkpoint_layout(actor, engine):
    route = load_function(
        ROOT / "verl_omni/pipelines/bagel_flow_grpo/bagel_corl.py",
        "route_actor_weight_for_und_ar",
        {"_UND_AR_TOP_LEVEL_PREFIXES": ("time_embedder.", "vae2llm.", "llm2vae.", "latent_pos_embed.")},
    )
    assert route(actor) == engine
    with pytest.raises(ValueError):
        route("layers.0.weight")


def test_policy_version_uses_tq_global_steps():
    tree = ast.parse((ROOT / "verl_omni/agent_loop/bagel_corl.py").read_text(encoding="utf-8"))
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "policy_version" for t in n.targets)
    )
    scope = {"sampling_params": {}, "kwargs": {"global_steps": 26, "policy_version": 4}}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "policy_version", "exec"), scope)
    assert scope["policy_version"] == 26


def test_thinker_only_receives_publication_and_checksum_rpcs():
    select = load_function(ROLLOUT / "vllm_omni_ar_strategy.py", "collective_rpc_stage_ids", class_name="ARStrategy")
    strategy = SimpleNamespace(_weight_sync_stage_ids=[0])
    for method in ("update_weights_from_ipc", "policy_param_checksum"):
        assert select(strategy, method) == [0]
    assert select(strategy, "unrelated") is None


def test_ipc_namespace_is_user_specific_and_idempotent(monkeypatch):
    zmq = load_file(ROLLOUT / "zmq_utils.py")
    monkeypatch.setattr(zmq.os, "getuid", lambda: 42, raising=False)
    handle = zmq.colocate_zmq_handle("job", 0, 1)
    assert handle == "ipc:///tmp/rl-colocate-zmq-u42-job-replica-0-rank-1.sock"
    assert zmq.namespace_colocate_zmq_handle(handle) == handle
    assert zmq.namespace_colocate_zmq_handle("ipc:///tmp/rl-colocate-zmq-job-replica-0-rank-1.sock") == handle
    assert zmq.namespace_colocate_zmq_handle("tcp://localhost:1") == "tcp://localhost:1"


@pytest.mark.parametrize("stale", [False, True])
def test_publication_verifies_multiple_sentinels_and_sleeps_on_failure(monkeypatch, stale):
    fp = load_file(ROLLOUT / "weight_fingerprint.py")
    monkeypatch.setitem(sys.modules, "verl_omni.workers.rollout.vllm_rollout.weight_fingerprint", fp)
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=lambda result: result))
    path = ROOT / "verl_omni/trainer/omni/bagel_corl_trainer.py"
    flatten = load_function(path, "_flatten_rpc_results")
    publish = load_function(
        path,
        "_publish_und_ar_weights",
        {"_flatten_rpc_results": flatten, "logger": logging.getLogger(__name__)},
        "OmniBagelCoRLTrainerSync",
    )
    events = []
    names = ("language_model.model.layers.0.self_attn.o_proj.weight", "language_model.lm_head.weight")
    expected = dict(zip(names, (1.25, 2.5), strict=True))

    def rpc(method, kwargs):
        assert method == "policy_param_checksum"
        name = kwargs["name"]
        events.append(name)
        return [[expected[name] + (1 if stale and name == names[1] else 0)]]

    remote = lambda fn: SimpleNamespace(remote=fn)
    server = SimpleNamespace(
        wake_up=remote(lambda **kw: events.append("wake")),
        sleep=remote(lambda: events.append("sleep")),
        collective_rpc=remote(rpc),
    )

    def update(**kwargs):
        assert kwargs["receiver_ranks"] == [0]
        return [{"num_tensors": 50, "checksum": 1.25, "checksums": expected}]

    trainer = SimpleNamespace(
        und_rollout_replicas=[SimpleNamespace(servers=[server], replica_rank=0)],
        _und_stage_devices=lambda _: {0: [0], 1: [1]},
        _und_deploy_config_path=lambda: "deploy",
        UND_AR_POLICY_STAGE_ID=0,
        UND_AR_SYNC_CHECK_PARAM=names[0],
        UND_AR_SYNC_CHECK_PARAMS=names,
        actor_rollout_wg=SimpleNamespace(update_bagel_und_ar_weights=update),
        global_steps=26,
    )
    if stale:
        with pytest.raises(RuntimeError, match="pushed weights"):
            publish(trainer)
    else:
        publish(trainer)
        assert trainer._pending_sync_metrics["bagel_corl/und_sync/num_tensors"] == 50
    assert set(names).issubset(events)
    assert events[-1] == "sleep"


@pytest.mark.parametrize("rank,unmerged", [(0, False), (1, False), (0, True)])
def test_export_requires_merged_weights_drains_all_ranks_and_routes_head(monkeypatch, rank, unmerged):
    fp = load_file(ROLLOUT / "weight_fingerprint.py")
    route = load_function(
        ROOT / "verl_omni/pipelines/bagel_flow_grpo/bagel_corl.py",
        "route_actor_weight_for_und_ar",
        {"_UND_AR_TOP_LEVEL_PREFIXES": ()},
    )
    zmq = load_file(ROLLOUT / "zmq_utils.py")
    monkeypatch.setattr(zmq.os, "getuid", lambda: 42, raising=False)
    monkeypatch.setitem(sys.modules, "verl.utils.device", SimpleNamespace(is_support_ipc=lambda: True))
    monkeypatch.setitem(
        sys.modules,
        "verl_omni.pipelines.bagel_flow_grpo.bagel_corl",
        SimpleNamespace(route_actor_weight_for_und_ar=route),
    )
    monkeypatch.setitem(sys.modules, "verl_omni.workers.rollout.vllm_rollout.weight_fingerprint", fp)
    monkeypatch.setitem(
        sys.modules, "ray", SimpleNamespace(get_runtime_context=lambda: SimpleNamespace(get_job_id=lambda: "job"))
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    events, sent = [], []

    async def complete(*args, **kwargs):
        events.append((args, kwargs))

    remote = SimpleNamespace(remote=complete)

    class Sender:
        def __init__(self, **kwargs):
            assert kwargs["zmq_handle"].endswith("replica-0-rank-0-update-und-ar-step-26-seq-1.sock")

        async def async_send_weights(self, weights):
            sent.extend(weights)

    def parameters():
        events.append("gather")
        yield "transformer.lm_head.weight", torch.tensor([2.0])
        events.append("drained")

    def export(**kwargs):
        assert kwargs == {"layered_summon": True, "base_sync_done": True, "adapter_name": "default"}
        return parameters(), object() if unmerged else None

    worker = SimpleNamespace(
        actor=SimpleNamespace(engine=SimpleNamespace(get_per_tensor_param=export)),
        layered_summon=True,
        rollout_adapter="default",
        rank=rank,
        _offload_actor_and_empty_cache=lambda: events.append("offload"),
        config=SimpleNamespace(
            rollout=SimpleNamespace(checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=16))
        ),
    )
    export_method = load_function(
        ROOT / "verl_omni/workers/engine_workers.py",
        "update_bagel_und_ar_weights",
        {
            "torch": torch,
            "set_expandable_segments": lambda enabled: events.append(enabled),
            "BucketedWeightSender": Sender,
            "colocate_zmq_handle": zmq.colocate_zmq_handle,
            "make_update_zmq_id": zmq.make_update_zmq_id,
            "make_update_zmq_handle": zmq.make_update_zmq_handle,
            "RLInsightLogger": SimpleNamespace(trace_state=lambda *a, **kw: contextlib.nullcontext()),
        },
        "ActorRolloutRefWorker",
    )
    call = export_method(
        worker,
        SimpleNamespace(collective_rpc=remote, clear_kv_cache=remote, set_global_steps=remote),
        0,
        [0],
        checksum_name="language_model.lm_head.weight",
        global_steps=26,
    )
    if unmerged:
        with pytest.raises(ValueError, match="merged full weights"):
            asyncio.run(call)
    else:
        result = asyncio.run(call)
        assert "drained" in events
        if rank == 0:
            assert sent[0][0] == "language_model.lm_head.weight"
            assert result["checksums"]["language_model.lm_head.weight"] is not None
            assert result["num_tensors"] == 1
        else:
            assert sent == [] and result is None
    assert events[-2:] == ["offload", True]


def test_worker_checksum_reads_real_parameter_and_handles_missing_stage(monkeypatch):
    fp = load_file(ROLLOUT / "weight_fingerprint.py")
    monkeypatch.setitem(sys.modules, "verl_omni.workers.rollout.vllm_rollout.weight_fingerprint", fp)
    checksum = load_function(
        ROLLOUT / "utils.py", "policy_param_checksum", class_name="vLLMOmniColocateWorkerExtension"
    )
    model = torch.nn.Linear(2, 2)
    worker = SimpleNamespace(_get_standard_weight_model_and_config=lambda: (model, None))
    first = checksum(worker, "weight")
    with torch.no_grad():
        model.weight[0, 0] += 1
    assert not fp.fingerprints_match(first, checksum(worker, "weight"))
    assert checksum(worker, "absent") is None
    worker._get_standard_weight_model_and_config = lambda: None
    assert checksum(worker, "weight") is None


def test_ar_builder_disables_engine_lora_and_routes_to_thinker(tmp_path):
    from omegaconf import OmegaConf

    path = ROOT / "verl_omni/trainer/omni/bagel_corl_trainer.py"
    fixture_path = ROOT / "tests/trainer/omni/test_bagel_corl_und_ar_weights_on_cpu.py"
    config = load_function(fixture_path, "_entry_config", {"OmegaConf": OmegaConf})()
    deploy = tmp_path / "deploy.yaml"
    deploy.write_text(
        "stages:\n  - stage_id: 0\n    devices: '0'\n    max_model_len: 16384\n  - stage_id: 1\n    devices: '1'\n",
        encoding="utf-8",
    )
    builder = load_function(
        path,
        "_build_und_ar_entrypoint_config",
        {
            "_actor_merges_lora": load_function(path, "_actor_merges_lora"),
        },
        "OmniBagelCoRLTrainerSync",
    )
    trainer = SimpleNamespace(
        config=config,
        UND_AR_LOAD_FORMAT="auto",
        UND_AR_POLICY_STAGE_ID=0,
        _und_deploy_config_path=lambda: str(deploy),
        _resolve_und_hf_config_path=lambda *args: "llm.json",
        _und_ar_stage_ids=lambda _: (0, 1),
    )
    max_len = load_function(
        path, "_und_stage_max_model_len", {"OmegaConf": OmegaConf, "os": os}, "OmniBagelCoRLTrainerSync"
    )
    trainer._und_stage_max_model_len = max_len
    result = builder(trainer)
    assert result.actor_rollout_ref.model.lora_rank == 0
    assert result.actor_rollout_ref.rollout.engine_kwargs.vllm_omni.weight_sync_stage_ids == [0]
    assert result.actor_rollout_ref.rollout.load_format == "auto"
    assert config.actor_rollout_ref.rollout.load_format == "dummy"


def test_server_returns_checksum_results_from_selected_stage():
    async def rpc(**kwargs):
        assert kwargs["stage_ids"] == [0]
        return [[1.25]]

    method = load_function(ROLLOUT / "vllm_omni_async_server.py", "collective_rpc", class_name="vLLMOmniHttpServer")
    server = SimpleNamespace(
        engine=SimpleNamespace(collective_rpc=rpc),
        _generate_strategy=SimpleNamespace(collective_rpc_stage_ids=lambda _: [0]),
    )
    assert asyncio.run(method(server, "policy_param_checksum", kwargs={"name": "head"})) == [[1.25]]
