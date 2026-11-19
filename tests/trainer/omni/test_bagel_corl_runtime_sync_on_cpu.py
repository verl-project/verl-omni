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
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


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


def test_requested_policy_version_uses_tq_request_step_only():
    tree = ast.parse(
        (ROOT / "verl_omni/pipelines/bagel_flow_grpo/agentic_und_gen/agent_loop.py").read_text(encoding="utf-8")
    )
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "requested_policy_version" for t in n.targets)
    )
    scope = {"sampling_params": {}, "kwargs": {"global_steps": 26, "policy_version": 4}}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "policy_version", "exec"),
        scope,
    )
    assert scope["requested_policy_version"] == 26


@pytest.mark.parametrize("stale", [False, True])
def test_publication_verifies_multiple_sentinels_and_sleeps_on_failure(monkeypatch, stale):
    import verl_omni.workers.rollout.vllm_rollout.weight_fingerprint  # noqa: F401
    from verl_omni.trainer.omni.bagel_corl_und_ar_runtime import publish_und_ar_weights

    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=lambda result: result))
    events = []
    names = (
        "language_model.model.layers.0.self_attn.o_proj.weight",
        "language_model.lm_head.weight",
    )
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

    replicas = [SimpleNamespace(servers=[server], replica_rank=0)]
    publish = lambda: publish_und_ar_weights(
        replicas,
        actor_rollout_wg=SimpleNamespace(update_bagel_und_ar_weights=update),
        receiver_ranks=[0],
        check_names=names,
        global_steps=26,
    )
    if stale:
        with pytest.raises(RuntimeError, match="pushed weights"):
            publish()
    else:
        num_tensors, fingerprint, _seconds = publish()
        assert num_tensors == 50
        assert fingerprint == 1.25
    assert set(names).issubset(events)
    assert events[-1] == "sleep"


def test_ar_builder_disables_engine_lora_and_routes_to_thinker(tmp_path):
    from omegaconf import OmegaConf

    from verl_omni.trainer.omni.bagel_corl_und_ar import build_und_ar_entrypoint_config

    fixture_path = ROOT / "tests/trainer/omni/test_bagel_corl_und_ar_weights_on_cpu.py"
    config = load_function(fixture_path, "_entry_config", {"OmegaConf": OmegaConf})()
    deploy = tmp_path / "deploy.yaml"
    deploy.write_text(
        "stages:\n  - stage_id: 0\n    devices: '0'\n    max_model_len: 16384\n  - stage_id: 1\n    devices: '1'\n",
        encoding="utf-8",
    )
    config.actor_rollout_ref.rollout.agent.und_deploy_config = str(deploy)
    result = build_und_ar_entrypoint_config(config, load_format="auto", policy_stage_id=0, hf_config_path="llm.json")
    assert result.actor_rollout_ref.model.lora_rank == 0
    assert result.actor_rollout_ref.model.hf_config_path == "llm.json"
    assert result.actor_rollout_ref.rollout.engine_kwargs.vllm_omni.weight_sync_stage_ids == [0]
    assert result.actor_rollout_ref.rollout.engine_kwargs.vllm_omni.stage_overrides == {
        "0": {"tensor_parallel_size": 1},
        "1": {"tensor_parallel_size": 1},
    }
    assert result.actor_rollout_ref.rollout.max_model_len == 16384
    assert result.actor_rollout_ref.rollout.load_format == "auto"
    assert config.actor_rollout_ref.rollout.load_format == "dummy"


def test_trainer_rollout_gap_preserves_distinct_update_and_reference_metrics():
    from verl_omni.trainer.omni.bagel_corl_metrics import publish_lane_metrics

    metrics = {
        "actor/und/update_kl": 0.03,
        "actor/und/ref_kl": 0.7,
        "actor/gen/update_kl": 0.002,
        "rollout_corr/kl": 0.19,
    }
    publish_lane_metrics(metrics)
    assert metrics["und/update_kl"] == 0.03
    assert metrics["und/ref_kl"] == 0.7
    assert metrics["gen/update_kl"] == 0.002
    assert metrics["und/rollout_actor_logp_gap"] == 0.19
