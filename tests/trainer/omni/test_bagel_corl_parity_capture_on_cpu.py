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

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def load():
    spec = importlib.util.spec_from_file_location("capture", ROOT / "verl_omni/trainer/omni/bagel_corl_parity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture(tmp_path):
    return load().FrozenPolicyCapture(
        tmp_path,
        dict(
            policy_version=2,
            calibration_id="host-calibration",
            logprob_convention="raw_policy",
            kernel_config={"und": {"dtype": "float32"}, "gen": {"dtype": "float32"}},
        ),
        manual_replica_label="physical-engine-0",
    )


def und():
    return dict(
        episode_uid="ep",
        policy_version=2,
        input_ids=torch.tensor([1, 2, 3, 4]),
        responses=torch.tensor([2, 3, 4]),
        response_mask=torch.tensor([1, 0, 1]),
        position_ids=torch.arange(4),
        rollout_log_probs=torch.tensor([-0.1, 0.0, -0.3]),
        old_log_probs=torch.tensor([-0.11, 0.0, -0.31]),
    )


def test_disabled_capture_needs_no_provenance(monkeypatch):
    monkeypatch.delenv("BAGEL_CORL_PARITY_CAPTURE_DIR", raising=False)
    assert load().FrozenPolicyCapture.from_environment(SimpleNamespace()) is None


def test_exact_environment_spans_and_actor_repeats_are_saved(tmp_path):
    cap = capture(tmp_path)
    row = und()
    cap.write_und([row], [dict(row)], [dict(row)])
    data = json.loads(cap.path.read_text().splitlines()[0])
    assert data["context"]["input_ids"]["values"] == [1, 2, 3, 4]
    assert data["mask"]["values"] == [1, 0, 1]
    assert data["rollout_logp"] == pytest.approx(-0.1)
    assert data["actor_old_logp"] == pytest.approx(-0.11)
    assert data["actor_repeat_logp"] == pytest.approx(-0.11)
    assert data["replica_id"] == "physical-engine-0"
    assert data["replica_identity_source"] == "manual_physical_replica_calibration_label"
    assert not data["complete_replica_coverage_verified"]
    lines = [json.loads(line) for line in cap.path.read_text().splitlines()]
    assert [line["timestep"] for line in lines] == [0, 2]


def test_capture_fails_without_replica_version_or_exact_context(tmp_path):
    cap = capture(tmp_path)
    row = und()
    for key in ("policy_version", "input_ids"):
        bad = dict(row)
        bad.pop(key)
        with pytest.raises(ValueError, match=key):
            cap.write_und([bad], [row], [row])
    cap.manual_replica_label = None
    with pytest.raises(ValueError, match="replica"):
        cap.write_und([row], [row], [row])
    assert not cap.path.exists()


def test_context_mutation_and_stale_version_fail(tmp_path):
    cap = capture(tmp_path)
    row = und()
    changed = dict(row, input_ids=torch.tensor([9, 2, 3, 4]))
    with pytest.raises(ValueError, match="context"):
        cap.write_und([row], [row], [changed])
    with pytest.raises(ValueError, match="version"):
        cap.write_und([dict(row, policy_version=1)], [row], [row])


def test_runtime_method_has_two_gen_calls_and_capture_und_wrapper():
    tree = ast.parse((ROOT / "verl_omni/trainer/omni/bagel_corl_trainer.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OmniBagelCoRLTrainerSync")
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_compute_old_log_prob"
    )
    text = ast.unparse(method)
    assert "FrozenPolicyCapture.from_environment(self)" in text
    assert "capture.recompute_und" in text
    assert "capture.write_gen" in text
    assert text.count("._compute_old_log_prob(proto)") == 2


@pytest.mark.parametrize("with_gen", [False, True])
def test_real_runtime_method_replays_and_restores_first_old(tmp_path, monkeypatch, with_gen):
    import copy
    import logging
    import sys

    module = load()
    cap = module.FrozenPolicyCapture(
        tmp_path,
        dict(
            policy_version=2,
            calibration_id="fixture",
            logprob_convention="raw_policy",
            kernel_config={"und": {"dtype": "float32"}, "gen": {"dtype": "float32"}},
        ),
        manual_replica_label="physical-0",
    )
    monkeypatch.setattr(module.FrozenPolicyCapture, "from_environment", classmethod(lambda cls, trainer: cap))
    monkeypatch.setitem(sys.modules, "verl_omni.trainer.omni.bagel_corl_parity", module)
    raw = und()
    queue = {"row": copy.deepcopy(raw)}

    class QueueData:
        def __init__(self, rows):
            self.rows = rows

        def select(self, *fields):
            return QueueData({key: {field: row[field] for field in fields} for key, row in self.rows.items()})

        def clone(self):
            return copy.deepcopy(self)

    def put(**kwargs):
        for key, row in kwargs["fields"].rows.items():
            queue[key].update(copy.deepcopy(row))

    monkeypatch.setitem(
        sys.modules,
        "transfer_queue",
        SimpleNamespace(kv_batch_get=lambda **kwargs: QueueData(copy.deepcopy(queue)), kv_batch_put=put),
    )

    class Batch:
        keys = ["row"]
        partition_id = "p"
        extra_info = {}

        def __len__(self):
            return 1

    counts = {"und": 0, "gen": 0}

    class Base:
        def _compute_old_log_prob(self, batch, metrics):
            counts["und"] += 1
            queue["row"]["old_log_probs"] = torch.tensor([-0.11 * counts["und"], 0.0, -0.31])
            queue["row"]["entropy"] = torch.zeros(3)
            return batch

    class Proto:
        def __init__(self, batch):
            self.batch = batch
            self.non_tensor_batch = {"prompt_token_ids": [[1, 2]]}

        def __len__(self):
            return 1

    gen = dict(
        episode_uid="ep",
        gen_sample_uid="sample",
        all_latents=torch.tensor([[1.0], [2.0], [3.0]]),
        timesteps=torch.tensor([1.0, 0.5]),
        rollout_log_probs=torch.tensor([-0.3, -0.4]),
        rm_score=0.5,
        gen_group_uid="g",
        prompt_token_ids=[1, 2],
    )
    proto = Proto(
        dict(
            all_latents=gen["all_latents"].unsqueeze(0),
            all_timesteps=gen["timesteps"].unsqueeze(0),
            gen_transition_valid=torch.ones(1, 2),
            old_log_probs=gen["rollout_log_probs"].unsqueeze(0),
            gen_row_id=torch.tensor([0]),
        )
    )

    class Lane:
        def _compute_old_log_prob(self, data):
            counts["gen"] += 1
            # Mutate the worker input to prove replay/capture owns independent copies.
            result = Proto({"old_log_probs": data.batch["old_log_probs"] - 0.01 * counts["gen"]})
            data.batch["all_latents"].add_(99)
            return result

    tree = ast.parse((ROOT / "verl_omni/trainer/omni/bagel_corl_trainer.py").read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OmniBagelCoRLTrainerSync")
    method = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_compute_old_log_prob"
    )
    new_cls = ast.ClassDef(
        name="OmniBagelCoRLTrainerSync",
        bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    namespace = dict(
        Base=Base,
        logger=logging.getLogger("capture-test"),
        _normalize_tq_kv_get_result=lambda data, keys: {key: {"fields": data.rows[key]} for key in keys},
        _bagel_role_histogram=lambda tags: {},
        build_gen_flowgrpo_proto=lambda rows: copy.deepcopy(proto) if rows else None,
    )
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[new_cls], type_ignores=[])), "<trainer-method>", "exec"),
        namespace,
    )
    trainer = namespace["OmniBagelCoRLTrainerSync"]()
    trainer.actor_rollout_wg = object()
    trainer._extra_info = lambda batch: batch.extra_info
    trainer._gen_batch_from_step = lambda batch: [gen] if with_gen else []
    trainer._gen_dispatch_divisor = lambda: 1
    trainer._diffusion_v1_gen_lane = lambda: Lane()
    batch = Batch()
    trainer._compute_old_log_prob(batch, {})
    assert counts == {"und": 2, "gen": 2 if with_gen else 0}
    assert queue["row"]["old_log_probs"][0] == pytest.approx(-0.11)
    rows = [json.loads(line) for line in cap.path.read_text().splitlines()]
    assert {row["lane"] for row in rows} == ({"und", "gen"} if with_gen else {"und"})
    first = next(row for row in rows if row["lane"] == "und")
    assert first["actor_old_logp"] == pytest.approx(-0.11)
    assert first["actor_repeat_logp"] == pytest.approx(-0.22)
    if with_gen:
        first_gen = next(row for row in rows if row["lane"] == "gen")
        assert first_gen["context"]["all_latents"]["values"] == [[1.0], [2.0], [3.0]]
        assert first_gen["rollout_logp"] == pytest.approx(-0.3)
        assert first_gen["actor_old_logp"] == pytest.approx(-0.31)
        assert first_gen["actor_repeat_logp"] == pytest.approx(-0.32)


def test_enabled_capture_requires_manifest_and_rejects_bypass(tmp_path, monkeypatch):
    module = load()
    monkeypatch.setenv("BAGEL_CORL_PARITY_CAPTURE_DIR", str(tmp_path))
    monkeypatch.delenv("BAGEL_CORL_PARITY_MANIFEST", raising=False)
    with pytest.raises(ValueError, match="BAGEL_CORL_PARITY_MANIFEST"):
        module.FrozenPolicyCapture.from_environment(SimpleNamespace())
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            dict(
                policy_version=2,
                calibration_id="fixture",
                logprob_convention="raw_policy",
                kernel_config={"und": {"dtype": "float32"}, "gen": {"dtype": "float32"}},
            )
        )
    )
    monkeypatch.setenv("BAGEL_CORL_PARITY_MANIFEST", str(path))
    trainer = SimpleNamespace(global_steps=1)
    with pytest.raises(ValueError, match="policy_version"):
        module.FrozenPolicyCapture.from_environment(trainer)
    trainer.global_steps = 2
    trainer.config = SimpleNamespace(algorithm={"rollout_correction": {"bypass_mode": True}})
    with pytest.raises(ValueError, match="bypass_mode"):
        module.FrozenPolicyCapture.from_environment(trainer)


def test_recorded_replica_identity_takes_precedence(tmp_path):
    cap = capture(tmp_path)
    row = dict(und(), rollout_replica_id=7)
    cap.write_und([row], [row], [row])
    saved = json.loads(cap.path.read_text().splitlines()[0])
    assert saved["replica_id"] == 7
    assert saved["replica_identity_source"] == "rollout_metadata"
    assert not saved["complete_replica_coverage_verified"]
