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
"""CPU-only checks for the named reward configuration preflight."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl_omni.workers.config import reward as reward_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "verl_omni/trainer/config"
SCRIPT = ROOT / "tools/check_named_reward_config.py"
SPEC = importlib.util.spec_from_file_location("check_named_reward_config", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)
_inspect_named_rewards = CLI._inspect_named_rewards
main = CLI.main
SCORER = "pkg://does.not.exist:missing_scorer"
EXECUTOR = "does.not.exist:MissingModel"


def _native(name: str, *, resource: str = "cpu") -> list[str]:
    placement = [f"++reward.models.{name}.placement.resource={resource}"]
    if resource == "cpu":
        placement.append(f"++reward.models.{name}.placement.num_replicas=1")
    else:
        placement.append(f"++reward.models.{name}.placement.devices=[0]")
    return [
        f"++reward.models.{name}.backend=native",
        *placement,
        f"++reward.models.{name}.executor.model={EXECUTOR}",
    ]


def _engine(name: str) -> list[str]:
    return [f"++reward.models.{name}.backend=engine", f"++reward.models.{name}.replicas=2"]


def _term(name: str, model: str | None = None) -> list[str]:
    overrides = [
        f"++reward.reward_functions.{name}.path={SCORER}",
        f"++reward.reward_functions.{name}.name=compute_score",
    ]
    if model is not None:
        overrides.append(f"++reward.reward_functions.{name}.model={model}")
    return overrides


def _argv(config_name: str = "diffusion_trainer", *overrides: str) -> list[str]:
    return ["--config-dir", str(CONFIG_DIR), "--config-name", config_name, *overrides]


@pytest.mark.parametrize("config_name", ["diffusion_trainer", "omni_trainer"])
def test_cli_reports_ordered_native_and_engine_bindings_without_importing_targets(config_name, capsys):
    overrides = [
        *_native("cpu"),
        *_native("gpu", resource="accelerator"),
        *_engine("engine"),
        *_term("scored", "cpu"),
        *_term("gpu"),
        *_term("engine"),
    ]
    assert main(_argv(config_name, *overrides)) == 0
    captured = capsys.readouterr()
    assert "Named reward configuration error" not in captured.err
    assert json.loads(captured.out) == {
        "scope": "named_reward_schema_and_references",
        "models": [
            {"name": "cpu", "backend": "native"},
            {"name": "gpu", "backend": "native"},
            {"name": "engine", "backend": "engine"},
        ],
        "named_terms": [
            {"name": "scored", "model": "cpu"},
            {"name": "gpu", "model": "gpu"},
            {"name": "engine", "model": "engine"},
        ],
        "requires_reward_role": True,
    }
    assert SCORER not in captured.out and EXECUTOR not in captured.out


@pytest.mark.parametrize("whole_sections", [False, True])
def test_helper_resolves_native_placement_without_mutating_composed_configuration(monkeypatch, whole_sections):
    monkeypatch.delenv("WORKER_ONLY_TOKEN_7321", raising=False)
    if whole_sections:
        section_overrides = [
            "++reward.cpu_placement.resource=cpu",
            "++reward.cpu_placement.num_replicas=${reward.reward_model.nnodes}",
            "++reward.cpu_placement.cpus_per_worker=${reward.num_workers}",
            "++reward.worker_kwargs.device=${reward.cpu_placement.resource}",
            "++reward.worker_kwargs.api_key=${oc.env:WORKER_ONLY_TOKEN_7321}",
            f"++reward.native_executor.model={EXECUTOR}",
            "++reward.native_executor.kwargs=${reward.worker_kwargs}",
            "++reward.models.local.placement=${reward.cpu_placement}",
            "++reward.models.local.executor=${reward.native_executor}",
        ]
    else:
        section_overrides = [
            "++reward.models.local.placement.resource=cpu",
            "++reward.models.local.placement.num_replicas=${reward.reward_model.nnodes}",
            "++reward.models.local.placement.cpus_per_worker=${reward.num_workers}",
            f"++reward.models.local.executor.model={EXECUTOR}",
            "++reward.models.local.executor.kwargs.device=${reward.models.local.placement.resource}",
            "++reward.models.local.executor.kwargs.api_key=${oc.env:WORKER_ONLY_TOKEN_7321}",
        ]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        config = compose(
            config_name="diffusion_trainer",
            overrides=[
                "reward.reward_model.nnodes=3",
                "reward.num_workers=2",
                "++reward.models.local.backend=native",
                *section_overrides,
                *_term("local"),
                "++reward.reward_functions.local.weight=0.7",
            ],
        )
    parsed = []
    original_parse = reward_config.parse_reward_model_config

    def capture_parse(name, value):
        parsed.append((value, original_parse(name, value)))
        return parsed[-1][1]

    monkeypatch.setattr(reward_config, "parse_reward_model_config", capture_parse)
    before = OmegaConf.to_container(config, resolve=False)
    report = _inspect_named_rewards(config)
    assert OmegaConf.to_container(config, resolve=False) == before
    assert parsed[0][0]["placement"]["num_replicas"] == 3
    assert parsed[0][0]["placement"]["cpus_per_worker"] == 2
    assert parsed[0][1].placement.num_replicas == 3
    assert parsed[0][1].placement.cpus_per_worker == 2
    assert parsed[0][0]["executor"]["kwargs"]["device"] == "cpu"
    assert parsed[0][1].executor.kwargs["device"] == "cpu"
    assert parsed[0][1].executor.kwargs["api_key"] == "${oc.env:WORKER_ONLY_TOKEN_7321}"
    assert report["requires_reward_role"] is False
    assert report["named_terms"] == [{"name": "local", "model": "local"}]
    assert "WORKER_ONLY_TOKEN_7321" not in json.dumps(report)


@pytest.mark.parametrize("replicas", ["3", "${reward.reward_model.nnodes}"])
@pytest.mark.parametrize("whole_rollout", [False, True])
def test_cli_parses_literal_and_parent_interpolated_engine_replicas(replicas, whole_rollout, monkeypatch, capsys):
    monkeypatch.delenv("WORKER_ONLY_TOKEN_7321", raising=False)
    parsed = []
    original_parse = reward_config.parse_reward_model_config

    def capture_parse(name, value):
        parsed.append((value, original_parse(name, value)))
        return parsed[-1][1]

    monkeypatch.setattr(reward_config, "parse_reward_model_config", capture_parse)
    overrides = [
        "reward.reward_model.nnodes=3",
        "++reward.models.engine.backend=engine",
        f"++reward.models.engine.replicas={replicas}",
        *(
            [
                "++reward.base_rollout.worker_only=${oc.env:WORKER_ONLY_TOKEN_7321}",
                "++reward.models.engine.rollout=${reward.base_rollout}",
            ]
            if whole_rollout
            else ["++reward.models.engine.rollout.worker_only=${oc.env:WORKER_ONLY_TOKEN_7321}"]
        ),
        *_term("engine"),
    ]
    assert main(_argv("diffusion_trainer", *overrides)) == 0
    assert parsed[0][0]["replicas"] == 3
    assert isinstance(parsed[0][0]["replicas"], int)
    assert parsed[0][1].replicas == 3
    assert parsed[0][1].rollout["worker_only"] == "${oc.env:WORKER_ONLY_TOKEN_7321}"
    captured = capsys.readouterr()
    assert json.loads(captured.out)["models"] == [{"name": "engine", "backend": "engine"}]
    assert "WORKER_ONLY_TOKEN_7321" not in captured.out


def test_unrelated_broken_interpolation_is_outside_named_reward_scope(capsys):
    overrides = [
        *_native("local"),
        *_term("local"),
        "++reward.custom_reward_function.unrelated=${missing.private_value}",
    ]
    assert main(_argv("diffusion_trainer", *overrides)) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["models"] == [{"name": "local", "backend": "native"}]
    assert "private_value" not in captured.out


def test_broken_named_model_interpolation_fails_without_echoing_value(capsys):
    secret = "PRIVATE_INTERPOLATION_7321"
    overrides = [
        "++reward.models.engine.backend=engine",
        f"++reward.models.engine.replicas=${{missing.{secret}}}",
        *_term("engine"),
    ]
    assert main(_argv("diffusion_trainer", *overrides)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert secret not in captured.err
    assert "Named reward configuration error" in captured.err
    assert "reward.models" in captured.err


def test_model_and_term_names_preserve_case(capsys):
    overrides = [*_native("Quality"), *_native("quality"), *_term("Quality"), *_term("lower", "quality")]
    assert main(_argv("diffusion_trainer", *overrides)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["models"] == [
        {"name": "Quality", "backend": "native"},
        {"name": "quality", "backend": "native"},
    ]
    assert report["named_terms"] == [
        {"name": "Quality", "model": "Quality"},
        {"name": "lower", "model": "quality"},
    ]
    assert main(_argv("diffusion_trainer", *_native("Quality"), *_term("bad", "QUALITY"))) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "model references and path/name" in captured.err


@pytest.mark.parametrize("overrides", [[], ["++reward.models=null", *_term("unbound")]])
def test_empty_named_models_is_valid_and_scoped(overrides, capsys):
    assert main(_argv("diffusion_trainer", *overrides)) == 0
    captured = capsys.readouterr()
    assert "Named reward configuration error" not in captured.err
    assert json.loads(captured.out) == {
        "scope": "named_reward_schema_and_references",
        "models": [],
        "named_terms": [],
        "requires_reward_role": False,
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ([*_term("unknown", "absent")], "model references and path/name"),
        ([*_native("m"), "++reward.reward_functions.m.model=m"], "model references and path/name"),
        ([*_native("m"), "++reward.reward_functions.m.path=pkg://missing"], "model references and path/name"),
        (["++reward.models.m.backend=bad"], "check backend, placement"),
        ([*_native("m"), "++reward.models.m.placment.devices=[0]"], "supported fields"),
        ([*_native("m"), "++reward.models.m.unknown=${missing.private_value}"], "supported fields"),
        ([*_native("m"), "++reward.models.m.placement.devices=[0]"], "check backend, placement"),
        (["++reward.models.m.backend=native", "++reward.models.m.placement=bad"], "must be a mapping"),
        (["++reward.models.m=bad"], "must be a mapping"),
        (["++reward.reward_functions.m=bad"], "must be a mapping"),
        (["++reward.reward_functions.m=null"], "must be a mapping"),
        ([*_native("m"), "reward.reward_model.enable=true"], "cannot be combined"),
    ],
)
def test_config_errors_exit_two_without_json(overrides, message, capsys):
    assert main(_argv("diffusion_trainer", *overrides)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert message in captured.err


def test_standalone_yaml_does_not_gain_missing_trainer_defaults(tmp_path, capsys):
    (tmp_path / "bare.yaml").write_text("reward:\n  models: {}\n  reward_functions: {}\n", encoding="utf-8")
    assert main(["--config-dir", str(tmp_path), "--config-name", "bare"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "reward_model" in captured.err


def test_malformed_override_does_not_echo_opaque_value(capsys):
    secret = "UNIQUE_TEST_SECRET_7321"
    assert main(_argv("diffusion_trainer", f"++reward.models.m.backend=[{secret}")) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert secret not in captured.err
    assert "override keys or syntax" in captured.err


@pytest.mark.parametrize(
    ("overrides", "sensitive_value"),
    [
        (["++reward.models.m.backend=PRIVATE_BACKEND_7321"], "PRIVATE_BACKEND_7321"),
        ([*_native("m"), "++reward.models.m.executor.kwargs.device=cuda:3317"], "cuda:3317"),
    ],
)
def test_valid_syntax_schema_error_does_not_echo_sensitive_value(overrides, sensitive_value, capsys):
    assert main(_argv("diffusion_trainer", *overrides)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert sensitive_value not in captured.err
    assert "reward.models: check backend, placement, executor" in captured.err


def test_real_script_invocation_prints_only_json_and_exits_two_on_bad_reference(monkeypatch):
    monkeypatch.delenv("WORKER_ONLY_TOKEN_7321", raising=False)
    good = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            *_argv(
                "diffusion_trainer",
                "reward.reward_model.nnodes=3",
                "++reward.base_rollout.worker_only=${oc.env:WORKER_ONLY_TOKEN_7321}",
                "++reward.models.m.backend=engine",
                "++reward.models.m.replicas=${reward.reward_model.nnodes}",
                "++reward.models.m.rollout=${reward.base_rollout}",
                *_term("m"),
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert good.returncode == 0, good.stderr
    assert json.loads(good.stdout)["models"] == [{"name": "m", "backend": "engine"}]
    assert good.stdout.count("\n") == 1
    assert "WORKER_ONLY_TOKEN_7321" not in good.stdout
    bad = subprocess.run(
        [sys.executable, str(SCRIPT), *_argv("diffusion_trainer", *_term("m", "missing"))],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode == 2
    assert bad.stdout == ""
    assert "model references and path/name" in bad.stderr


@pytest.mark.parametrize("columns", [40, 80, 120])
def test_help_describes_scope_and_needs_no_cli_config(capsys, monkeypatch, columns):
    monkeypatch.setenv("COLUMNS", str(columns))
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    help_text = " ".join(captured.out.split())
    assert "not import availability or training readiness" in help_text
    assert captured.err == ""


@pytest.mark.parametrize("columns", [40, 80, 120])
def test_help_subprocess_runs_without_site_packages(monkeypatch, columns):
    monkeypatch.setenv("COLUMNS", str(columns))
    result = subprocess.run(
        [sys.executable, "-S", str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    help_text = " ".join(result.stdout.split())
    assert "not import availability or training readiness" in help_text
    assert result.stderr == ""
