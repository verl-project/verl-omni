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
"""CPU coverage for explicit scorer config at launch and manager consumption."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl import DataProto

from verl_omni.reward_loop.reward_manager.multi import MultiVisualRewardManager
from verl_omni.workers.config.reward import REWARD_TERM_METADATA_KEYS, normalize_reward_config

CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "verl_omni" / "trainer" / "config")
SCORER_PATH = "tests/reward_loop/test_reward_config_normalization_on_cpu.py"
EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "examples" / "flowgrpo_trainer" / "sd35" / "sd3_jpeg_scorers.yaml"


def reward_config_probe(solution_image, gain, bias, extra_info):
    """Return the forwarded nested parameters through the real manager."""
    assert solution_image.dtype == torch.uint8
    assert isinstance(extra_info, dict)
    return {"score": gain + bias}


def reward_config_kwargs_probe(solution_image, kwargs):
    """Consume a legacy scorer argument whose name matches the new wrapper."""
    assert solution_image.dtype == torch.uint8
    return {"score": kwargs["gain"]}


def _image_data():
    return DataProto.from_dict(
        tensors={"responses": torch.zeros((1, 3, 16, 16), dtype=torch.uint8)},
        non_tensors={
            "data_source": ["test"],
            "reward_model": [{"ground_truth": "test"}],
            "media_kind": ["image"],
            "extra_info": [{}],
        },
    )


def _compose(name="diffusion_trainer"):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name=name)


def _scorer(**fields):
    return {"path": SCORER_PATH, "name": "reward_config_probe", **fields}


def _manager():
    return {
        "name": "MultiVisualRewardManager",
        "source": "importlib",
        "module": {"path": "pkg://verl_omni.reward_loop.reward_manager"},
    }


def test_example_fragment_composes_with_hydra_base():
    config = OmegaConf.merge(_compose(), OmegaConf.load(EXAMPLE_PATH))
    normalized = normalize_reward_config(config)
    assert normalized.reward.reward_manager.name == "MultiVisualRewardManager"
    assert normalized.reward.reward_functions.jpeg.path == ("verl_omni/utils/reward_score/jpeg_compressibility.py")
    assert normalized.reward.reward_functions.jpeg.name == "compute_score"


@pytest.mark.parametrize("name", ["diffusion_trainer", "omni_trainer"])
def test_hydra_scoring_and_selected_interpolations(name):
    config = _compose(name)
    OmegaConf.update(config, "reward.reward_manager.name", "MultiVisualRewardManager", force_add=True)
    OmegaConf.update(config, "reward.reward_manager.source", "importlib", force_add=True)
    OmegaConf.update(
        config,
        "reward.reward_manager.module.path",
        "pkg://verl_omni.reward_loop.reward_manager",
        force_add=True,
    )
    OmegaConf.update(config, "reward.num_workers", 8, force_add=True)
    OmegaConf.update(
        config,
        "reward.scorers",
        {
            "probe": _scorer(
                weight=1.0, required=True, kwargs={"gain": "${reward.num_workers}", "bias": "${....num_workers}"}
            )
        },
        force_add=True,
    )
    OmegaConf.update(config, "reward.reward_model.enable", False, force_add=True)
    original = OmegaConf.to_container(config.reward.scorers, resolve=False)
    normalized = normalize_reward_config(config)

    assert OmegaConf.to_container(config.reward.scorers, resolve=False) == original
    assert normalized.reward.scorers.probe.kwargs.gain == 8
    assert OmegaConf.is_struct(normalized) == OmegaConf.is_struct(config)
    assert OmegaConf.is_struct(normalized.reward) == OmegaConf.is_struct(config.reward)
    assert list(normalized.reward.reward_functions) == ["probe"]
    assert normalized.reward.reward_functions.probe.gain == 8
    assert normalized.reward.reward_functions.probe.bias == 8
    assert normalized.reward.reward_functions.probe.required is True
    assert normalize_reward_config(normalized) is normalized

    if name == "omni_trainer":
        return
    manager = MultiVisualRewardManager(normalized, MagicMock(), compute_score=None)
    data = DataProto.from_dict(
        tensors={"responses": torch.randint(256, (1, 3, 16, 16), dtype=torch.uint8)},
        non_tensors={
            "data_source": ["test"],
            "reward_model": [{"ground_truth": "test"}],
            "media_kind": ["image"],
            "extra_info": [{}],
        },
    )
    result = manager.loop.run_until_complete(manager.run_single(data))
    assert result["reward_score"] == pytest.approx(16.0)


def test_term_order_metadata_and_null_kwargs_are_preserved():
    config = OmegaConf.create(
        {
            "reward": {
                "scorers": {
                    "first": _scorer(
                        model="chosen",
                        routing_weights={"a": 0.25},
                        independent=True,
                        use_rollout_sampling_params=False,
                        kwargs=None,
                    ),
                    "second": _scorer(weight=2.5, required=False, kwargs={"options": [1, 2]}),
                },
                "reward_functions": {},
                "custom_reward_function": {"path": None},
                "reward_manager": _manager(),
            }
        }
    )
    normalized = normalize_reward_config(config)
    assert list(normalized.reward.reward_functions) == ["first", "second"]
    assert OmegaConf.to_container(normalized.reward.reward_functions.first) == {
        "path": SCORER_PATH,
        "name": "reward_config_probe",
        "model": "chosen",
        "routing_weights": {"a": 0.25},
        "independent": True,
        "use_rollout_sampling_params": False,
    }
    assert normalized.reward.reward_functions.second.options == [1, 2]


def test_manager_consumes_metadata_without_forwarding_it_to_scorer():
    config = OmegaConf.merge(
        _compose(),
        {
            "reward": {
                "scorers": {
                    "probe": _scorer(
                        model=None,
                        weight=1.0,
                        required=True,
                        independent=False,
                        routing_weights={"image": 1.0},
                        use_rollout_sampling_params=False,
                        kwargs={"gain": 3, "bias": 4},
                    )
                },
                "reward_functions": {},
                "reward_manager": _manager(),
            }
        },
    )
    manager = MultiVisualRewardManager(normalize_reward_config(config), MagicMock(), compute_score=None)
    assert manager._sub_rewards[0]["extra_args"] == {"gain": 3, "bias": 4}
    assert manager._sub_rewards[0]["required"] is True
    result = manager.loop.run_until_complete(manager.run_single(_image_data()))
    assert result["reward_score"] == pytest.approx(7.0)


def test_legacy_kwargs_is_still_a_scorer_argument():
    config = OmegaConf.merge(
        _compose(),
        {
            "reward": {
                "scorers": None,
                "reward_manager": _manager(),
            }
        },
    )
    OmegaConf.update(
        config,
        "reward.reward_functions.probe",
        _scorer(name="reward_config_kwargs_probe", kwargs={"gain": 3}),
        force_add=True,
    )
    assert normalize_reward_config(config) is config
    manager = MultiVisualRewardManager(config, MagicMock(), compute_score=None)
    assert manager._sub_rewards[0]["extra_args"] == {"kwargs": {"gain": 3}}
    result = manager.loop.run_until_complete(manager.run_single(_image_data()))
    assert result["reward_score"] == pytest.approx(3.0)


@pytest.mark.parametrize("key", sorted(REWARD_TERM_METADATA_KEYS | {"kwargs"}))
def test_scorer_kwargs_cannot_shadow_metadata_or_itself(key):
    config = OmegaConf.create(
        {
            "reward": {
                "scorers": {"probe": _scorer(kwargs={key: "shadow"})},
                "reward_functions": {},
                "reward_manager": _manager(),
            }
        }
    )
    with pytest.raises(ValueError, match="kwargs contains reserved fields"):
        normalize_reward_config(config)


def test_external_reference_and_matching_dual_projection():
    config = OmegaConf.create(
        {
            "reward": {
                "scorers": {"one": _scorer(kwargs={"gain": 3, "bias": 4})},
                "reward_functions": {},
                "custom_reward_function": {"path": None},
                "reward_manager": _manager(),
            },
            "external": "${reward.scorers.one.kwargs.gain}",
        }
    )
    normalized = normalize_reward_config(config)
    assert OmegaConf.select(normalized, "external") == 3
    assert normalize_reward_config(normalized) is normalized
    same = OmegaConf.create(OmegaConf.to_container(normalized, resolve=False))
    assert normalize_reward_config(same) is same
    same.reward.reward_functions.one.gain = 5
    with pytest.raises(ValueError, match="conflicts"):
        normalize_reward_config(same)


def test_matching_terms_in_different_order_conflict():
    config = OmegaConf.create(
        {
            "reward": {
                "scorers": {"one": _scorer(), "two": _scorer()},
                "reward_functions": {"two": _scorer(), "one": _scorer()},
                "custom_reward_function": {"path": None},
                "reward_manager": _manager(),
            }
        }
    )
    with pytest.raises(ValueError, match="conflicts"):
        normalize_reward_config(config)


@pytest.mark.parametrize("manager_name", ["VisualRewardManager", "naive", None])
def test_explicit_scorers_require_visual_multi_manager(manager_name):
    config = _compose()
    OmegaConf.update(config, "reward.scorers", {"jpeg": _scorer()}, force_add=True)
    if manager_name is None:
        del config.reward.reward_manager
    else:
        config.reward.reward_manager.name = manager_name
    with pytest.raises(ValueError, match="reward.reward_manager.name=MultiVisualRewardManager"):
        normalize_reward_config(config)


def test_matching_projection_cannot_bypass_manager_guard():
    config = OmegaConf.create(
        {
            "reward": {
                "scorers": {"jpeg": _scorer()},
                "reward_functions": {"jpeg": _scorer()},
                "reward_manager": {"name": "naive"},
            }
        }
    )
    with pytest.raises(ValueError, match="reward.reward_manager.name=MultiVisualRewardManager"):
        normalize_reward_config(config)


@pytest.mark.parametrize(
    "manager",
    [
        {
            "name": "MultiVisualRewardManager",
            "source": "register",
            "module": {"path": "pkg://verl_omni.reward_loop.reward_manager"},
        },
        {"name": "MultiVisualRewardManager", "source": "importlib", "module": {"path": "pkg://custom.reward_manager"}},
        {"name": "MultiVisualRewardManager", "source": "importlib", "module": "custom"},
        "MultiVisualRewardManager",
    ],
)
def test_same_name_with_different_loader_is_rejected(manager):
    config = OmegaConf.create(
        {"reward": {"scorers": {"jpeg": _scorer()}, "reward_functions": {}, "reward_manager": manager}}
    )
    with pytest.raises(ValueError, match="reward.reward_manager.module.path=pkg://"):
        normalize_reward_config(config)


def test_legacy_null_and_absent_are_identity_without_resolving_missing():
    legacy = OmegaConf.create(
        {
            "reward": {
                "scorers": None,
                "reward_functions": {"old": {"custom_option": 2}},
                "custom_reward_function": {"path": "old.py", "name": "score"},
                "reward_manager": {"name": "VisualRewardManager"},
            },
            "unused": "???",
        }
    )
    assert normalize_reward_config(legacy) is legacy
    assert legacy.reward.reward_functions.old.custom_option == 2
    del legacy.reward.scorers
    assert normalize_reward_config(legacy) is legacy


@pytest.mark.parametrize(
    ("scorers", "match"),
    [
        ({}, "non-empty mapping"),
        (7, "non-empty mapping"),
        ({"x": None}, "must be a mapping"),
        ({"x": 1}, "must be a mapping"),
        ({"x": _scorer(unknown=True)}, "unknown fields"),
        ({"x": _scorer(kwargs=1)}, "kwargs must be a mapping"),
        ({"x": _scorer(kwargs={"required": True})}, "reserved fields"),
        ({"x": {"name": "score"}}, "path must be"),
        ({"x": {"path": "score.py"}}, "name must be"),
        ({"x": _scorer(name="")}, "name must be"),
    ],
)
def test_invalid_scorers_rejected(scorers, match):
    config = OmegaConf.create(
        {
            "reward": {
                "scorers": scorers,
                "reward_functions": {},
                "custom_reward_function": {"path": None},
                "reward_manager": _manager(),
            }
        }
    )
    with pytest.raises(ValueError, match=match):
        normalize_reward_config(config)


@pytest.mark.parametrize("conflict", ["reward_functions", "custom_reward_function"])
def test_conflicts_rejected(conflict):
    config = OmegaConf.create(
        {
            "reward": {
                "scorers": {"x": _scorer()},
                "reward_functions": {},
                "custom_reward_function": {"path": None},
                "reward_manager": _manager(),
            }
        }
    )
    if conflict == "reward_functions":
        config.reward.reward_functions = {"old": {"path": "old.py"}}
    else:
        config.reward.custom_reward_function.path = "old.py"
    with pytest.raises(ValueError, match="conflicts"):
        normalize_reward_config(config)


@pytest.mark.parametrize(
    "module_name,run_name",
    [("main_diffusion", "run_diffusion"), ("main_diffusion_v1", "run_diffusion_v1"), ("main_omni", "run_omni")],
)
def test_entrypoints_project_before_warning_or_ray(monkeypatch, module_name, run_name):
    from importlib import import_module

    module = import_module(f"verl_omni.trainer.{module_name}")
    config = OmegaConf.create(
        {
            "reward": {
                "scorers": {"x": _scorer(kwargs={"gain": 2, "bias": 3})},
                "reward_functions": {},
                "custom_reward_function": {"path": None},
                "reward_manager": _manager(),
            },
            "trainer": {"use_v1": False},
        }
    )

    class WarningReached(Exception):
        pass

    def inspect_warning(received):
        assert received is not config
        assert received.reward.scorers.x.kwargs.gain == 2
        assert received.reward.reward_functions.x.gain == 2
        raise WarningReached

    monkeypatch.setattr(module, "warn_deprecated_reward_config", inspect_warning)
    monkeypatch.setattr(module.ray, "is_initialized", lambda: pytest.fail("Ray reached before warning"))
    with pytest.raises(WarningReached):
        getattr(module, run_name)(config)


def _fsdp_scorer_launch_config(required, *, model="pickscore"):
    term = _scorer(
        path="pkg://verl_omni.utils.reward_score.pickscore_reward",
        name="compute_score_pickscore_native",
        model=model,
        kwargs={"score_divisor": 13.0},
    )
    if required is not None:
        term["required"] = required
    return OmegaConf.create(
        {
            "trainer": {"use_v1": True, "v1": {"trainer_mode": "sync"}},
            "reward": {
                "reward_manager": _manager(),
                "custom_reward_function": {"path": None},
                "reward_functions": {},
                "scorers": {"probe": term},
                "reward_model": {"enable_resource_pool": True, "nnodes": 1, "n_gpus_per_node": 2},
                "models": {
                    "pickscore": {
                        "backend": "native",
                        "model_path": "immutable-checkpoint",
                        "placement": {"replica_groups": [[0, 1]]},
                        "executor": {"model": "verl_omni.utils.reward_score.pickscore_reward:PickScoreNativeModel"},
                    }
                },
            },
        }
    )


def _stub_fsdp_cli_runtime(monkeypatch):
    from verl_omni.trainer import main_diffusion_v1

    monkeypatch.setattr(main_diffusion_v1, "auto_set_device", lambda config: None)
    monkeypatch.setattr(main_diffusion_v1, "validate_config", lambda config: None)
    monkeypatch.setattr(main_diffusion_v1, "validate_attention_consistency", lambda config: None)
    monkeypatch.setattr(main_diffusion_v1, "is_cuda_available", True)
    monkeypatch.setattr(main_diffusion_v1.ray, "init", lambda **kwargs: pytest.fail("Ray allocated"))
    monkeypatch.setattr(main_diffusion_v1, "run_diffusion_v1", lambda config: pytest.fail("trainer selected"))
    return main_diffusion_v1


@pytest.mark.parametrize("required", [None, False], ids=["omitted", "false"])
def test_fsdp_cli_rejects_optional_explicit_scorer_before_launch(monkeypatch, required):
    config = _fsdp_scorer_launch_config(required)
    module = _stub_fsdp_cli_runtime(monkeypatch)
    with pytest.raises(ValueError, match="Native FSDP term 'probe' requires required=true"):
        module.main.__wrapped__(config)
    assert not config.reward.reward_functions


def test_fsdp_cli_rejects_unknown_explicit_model_before_launch(monkeypatch):
    config = _fsdp_scorer_launch_config(True, model="missing")
    module = _stub_fsdp_cli_runtime(monkeypatch)
    with pytest.raises(ValueError, match="references unknown model 'missing'"):
        module.main.__wrapped__(config)
    assert not config.reward.reward_functions


def test_fsdp_cli_validates_projection_and_dispatches_required_scorer(monkeypatch):
    config = _fsdp_scorer_launch_config(True)
    module = _stub_fsdp_cli_runtime(monkeypatch)
    checked = []

    def inspect_config(received):
        assert received.reward.reward_functions.probe.model == "pickscore"
        assert received.reward.reward_functions.probe.required is True
        checked.append(received)

    def inspect_launch(received):
        assert received is checked[0]
        assert normalize_reward_config(received) is received
        assert received.reward.reward_functions.probe.score_divisor == 13.0

    monkeypatch.setattr(module, "validate_config", inspect_config)
    monkeypatch.setattr(module, "run_diffusion_v1", inspect_launch)
    module.main.__wrapped__(config)
    assert len(checked) == 1
    assert not config.reward.reward_functions
