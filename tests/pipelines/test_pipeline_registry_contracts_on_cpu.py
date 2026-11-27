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
"""Registry-wide L1 contract tests for all pipelines and unseen TQ fields."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PIPELINES_DIR = _REPO_ROOT / "verl_omni" / "pipelines"
_SHARED_SUBPACKAGES = frozenset({"schedulers"})
_STATELESS_BASES = frozenset(
    {
        "DiffusionModelBase",
        "DiffusionI2IModelBase",
        "DiffusionModelARBase",
        "OmniModelBase",
        "OmniRolloutPipelineBase",
    }
)
_OFFLINE_ONLY_DIFFUSION_KEYS = frozenset({("StableDiffusion3Pipeline", "dpo")})
_OPTIONAL_VLLM_OMNI_ROLLOUT_KEYS = frozenset(
    {
        ("MiniMaxH3ImagePipeline", "diffusion_nft"),
        ("MiniMaxH3ImagePipeline", "flow_grpo"),
        ("WanPipeline", "dance_grpo"),
    }
)
_ALLOWED_METHOD_DECORATORS = frozenset(
    {
        "classmethod",
        "staticmethod",
        "property",
        "builtins.classmethod",
        "builtins.staticmethod",
        "builtins.property",
    }
)


def _concrete_pipeline_packages() -> list[str]:
    return sorted(
        entry.name
        for entry in _PIPELINES_DIR.iterdir()
        if entry.is_dir() and (entry / "__init__.py").is_file() and entry.name not in _SHARED_SUBPACKAGES
    )


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else ""
    return ""


def _class_leaf_bases(node: ast.ClassDef) -> set[str]:
    return {_qualified_name(base).rsplit(".", 1)[-1] for base in node.bases if _qualified_name(base)}


def _extract_register_key(decorator: ast.expr, expected_base: str) -> tuple[str, str] | None:
    if not (
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "register"
        and _qualified_name(decorator.func.value).rsplit(".", 1)[-1] == expected_base
    ):
        return None
    if (
        not decorator.args
        or not isinstance(decorator.args[0], ast.Constant)
        or not isinstance(decorator.args[0].value, str)
    ):
        return None
    arch = decorator.args[0].value
    algo = ""
    if (
        len(decorator.args) >= 2
        and isinstance(decorator.args[1], ast.Constant)
        and isinstance(decorator.args[1].value, str)
    ):
        algo = decorator.args[1].value
    else:
        for kw in decorator.keywords:
            if kw.arg == "algorithm" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                algo = kw.value.value
                break
    return (arch, algo) if algo else None


def _ast_register_keys(expected_base: str) -> dict[tuple[str, str], tuple[str, str]]:
    found: dict[tuple[str, str], tuple[str, str]] = {}
    for pkg in _concrete_pipeline_packages():
        for py_file in sorted((_PIPELINES_DIR / pkg).rglob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                for decorator in node.decorator_list:
                    key = _extract_register_key(decorator, expected_base)
                    if key is not None:
                        found[key] = (pkg, node.name)
    return found


def _assert_stateless_class(registry_name: str, key: object, adapter_cls: type) -> None:
    assert "__init__" not in adapter_cls.__dict__, f"{registry_name}{key} defines __init__"
    for attr_name, raw_attr in adapter_cls.__dict__.items():
        if attr_name.startswith("__") and attr_name.endswith("__"):
            continue
        if isinstance(raw_attr, classmethod | staticmethod | property):
            continue
        if inspect.isfunction(raw_attr):
            pytest.fail(f"{registry_name} adapter {key} defines plain instance method {attr_name!r}")


def test_every_pipeline_package_is_imported_and_exported_at_entrypoint() -> None:
    packages = _concrete_pipeline_packages()
    assert packages, "Expected at least one concrete pipeline package under verl_omni/pipelines/"

    tree = ast.parse((_PIPELINES_DIR / "__init__.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    exported_packages: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None:
            imported.update(alias.name for alias in node.names)
        elif (
            isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
        ) or (isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__"):
            source = ast.unparse(node)
            for pkg in packages:
                if f"{pkg}.__all__" in source:
                    exported_packages.add(pkg)

    missing_imports = sorted(set(packages) - imported)
    missing_exports = sorted(set(packages) - exported_packages)
    assert not missing_imports, f"Pipeline packages missing from verl_omni/pipelines/__init__.py: {missing_imports}"
    assert not missing_exports, (
        f"Pipeline packages missing from verl_omni/pipelines/__init__.py __all__: {missing_exports}"
    )


def test_every_training_adapter_in_tree_is_stateless() -> None:
    known_bases = set(_STATELESS_BASES)
    parsed: list[tuple[Path, ast.Module]] = []
    for pkg in _concrete_pipeline_packages():
        for py_file in sorted((_PIPELINES_DIR / pkg).rglob("*.py")):
            parsed.append((py_file, ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))))

    # Propagate subclass relationships across files so multi-hop adapters are checked too.
    changed = True
    while changed:
        changed = False
        for _path, tree in parsed:
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name not in known_bases:
                    if _class_leaf_bases(node) & known_bases:
                        known_bases.add(node.name)
                        changed = True

    violations: list[str] = []
    for py_file, tree in parsed:
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            registered = any(
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Attribute)
                and dec.func.attr == "register"
                and _qualified_name(dec.func.value).rsplit(".", 1)[-1] in _STATELESS_BASES
                for dec in node.decorator_list
            )
            if not registered and not (_class_leaf_bases(node) & known_bases):
                continue
            for item in node.body:
                if not isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                decorators = {_qualified_name(dec) for dec in item.decorator_list}
                if item.name == "__init__" or not (decorators & _ALLOWED_METHOD_DECORATORS):
                    rel = py_file.relative_to(_REPO_ROOT).as_posix()
                    violations.append(f"{rel}:{item.lineno} {node.name}.{item.name}")

    assert not violations, f"Training adapters must be stateless (no __init__ or instance methods): {violations}"


def test_every_registered_vllm_rollout_adapter_declares_or_inherits_io_spec() -> None:
    classes: dict[str, tuple[str, set[str], bool]] = {}
    registered: list[tuple[str, str]] = []
    for pkg in _concrete_pipeline_packages():
        for py_file in sorted((_PIPELINES_DIR / pkg).rglob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                is_registered = any(
                    isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr == "register"
                    and _qualified_name(dec.func.value).rsplit(".", 1)[-1] == "VllmOmniPipelineBase"
                    for dec in node.decorator_list
                )
                declares_spec = any(
                    (
                        isinstance(stmt, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "diffusion_io_spec" for t in stmt.targets)
                        and not (isinstance(stmt.value, ast.Constant) and stmt.value.value is None)
                    )
                    or (
                        isinstance(stmt, ast.AnnAssign)
                        and isinstance(stmt.target, ast.Name)
                        and stmt.target.id == "diffusion_io_spec"
                        and stmt.value is not None
                        and not (isinstance(stmt.value, ast.Constant) and stmt.value.value is None)
                    )
                    for stmt in node.body
                )
                classes[node.name] = (pkg, _class_leaf_bases(node), declares_spec)
                if is_registered:
                    registered.append((pkg, node.name))

    assert registered, "Expected registered VllmOmniPipelineBase adapters under verl_omni/pipelines/"

    def _has_spec(name: str, seen: set[str]) -> bool:
        if name in seen or name not in classes:
            return False
        seen.add(name)
        _pkg, bases, declares = classes[name]
        return declares or any(_has_spec(base, seen) for base in bases)

    missing = sorted(f"{pkg}:{name}" for pkg, name in registered if not _has_spec(name, set()))
    assert not missing, f"Registered rollout adapters missing diffusion_io_spec declaration: {missing}"


def test_every_rollout_adapter_module_is_imported_by_package_init() -> None:
    missing_init_imports: list[str] = []
    for pkg in _concrete_pipeline_packages():
        rollout_file = _PIPELINES_DIR / pkg / "vllm_omni_rollout_adapter.py"
        if not rollout_file.is_file():
            continue
        init_tree = ast.parse((_PIPELINES_DIR / pkg / "__init__.py").read_text(encoding="utf-8"))
        imports_rollout = any(
            (isinstance(node, ast.ImportFrom) and node.module == "vllm_omni_rollout_adapter")
            or (
                isinstance(node, ast.Import)
                and any(alias.name.endswith(".vllm_omni_rollout_adapter") for alias in node.names)
            )
            for node in ast.walk(init_tree)
        )
        if not imports_rollout:
            missing_init_imports.append(pkg)
    assert not missing_init_imports, (
        f"Pipeline packages with vllm_omni_rollout_adapter.py not imported in __init__.py: {missing_init_imports}"
    )


def test_online_diffusion_adapters_pair_with_vllm_rollout_registrations() -> None:
    train_keys = _ast_register_keys("DiffusionModelBase")
    rollout_keys = _ast_register_keys("VllmOmniPipelineBase")
    assert train_keys and rollout_keys
    missing_rollout = set(train_keys) - set(rollout_keys) - _OFFLINE_ONLY_DIFFUSION_KEYS
    extra_rollout = set(rollout_keys) - set(train_keys)
    assert not missing_rollout, f"Online DiffusionModelBase adapters missing rollout registration: {missing_rollout}"
    assert not extra_rollout, f"Rollout adapters missing DiffusionModelBase registration: {extra_rollout}"


def test_runtime_registries_satisfy_stateless_and_io_spec_contracts() -> None:
    pytest.importorskip("verl")
    import verl_omni.pipelines  # noqa: F401
    from verl_omni.pipelines.model_base import (
        DiffusionModelARBase,
        DiffusionModelBase,
        OmniModelBase,
        OmniRolloutPipelineBase,
        VllmOmniPipelineBase,
    )
    from verl_omni.pipelines.rollout_media import DiffusionIOSpec

    assert DiffusionModelBase._registry, "Expected built-in DiffusionModelBase registrations"
    for key, adapter_cls in DiffusionModelBase._registry.items():
        _assert_stateless_class("DiffusionModelBase", key, adapter_cls)
        for required_hook in ("build_scheduler", "set_timesteps", "prepare_model_inputs"):
            hook = getattr(adapter_cls, required_hook, None)
            assert callable(hook), f"{key} missing {required_hook}"
            assert not getattr(hook, "__isabstractmethod__", False), f"{key} leaves {required_hook} abstract"

    for registry_name, registry in (
        ("DiffusionModelARBase", DiffusionModelARBase._registry),
        ("OmniModelBase", OmniModelBase._registry),
        ("OmniRolloutPipelineBase", OmniRolloutPipelineBase._registry),
    ):
        for key, adapter_cls in registry.items():
            _assert_stateless_class(registry_name, key, adapter_cls)

    assert VllmOmniPipelineBase._registry, "Expected built-in VllmOmniPipelineBase registrations"
    expected_rollout = set(_ast_register_keys("VllmOmniPipelineBase")) - _OPTIONAL_VLLM_OMNI_ROLLOUT_KEYS
    missing_runtime_rollout = expected_rollout - set(VllmOmniPipelineBase._registry)
    assert not missing_runtime_rollout, (
        f"Unconditional rollout adapters missing from VllmOmniPipelineBase._registry: {missing_runtime_rollout}"
    )
    for key, rollout_cls in VllmOmniPipelineBase._registry.items():
        spec = getattr(rollout_cls, "diffusion_io_spec", None)
        assert isinstance(spec, DiffusionIOSpec), f"VllmOmniPipelineBase{key} must declare DiffusionIOSpec"


def test_unseen_pipeline_fields_and_artifacts_traverse_real_tq_paths_without_core_edits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new pipeline declaring unseen dense, ragged, and artifact fields needs no shared trainer edits."""
    pytest.importorskip("verl")
    pytest.importorskip("transfer_queue")
    import torch
    from tensordict import NonTensorStack, TensorDict
    from transfer_queue import KVBatchMeta

    from verl_omni.pipelines.model_base import DiffusionModelBase
    from verl_omni.trainer.diffusion.v1 import tq_utils, trainer_base
    from verl_omni.workers.config.diffusion.model import DIFFUSION_TQ_KEYS, DiffusionTrajectoryFieldsConfig
    from verl_omni.workers.diffusion_tq import materialize_diffusion_trajectory

    key = ("_ContractUnseenPipeline", "flow_grpo")

    @DiffusionModelBase.register(*key)
    class _ContractUnseenAdapter(DiffusionModelBase):
        @classmethod
        def build_scheduler(cls, model_config):
            return None

        @classmethod
        def set_timesteps(cls, scheduler, model_config, device):
            return None

        @classmethod
        def prepare_model_inputs(cls, module, model_config, *args, **kwargs):
            return {}

    payload = TensorDict(
        {
            "prompts": torch.zeros(2, 4, dtype=torch.long),
            "uid": NonTensorStack("p0", "p1"),
            "rm_scores": torch.tensor([[0.5], [1.5]]),
            "all_timesteps": torch.tensor([[800.0, 400.0], [800.0, 400.0]]),
            "rollout_log_probs": torch.zeros(2, 2),
            "extra_fields": NonTensorStack({"source": "synthetic"}, {"source": "synthetic"}),
            "custom_extra_latents": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
            "custom_cond_rows": torch.arange(48, dtype=torch.float32).reshape(2, 3, 8),
            "custom_cond_rows_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long),
            "media_artifact__custom_preview": torch.zeros(2, 3, 4, 4, dtype=torch.uint8),
        },
        batch_size=[2],
    )
    reads: list[tuple[list[str], list[str] | None]] = []

    def kv_batch_get(*, keys, partition_id, select_fields):
        assert partition_id == "train"
        reads.append((list(keys), select_fields))
        indices = [0 if row_key.startswith("p0_") else 1 for row_key in keys]
        selected = payload.select(*(field for field in select_fields if field in payload.keys()))
        return selected[indices]

    def kv_retrieve_meta(*, keys, partition_id, create):
        assert partition_id == "train" and create is False
        return SimpleNamespace(field_names=list(payload.keys()))

    monkeypatch.setattr(tq_utils.tq, "kv_batch_get", kv_batch_get)
    monkeypatch.setattr(tq_utils.tq, "get_client", lambda: SimpleNamespace(kv_retrieve_meta=kv_retrieve_meta))

    try:
        assert DiffusionModelBase.peek_class(*key) is _ContractUnseenAdapter
        meta = KVBatchMeta(keys=["p1_0_0", "p0_0_0"], tags=[{}, {}], partition_id="train")
        trainer = SimpleNamespace(worker_side_trajectory=True, tokenizer=SimpleNamespace(pad_token_id=0))

        projected = trainer_base.PolicyGradientDiffusionTrainerV1._read_training_batch(trainer, meta)
        assert "custom_extra_latents" not in projected.batch.keys()
        assert "custom_cond_rows" not in projected.batch.keys()
        assert "media_artifact__custom_preview" not in projected.batch.keys()
        assert projected.non_tensor_batch[DIFFUSION_TQ_KEYS].tolist() == ["p0_0_0", "p1_0_0"]

        fields = DiffusionTrajectoryFieldsConfig(
            required=["custom_extra_latents", "custom_cond_rows", "custom_cond_rows_mask"],
            optional=["custom_optional_unused"],
        )
        materialized = materialize_diffusion_trajectory(projected.to_tensordict(), fields)
        torch.testing.assert_close(materialized["custom_extra_latents"], payload["custom_extra_latents"])
        assert materialized["custom_cond_rows"].is_nested
        assert [row.shape[0] for row in materialized["custom_cond_rows"].unbind()] == [2, 3]
        assert "custom_optional_unused" not in materialized.keys()

        dump_fields = tq_utils.diffusion_dump_tq_fields(meta)
        assert "media_artifact__custom_preview" in dump_fields
        assert "custom_extra_latents" not in dump_fields

        payload = payload.exclude("custom_extra_latents")
        with pytest.raises(ValueError, match="custom_extra_latents"):
            materialize_diffusion_trajectory(projected.to_tensordict(), fields)
        assert all(selected is not None for _keys, selected in reads)
    finally:
        DiffusionModelBase._registry.pop(key, None)
