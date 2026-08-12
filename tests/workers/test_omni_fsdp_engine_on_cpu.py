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
"""CPU-only wiring tests for ``OmniFSDPEngine`` (``unittest.mock``, no GPU).

Loads ``omni_impl.py`` via ``importlib`` with pre-mocked ``sys.modules`` to
bypass ``vllm_omni`` (CUDA-only). Verifies registration, import sources,
``_build_module`` dispatch, LoRA dtype handling, and merged-weight streaming.
"""

import ast
import importlib.util
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_OMNI_IMPL_PATH = str(
    (Path(__file__).parent.parent.parent / "verl_omni" / "workers" / "engine" / "fsdp" / "omni_impl.py").resolve()
)

_VERL_OMNI_DIR = str((Path(__file__).parent.parent.parent / "verl_omni").resolve())


def _parse_omni_impl_ast():
    """Parse ``omni_impl.py`` to an AST for static-analysis tests."""
    with open(_OMNI_IMPL_PATH) as f:
        return ast.parse(f.read())


def _make_mock_model_config(**overrides):
    """Build a minimal ``OmniModelConfig``-like mock for test wiring."""
    cfg = MagicMock()
    cfg.architecture = "Qwen3OmniMoeForConditionalGeneration"
    cfg.model_stage = "thinker"
    cfg.local_path = "/fake/model/path"
    cfg.trust_remote_code = False
    cfg.use_liger = False
    cfg.use_fused_kernels = False
    cfg.enable_gradient_checkpointing = False
    cfg.lora_rank = 0
    cfg.lora = {}

    hf_config = MagicMock()
    thinker_config = types.SimpleNamespace(tie_word_embeddings=False)
    hf_config.thinker_config = thinker_config
    cfg.hf_config = hf_config

    def _get(key, default=None):
        return getattr(cfg, key, default)

    cfg.get = _get

    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ---------------------------------------------------------------------------
# Isolated module loader
# ---------------------------------------------------------------------------


_omni_impl_cache = None


def _get_omni_impl_module():
    """Load ``omni_impl.py`` with pre-mocked ``sys.modules`` to bypass CUDA.

    Returns a cached module after the first call.
    """
    global _omni_impl_cache
    if _omni_impl_cache is not None:
        return _omni_impl_cache

    # If the module was already imported during pytest collection (e.g. by
    # ``test_omni_fsdp_merge_on_cpu.py``), reuse it to avoid re-running the
    # ``@EngineRegistry.register`` decorator and triggering a duplicate-key
    # assertion.
    _OMNI_IMPL_FQN = "verl_omni.workers.engine.fsdp.omni_impl"
    if _OMNI_IMPL_FQN in sys.modules:
        _omni_impl_cache = sys.modules[_OMNI_IMPL_FQN]
        return _omni_impl_cache

    # Pre-mock verl_omni sub-packages whose ``__init__.py`` triggers CUDA.
    for modname in (
        "verl_omni.pipelines",
        "verl_omni.models",
        "verl_omni.reward_loop",
        "verl_omni.trainer",
        "verl_omni.workers.rollout",
    ):
        if modname not in sys.modules:
            sys.modules[modname] = types.ModuleType(modname)

    # Make ``verl_omni.pipelines`` a real package so ``model_base`` is found.
    pipelines_mod = sys.modules.get("verl_omni.pipelines")
    if not hasattr(pipelines_mod, "__path__"):
        pipelines_mod.__path__ = [os.path.join(_VERL_OMNI_DIR, "pipelines")]

    # ``verl_omni.workers.engine`` — mock as a package so submodule traversal works.
    engine_mod = sys.modules.get("verl_omni.workers.engine")
    if engine_mod is None or not hasattr(engine_mod, "__path__"):
        engine_mod = types.ModuleType("verl_omni.workers.engine")
        engine_mod.__path__ = [os.path.join(_VERL_OMNI_DIR, "workers", "engine")]
        sys.modules["verl_omni.workers.engine"] = engine_mod

    # Mock ``vllm_omni`` (CUDA-only, pulled in transitively).
    if "vllm_omni" not in sys.modules:
        sys.modules["vllm_omni"] = types.ModuleType("vllm_omni")

    # Load ``verl_omni.pipelines.model_base`` (needed by ``_build_module``).
    model_base_path = os.path.join(_VERL_OMNI_DIR, "pipelines", "model_base.py")
    spec_mb = importlib.util.spec_from_file_location("verl_omni.pipelines.model_base", model_base_path)
    model_base_mod = importlib.util.module_from_spec(spec_mb)
    sys.modules["verl_omni.pipelines.model_base"] = model_base_mod
    spec_mb.loader.exec_module(model_base_mod)

    # Load ``omni_impl.py``.
    spec = importlib.util.spec_from_file_location("verl_omni.workers.engine.fsdp.omni_impl", _OMNI_IMPL_PATH)
    omni_impl = importlib.util.module_from_spec(spec)
    sys.modules["verl_omni.workers.engine.fsdp.omni_impl"] = omni_impl
    spec.loader.exec_module(omni_impl)

    _omni_impl_cache = omni_impl
    return omni_impl


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_engine_registered_via_decorator():
    """``@EngineRegistry.register`` decorator metadata via AST inspection."""
    tree = _parse_omni_impl_ast()
    class_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "OmniFSDPEngine":
            class_def = node
            break

    assert class_def is not None, "OmniFSDPEngine class not found"

    decorator_found = False
    for decorator in class_def.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        func = decorator.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "EngineRegistry"
            and func.attr == "register"
        ):
            continue
        kwargs_found = 0
        for kw in decorator.keywords:
            if kw.arg == "model_type" and isinstance(kw.value, ast.Constant):
                if kw.value.value == "omni_model":
                    kwargs_found += 1
            if kw.arg == "backend" and isinstance(kw.value, ast.List):
                vals = [e.value for e in kw.value.elts if isinstance(e, ast.Constant)]
                if set(vals) == {"fsdp", "fsdp2"}:
                    kwargs_found += 1
            if kw.arg == "device" and isinstance(kw.value, ast.List):
                vals = [e.value for e in kw.value.elts if isinstance(e, ast.Constant)]
                if set(vals) == {"cuda", "npu"}:
                    kwargs_found += 1
        assert kwargs_found >= 2, (
            f"@EngineRegistry.register missing expected kwargs (model_type/backend/device); found {kwargs_found}/3"
        )
        decorator_found = True
        break

    assert decorator_found, "@EngineRegistry.register decorator not found on OmniFSDPEngine"


def test_engine_registered_for_fsdp_and_fsdp2_backends():
    """``EngineRegistry`` has ``omni_model`` registered for fsdp and fsdp2."""
    from verl.workers.engine.base import EngineRegistry

    _get_omni_impl_module()

    engines = EngineRegistry._engines
    assert "omni_model" in engines, f"'omni_model' not in EngineRegistry._engines. Keys: {list(engines)}"
    omni_registry = engines["omni_model"]

    for backend in ("fsdp", "fsdp2"):
        assert backend in omni_registry, f"'{backend}' not registered for omni_model; registered: {list(omni_registry)}"
        for device in ("cuda", "npu"):
            entry = omni_registry[backend].get(device)
            assert entry is not None, f"No engine for omni_model/{backend}/{device}"
            assert entry.__name__ == "OmniFSDPEngine", (
                f"Engine for omni_model/{backend}/{device} is {entry.__name__}, expected OmniFSDPEngine"
            )


def test_engine_get_engine_cls_resolves_to_omni_fsdp_engine():
    """``EngineRegistry.get_engine_cls`` returns ``OmniFSDPEngine``."""
    from verl.workers.engine.base import EngineRegistry

    _get_omni_impl_module()

    os.environ["VERL_ENGINE_DEVICE"] = "cuda"
    try:
        for backend in ("fsdp", "fsdp2"):
            cls = EngineRegistry.get_engine_cls("omni_model", backend)
            assert cls.__name__ == "OmniFSDPEngine", (
                f"get_engine_cls('omni_model', '{backend}') returned {cls.__name__}, expected OmniFSDPEngine"
            )
    finally:
        os.environ.pop("VERL_ENGINE_DEVICE", None)


# ---------------------------------------------------------------------------
# ``collect_lora_params`` import source
# ---------------------------------------------------------------------------


def test_uses_verl_omni_collect_lora_params():
    """``omni_impl.collect_lora_params`` is ``verl_omni.utils.fsdp_utils`` (not verl)."""
    from verl_omni.utils.fsdp_utils import collect_lora_params as expected_func

    omni_impl = _get_omni_impl_module()

    assert omni_impl.collect_lora_params is expected_func, "omni_impl.collect_lora_params is not the verl_omni copy"


def test_collect_lora_params_import_not_from_verl():
    """AST has no ``collect_lora_params`` from ``verl.utils.fsdp_utils``."""
    tree = _parse_omni_impl_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "verl.utils.fsdp_utils":
            for alias in node.names:
                assert alias.name != "collect_lora_params", (
                    "omni_impl imports collect_lora_params from verl.utils.fsdp_utils "
                    "(should be verl_omni.utils.fsdp_utils)"
                )


# ---------------------------------------------------------------------------
# ``_build_module`` calls adapter ``configure_model``
# ---------------------------------------------------------------------------


def test_build_module_uses_auto_model_for_multimodal_lm():
    """``_build_module`` uses ``AutoModelForMultimodalLM``, not ``AutoModelForCausalLM``."""
    omni_impl = _get_omni_impl_module()
    assert omni_impl.AutoModelForMultimodalLM is not None

    tree = _parse_omni_impl_ast()
    import_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "transformers":
            for alias in node.names:
                import_names.add(alias.name)
    assert "AutoModelForMultimodalLM" in import_names, (
        f"AutoModelForMultimodalLM not imported from transformers; imports: {import_names}"
    )
    assert "AutoModelForCausalLM" not in import_names, "AutoModelForCausalLM should NOT be imported from transformers"


@pytest.mark.parametrize("architecture", ["Qwen3OmniMoeForConditionalGeneration"])
def test_build_module_calls_adapter_configure_model(architecture):
    """Mock ``from_pretrained``; verify ``adapter_cls.configure_model(module, cfg)``."""
    omni_impl = _get_omni_impl_module()
    model_config = _make_mock_model_config(architecture=architecture)

    fake_module = MagicMock(spec=torch.nn.Module)
    fake_module.named_parameters.return_value = [("weight", torch.nn.Parameter(torch.randn(2, 2)))]

    fake_adapter_cls = MagicMock()
    fake_configured_module = MagicMock(spec=torch.nn.Module)
    fake_configured_module.named_parameters.return_value = [("weight", torch.nn.Parameter(torch.randn(2, 2)))]
    fake_adapter_cls.configure_model.return_value = fake_configured_module

    model_base_mod = sys.modules["verl_omni.pipelines.model_base"]

    with (
        patch.object(
            omni_impl.AutoModelForMultimodalLM, "from_pretrained", return_value=fake_module
        ) as mock_from_pretrained,
        patch.object(model_base_mod.OmniModelBase, "get_class_by_name", return_value=fake_adapter_cls) as mock_get_cls,
        patch.object(omni_impl, "get_init_weight_context_manager", return_value=MagicMock()),
        patch.object(omni_impl.warnings, "catch_warnings", return_value=MagicMock()),
        patch("verl.utils.torch_dtypes.PrecisionType"),
    ):
        engine = object.__new__(omni_impl.OmniFSDPEngine)
        engine.model_config = model_config
        engine.engine_config = MagicMock()
        engine.engine_config.model_dtype = None
        engine.engine_config.forward_only = False
        engine.device_mesh = None

        result = engine._build_module()

        mock_from_pretrained.assert_called_once()
        call_kwargs = mock_from_pretrained.call_args.kwargs
        assert call_kwargs["pretrained_model_name_or_path"] == model_config.local_path
        assert call_kwargs["trust_remote_code"] == model_config.trust_remote_code
        assert call_kwargs["config"] is model_config.hf_config

        mock_get_cls.assert_called_once_with(
            architecture,
            model_config.model_stage,
            model_config.get("external_lib"),
        )

        fake_adapter_cls.configure_model.assert_called_once_with(fake_module, model_config)
        assert result is fake_configured_module


# ---------------------------------------------------------------------------
# ``_build_lora_module`` dtype handling
# ---------------------------------------------------------------------------


def test_build_lora_module_casts_to_target_dtype():
    """Mock a module with LoRA params; verify cast to ``lora_dtype`` and tuner flag."""
    pytest.importorskip("peft")

    from peft.tuners.tuners_utils import BaseTunerLayer

    omni_impl = _get_omni_impl_module()
    model_config = _make_mock_model_config(lora_dtype="float32", lora_rank=8)

    fake_module = MagicMock(spec=torch.nn.Module)
    lora_param = torch.nn.Parameter(torch.randn(2, 2, dtype=torch.float32), requires_grad=True)
    base_param = torch.nn.Parameter(torch.randn(2, 2, dtype=torch.float32), requires_grad=False)
    fake_module.named_parameters.return_value = [
        ("base.weight", base_param),
        ("lora_A", lora_param),
    ]

    fake_tuner = MagicMock(spec=BaseTunerLayer)
    fake_module.modules.return_value = [fake_module, fake_tuner]

    with (
        patch.object(omni_impl.FSDPEngineWithLMHead, "_build_lora_module", return_value=fake_module),
        patch("verl.utils.torch_dtypes.PrecisionType") as mock_prec,
    ):
        mock_prec.to_dtype.return_value = torch.float32
        engine = object.__new__(omni_impl.OmniFSDPEngine)
        engine.model_config = model_config

        result = engine._build_lora_module(fake_module)
        assert result is fake_module
        assert fake_tuner.cast_input_dtype_enabled is False


def test_build_lora_module_bfloat16_cast():
    """LoRA params cast float32→bfloat16 when ``lora_dtype='bfloat16'``."""
    pytest.importorskip("peft")

    from peft.tuners.tuners_utils import BaseTunerLayer

    omni_impl = _get_omni_impl_module()
    model_config = _make_mock_model_config(lora_dtype="bfloat16", lora_rank=8)

    fake_module = MagicMock(spec=torch.nn.Module)
    lora_param = torch.nn.Parameter(torch.randn(2, 2, dtype=torch.float32), requires_grad=True)
    base_param = torch.nn.Parameter(torch.randn(2, 2), requires_grad=False)
    fake_module.named_parameters.return_value = [
        ("base.weight", base_param),
        ("lora_B", lora_param),
    ]
    fake_tuner = MagicMock(spec=BaseTunerLayer)
    fake_module.modules.return_value = [fake_module, fake_tuner]

    with (
        patch.object(omni_impl.FSDPEngineWithLMHead, "_build_lora_module", return_value=fake_module),
        patch("verl.utils.torch_dtypes.PrecisionType") as mock_prec,
    ):
        mock_prec.to_dtype.return_value = torch.bfloat16
        engine = object.__new__(omni_impl.OmniFSDPEngine)
        engine.model_config = model_config

        result = engine._build_lora_module(fake_module)
        assert result is fake_module
        assert lora_param.data.dtype == torch.bfloat16, f"lora_param.dtype={lora_param.data.dtype}, expected bfloat16"
        assert base_param.data.dtype == torch.float32, f"base_param.dtype={base_param.data.dtype}, expected unchanged"
        assert fake_tuner.cast_input_dtype_enabled is False


def test_build_lora_module_skips_when_lora_dtype_none():
    """No-op when ``lora_dtype`` is ``None``."""
    omni_impl = _get_omni_impl_module()
    model_config = _make_mock_model_config(lora_dtype=None, lora_rank=8)

    fake_module = MagicMock(spec=torch.nn.Module)
    fake_module.named_parameters.return_value = [("weight", torch.nn.Parameter(torch.randn(2, 2)))]

    with patch.object(omni_impl.FSDPEngineWithLMHead, "_build_lora_module", return_value=fake_module):
        engine = object.__new__(omni_impl.OmniFSDPEngine)
        engine.model_config = model_config
        assert engine._build_lora_module(fake_module) is fake_module


# ---------------------------------------------------------------------------
# ``_merged_lora_per_tensor_param`` streams merged weights
# ---------------------------------------------------------------------------


def _make_toy_module():
    """Return a simple module with predictable ``state_dict``."""

    class ToyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
            self.bias = torch.nn.Parameter(torch.tensor([0.5]))

    return ToyModule()


def test_merged_lora_per_tensor_param_yields_name_param_tuples():
    """``_merged_lora_per_tensor_param`` yields ``(name, param)`` tuples."""
    omni_impl = _get_omni_impl_module()
    module = _make_toy_module()

    with (
        patch.object(omni_impl, "merged_lora_context", MagicMock()),
        patch.object(omni_impl, "normalize_peft_param_name", side_effect=lambda state: state),
        patch.object(omni_impl, "convert_weight_keys", side_effect=lambda state, model: state),
        patch.object(omni_impl, "log_gpu_memory_usage", MagicMock()),
        patch.object(omni_impl, "get_device_id", return_value=torch.device("cpu")),
    ):
        engine = object.__new__(omni_impl.OmniFSDPEngine)
        engine.module = module
        engine._is_offload_param = False

        merged_weights = dict(engine._merged_lora_per_tensor_param())

        assert "weight" in merged_weights
        assert "bias" in merged_weights
        assert torch.equal(merged_weights["weight"], module.weight.data)
        assert torch.equal(merged_weights["bias"], module.bias.data)


def test_merged_lora_per_tensor_param_offload_on_finally():
    """``offload_fsdp_model_to_cpu`` called in ``finally`` when offload enabled."""
    omni_impl = _get_omni_impl_module()
    module = _make_toy_module()
    offload_calls = []

    with (
        patch.object(omni_impl, "merged_lora_context", MagicMock()),
        patch.object(omni_impl, "normalize_peft_param_name", side_effect=lambda state: state),
        patch.object(omni_impl, "convert_weight_keys", side_effect=lambda state, model: state),
        patch.object(omni_impl, "log_gpu_memory_usage", MagicMock()),
        patch.object(omni_impl, "get_device_id", return_value=torch.device("cpu")),
        patch.object(
            omni_impl,
            "offload_fsdp_model_to_cpu",
            side_effect=lambda m: offload_calls.append(m),
        ),
    ):
        engine = object.__new__(omni_impl.OmniFSDPEngine)
        engine.module = module
        engine._is_offload_param = True

        merged_weights = list(engine._merged_lora_per_tensor_param())

        assert len(merged_weights) == 2  # weight + bias
        assert len(offload_calls) == 1
        assert offload_calls[0] is module


def test_merged_lora_per_tensor_param_skip_offload_when_false():
    """``offload_fsdp_model_to_cpu`` NOT called when ``_is_offload_param=False``."""
    omni_impl = _get_omni_impl_module()
    module = _make_toy_module()

    with (
        patch.object(omni_impl, "merged_lora_context", MagicMock()),
        patch.object(omni_impl, "normalize_peft_param_name", side_effect=lambda state: state),
        patch.object(omni_impl, "convert_weight_keys", side_effect=lambda state, model: state),
        patch.object(omni_impl, "log_gpu_memory_usage", MagicMock()),
        patch.object(omni_impl, "get_device_id", return_value=torch.device("cpu")),
        patch.object(omni_impl, "offload_fsdp_model_to_cpu") as mock_offload,
    ):
        engine = object.__new__(omni_impl.OmniFSDPEngine)
        engine.module = module
        engine._is_offload_param = False

        list(engine._merged_lora_per_tensor_param())
        mock_offload.assert_not_called()


def test_merged_lora_per_tensor_param_with_lora_context_manager():
    """``merged_lora_context`` entered with ``backup_adapters=True``."""
    omni_impl = _get_omni_impl_module()
    module = _make_toy_module()

    entered = []

    @contextmanager
    def fake_merged_context(actor, backup_adapters=True):
        entered.append((actor, backup_adapters))
        yield

    with (
        patch.object(omni_impl, "merged_lora_context", fake_merged_context),
        patch.object(omni_impl, "normalize_peft_param_name", side_effect=lambda state: state),
        patch.object(omni_impl, "convert_weight_keys", side_effect=lambda state, model: state),
        patch.object(omni_impl, "log_gpu_memory_usage", MagicMock()),
        patch.object(omni_impl, "get_device_id", return_value=torch.device("cpu")),
    ):
        engine = object.__new__(omni_impl.OmniFSDPEngine)
        engine.module = module
        engine._is_offload_param = False

        list(engine._merged_lora_per_tensor_param())

        assert len(entered) == 1
        actor, backup = entered[0]
        assert actor is module
        assert backup is True


class TestAdapterNameForwarding:
    def test_get_per_tensor_param_routes_adapter_name(self):
        # A non-default adapter must reach collect_lora_params and select the
        # matching peft_config; silently falling back to "default" broadcasts
        # the wrong adapter in async LoRA sync.
        import torch.nn as nn

        omni_impl = _get_omni_impl_module()

        module = nn.Module()
        module.thinker = nn.Linear(2, 2)
        old_cfg, default_cfg = MagicMock(), MagicMock()
        old_cfg.to_dict.return_value = {"adapter": "old"}
        default_cfg.to_dict.return_value = {"adapter": "default"}
        module.peft_config = {"default": default_cfg, "old": old_cfg}

        engine = object.__new__(omni_impl.OmniFSDPEngine)
        engine.module = module
        engine.model_config = _make_mock_model_config()
        engine._uses_fsdp2_cpu_offload_policy = True
        engine._is_offload_param = False
        engine._qat_enabled = False

        captured = {}

        def fake_collect(module, layered_summon, base_sync_done, adapter_name="default", **kwargs):
            captured["adapter_name"] = adapter_name
            return {"w": torch.zeros(1)}

        with (
            patch.object(omni_impl, "log_gpu_memory_usage", MagicMock()),
            patch.object(omni_impl, "collect_lora_params", side_effect=fake_collect),
            patch.object(omni_impl, "convert_weight_keys", side_effect=lambda params, module: params),
        ):
            per_tensor_param, peft_config = engine.get_per_tensor_param(base_sync_done=True, adapter_name="old")

        assert captured["adapter_name"] == "old"
        assert peft_config == {"adapter": "old"}
        assert dict(per_tensor_param).keys() == {"w"}
