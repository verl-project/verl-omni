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

"""CPU-only tests for VeOmni gradient-checkpointing train/eval forward parity.

Verifies that gradient checkpointing is NOT disabled in train mode (which
would cause OOM in full-weight training), and that eval-mode forward takes
the same checkpointed path as train-mode forward (resolving the PPO ratio
bias that previously forced checkpointing to be disabled).

Uses ``importlib`` with pre-mocked ``sys.modules`` to bypass VeOmni (CUDA-only)
imports, following the pattern in ``test_omni_fsdp_engine_on_cpu.py``.
"""

import ast
import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock

import pytest
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_VERL_OMNI_DIR = str(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "verl_omni")))
_IMPL_PATH = os.path.join(_VERL_OMNI_DIR, "workers", "engine", "veomni", "diffusion_impl.py")
_ADAPTER_PATH = os.path.join(_VERL_OMNI_DIR, "pipelines", "ltx2_flow_grpo", "veomni_training_adapter.py")


# ---------------------------------------------------------------------------
# Isolated module loader (mocks veomni, loads diffusion_impl.py)
# ---------------------------------------------------------------------------

_impl_cache = None


def _get_impl_module():
    """Load ``diffusion_impl.py`` with pre-mocked ``sys.modules``."""
    global _impl_cache
    if _impl_cache is not None:
        return _impl_cache

    _FQN = "verl_omni.workers.engine.veomni.diffusion_impl"
    if _FQN in sys.modules:
        _impl_cache = sys.modules[_FQN]
        return _impl_cache

    # Mock veomni package hierarchy.
    veomni = types.ModuleType("veomni")
    veomni.__path__ = []
    sys.modules["veomni"] = veomni

    dist = types.ModuleType("veomni.distributed")
    dist.__path__ = []
    sys.modules["veomni.distributed"] = dist

    ps = types.ModuleType("veomni.distributed.parallel_state")

    class _PS:
        sp_enabled = False
        sp_rank = 0
        dp_rank = 0
        dp_size = 1
        dp_group = None
        device_mesh = MagicMock()

    ps.init_parallel_state = lambda **kw: None
    ps.get_parallel_state = lambda: _PS
    sys.modules["veomni.distributed.parallel_state"] = ps

    offload = types.ModuleType("veomni.distributed.offloading")
    offload.load_model_to_gpu = lambda *a, **kw: None
    offload.load_optimizer = lambda *a, **kw: None
    offload.offload_model_to_cpu = lambda *a, **kw: None
    offload.offload_optimizer = lambda *a, **kw: None
    sys.modules["veomni.distributed.offloading"] = offload

    args = types.ModuleType("veomni.arguments")

    class OpsImplementationConfig:
        def __init__(self, **kw):
            pass

    args.OpsImplementationConfig = OpsImplementationConfig
    for cls_name in (
        "AcceleratorConfig",
        "FSDPConfig",
        "GradientCheckpointingConfig",
        "MixedPrecisionConfig",
        "OffloadConfig",
        "OptimizerConfig",
    ):
        setattr(args, cls_name, type(cls_name, (), {"__init__": lambda s, **kw: None}))
    sys.modules["veomni.arguments"] = args

    trainer = types.ModuleType("veomni.trainer")
    trainer.__path__ = []
    sys.modules["veomni.trainer"] = trainer

    base = types.ModuleType("veomni.trainer.base")

    class BaseTrainer:
        @staticmethod
        def _build_model(self):
            pass

        @staticmethod
        def _build_parallelized_model(self):
            pass

        @staticmethod
        def _build_optimizer(self):
            pass

        @staticmethod
        def _build_lr_scheduler(self):
            pass

        @staticmethod
        def _build_training_context(self):
            pass

    base.BaseTrainer = BaseTrainer
    sys.modules["veomni.trainer.base"] = base

    dit = types.ModuleType("veomni.trainer.dit_trainer")
    dit.DiTTrainer = type("DiTTrainer", (), {})
    for cls_name in ("DiTDataArguments", "DiTModelArguments", "DiTTrainingArguments", "VeOmniDiTArguments"):
        setattr(dit, cls_name, type(cls_name, (), {}))
    sys.modules["veomni.trainer.dit_trainer"] = dit

    # Mock verl_omni sub-packages whose __init__ triggers CUDA imports.
    root = sys.modules.setdefault("verl_omni", types.ModuleType("verl_omni"))
    root.__path__ = [_VERL_OMNI_DIR]
    for modname in (
        "verl_omni.pipelines",
        "verl_omni.models",
        "verl_omni.reward_loop",
        "verl_omni.trainer",
        "verl_omni.workers",
    ):
        if modname not in sys.modules:
            m = types.ModuleType(modname)
            m.__path__ = [os.path.join(_VERL_OMNI_DIR, *modname.split(".")[1:])]
            sys.modules[modname] = m

    # Load model_base (real module, needed by diffusion_impl).
    spec_mb = importlib.util.spec_from_file_location(
        "verl_omni.pipelines.model_base",
        os.path.join(_VERL_OMNI_DIR, "pipelines", "model_base.py"),
    )
    model_base_mod = importlib.util.module_from_spec(spec_mb)
    sys.modules["verl_omni.pipelines.model_base"] = model_base_mod
    spec_mb.loader.exec_module(model_base_mod)

    # Mock config and utils.
    config = types.ModuleType("verl_omni.workers.config")
    config.DiffusionModelConfig = type("DiffusionModelConfig", (), {})
    config.VeOmniDiffusionEngineConfig = type("VeOmniDiffusionEngineConfig", (), {})
    config.VeOmniDiffusionOptimizerConfig = type("VeOmniDiffusionOptimizerConfig", (), {})
    sys.modules["verl_omni.workers.config"] = config

    utils = types.ModuleType("verl_omni.pipelines.utils")
    utils.build_scheduler = lambda *a, **kw: MagicMock()
    utils.forward_and_sample_previous_step = lambda *a, **kw: None
    utils.prepare_model_inputs = lambda *a, **kw: ({}, None)
    sys.modules["verl_omni.pipelines.utils"] = utils

    sched = types.ModuleType("verl_omni.pipelines.schedulers")
    sched.FlowMatchSDEDiscreteScheduler = type("FlowMatchSDEDiscreteScheduler", (), {})
    sys.modules["verl_omni.pipelines.schedulers"] = sched

    # Load diffusion_impl.
    spec = importlib.util.spec_from_file_location(_FQN, _IMPL_PATH)
    impl = importlib.util.module_from_spec(spec)
    sys.modules[_FQN] = impl
    spec.loader.exec_module(impl)

    _impl_cache = impl
    return impl


_adapter_cache = None


def _get_adapter_module():
    """Load ``veomni_training_adapter.py`` with pre-mocked ``sys.modules``."""
    global _adapter_cache
    if _adapter_cache is not None:
        return _adapter_cache

    _FQN = "verl_omni.pipelines.ltx2_flow_grpo.veomni_training_adapter"
    if _FQN in sys.modules:
        _adapter_cache = sys.modules[_FQN]
        return _adapter_cache

    # Ensure verl_omni root and sub-packages are mocked (reuses _get_impl_module setup).
    _get_impl_module()

    # Load common module (real, needed by adapter).
    common_path = os.path.join(_VERL_OMNI_DIR, "pipelines", "ltx2_flow_grpo", "common.py")
    spec_common = importlib.util.spec_from_file_location("verl_omni.pipelines.ltx2_flow_grpo.common", common_path)
    common_mod = importlib.util.module_from_spec(spec_common)
    sys.modules["verl_omni.pipelines.ltx2_flow_grpo.common"] = common_mod
    spec_common.loader.exec_module(common_mod)

    spec = importlib.util.spec_from_file_location(_FQN, _ADAPTER_PATH)
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[_FQN] = adapter
    spec.loader.exec_module(adapter)

    _adapter_cache = adapter
    return adapter


# ---------------------------------------------------------------------------
# Fake model helpers
# ---------------------------------------------------------------------------


class _FakeInnerModel:
    """Minimal stand-in for VeOmni's LTX2VideoTransformer3DModel inner model."""

    def __init__(self, gradient_checkpointing=True):
        self.gradient_checkpointing = gradient_checkpointing
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


class _FakeModule:
    """Stand-in for the FSDP2-wrapped model (has ``.module`` → inner model)."""

    def __init__(self, gradient_checkpointing=True):
        self.module = _FakeInnerModel(gradient_checkpointing)
        self.training = True
        self._reshard_called = False

    def eval(self):
        self.training = False
        self.module.training = False

    def train(self):
        self.training = True
        self.module.training = True

    def reshard(self):
        self._reshard_called = True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_configure_train_mode_preserves_gradient_checkpointing():
    """``configure_train_mode`` must NOT set ``gradient_checkpointing=False``."""
    adapter = _get_adapter_module()
    LTX23FlowGRPOVeOmni = adapter.LTX23FlowGRPOVeOmni

    module = _FakeModule(gradient_checkpointing=True)
    LTX23FlowGRPOVeOmni.configure_train_mode(module)

    assert module.module.gradient_checkpointing is True, (
        "gradient_checkpointing must remain True after configure_train_mode; "
        "disabling it causes OOM in full-weight training."
    )


def test_configure_train_mode_no_gradient_checkpointing_assignment():
    """Static check: ``configure_train_mode`` body must not assign False to gradient_checkpointing."""
    with open(_ADAPTER_PATH) as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "configure_train_mode"):
            continue
        # Walk the function body looking for assignments to .gradient_checkpointing.
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Attribute) and target.attr == "gradient_checkpointing":
                        pytest.fail(
                            "configure_train_mode must not assign to gradient_checkpointing; "
                            "this was the OOM root cause."
                        )
        return
    pytest.fail("configure_train_mode not found in adapter")


def test_eval_mode_sets_inner_training_true():
    """``EngineEvalModeCtx.__enter__`` sets ``inner.training=True`` when checkpointing is on."""
    impl = _get_impl_module()
    EngineEvalModeCtx = impl.EngineEvalModeCtx

    engine = MagicMock(spec=impl.VeOmniDiffusionEngine)
    engine.module = _FakeModule(gradient_checkpointing=True)
    engine.is_param_offload_enabled = False
    engine.is_optimizer_offload_enabled = False
    engine.mode = None

    ctx = EngineEvalModeCtx(engine)
    try:
        ctx.__enter__()
        assert engine.module.module.training is True, (
            "inner.training must be True after eval_mode enter so the checkpoint path matches train mode."
        )
        assert engine.module.module.gradient_checkpointing is True
    finally:
        ctx.__exit__(None, None, None)


def test_eval_mode_skips_training_flag_when_checkpointing_disabled():
    """When ``gradient_checkpointing=False``, eval mode must NOT set ``training=True``."""
    impl = _get_impl_module()
    EngineEvalModeCtx = impl.EngineEvalModeCtx

    engine = MagicMock(spec=impl.VeOmniDiffusionEngine)
    engine.module = _FakeModule(gradient_checkpointing=False)
    engine.is_param_offload_enabled = False
    engine.is_optimizer_offload_enabled = False
    engine.mode = None

    ctx = EngineEvalModeCtx(engine)
    try:
        ctx.__enter__()
        assert engine.module.module.training is False, (
            "inner.training should stay False when gradient_checkpointing is False."
        )
    finally:
        ctx.__exit__(None, None, None)


def test_eval_mode_exit_calls_reshard():
    """``EngineEvalModeCtx.__exit__`` must call ``reshard()`` unconditionally."""
    impl = _get_impl_module()
    EngineEvalModeCtx = impl.EngineEvalModeCtx

    engine = MagicMock(spec=impl.VeOmniDiffusionEngine)
    engine.module = _FakeModule(gradient_checkpointing=True)
    engine.is_param_offload_enabled = False
    engine.is_optimizer_offload_enabled = False
    engine.mode = None

    ctx = EngineEvalModeCtx(engine)
    ctx.__enter__()
    assert not engine.module._reshard_called
    ctx.__exit__(None, None, None)
    assert engine.module._reshard_called, "reshard() must be called on eval exit"


def test_forward_backward_batch_uses_enable_grad_for_eval():
    """Static check: ``forward_backward_batch`` uses ``enable_grad`` for ``forward_only``."""
    with open(_IMPL_PATH) as f:
        source = f.read()

    # The line should reference enable_grad, not no_grad, for forward_only.
    assert "torch.enable_grad()" in source, "forward_backward_batch must use torch.enable_grad() for eval passes"
    assert "torch.no_grad()" not in source, "forward_backward_batch must NOT use torch.no_grad() for eval passes"


def test_forward_step_detaches_on_eval():
    """``forward_step`` detaches model_output and loss when ``forward_only=True``."""
    impl = _get_impl_module()
    VeOmniDiffusionEngine = impl.VeOmniDiffusionEngine

    engine = MagicMock(spec=VeOmniDiffusionEngine)
    engine.module = MagicMock()
    engine.scheduler = MagicMock()
    engine.model_config = MagicMock()
    engine.ulysses_sequence_parallel_size = 1
    engine.use_ulysses_sp = False
    engine.get_data_parallel_group = lambda: None

    # Prepare tensors with requires_grad=True.
    log_prob = torch.randn(2, 3, requires_grad=True)
    prev_mean = torch.randn(2, 3, requires_grad=True)
    std_dev = torch.randn(2, 3, requires_grad=True)
    sqrt_dt = torch.randn(2, 3, requires_grad=True)

    engine.prepare_model_inputs = MagicMock(return_value=({}, None))
    engine.prepare_model_outputs = MagicMock(
        return_value={
            "log_probs": log_prob,
            "prev_sample_mean": prev_mean,
            "std_dev_t": std_dev,
            "sqrt_dt": sqrt_dt,
        }
    )

    impl.forward_and_sample_previous_step = MagicMock(return_value=(log_prob, prev_mean, std_dev, sqrt_dt))

    # Call forward_step with forward_only=True, loss_function=None.
    loss, output = VeOmniDiffusionEngine.forward_step(
        engine,
        micro_batch=MagicMock(),
        loss_function=None,
        forward_only=True,
        step=0,
    )

    # Outputs must be detached.
    assert not output["model_output"]["log_probs"].requires_grad
    assert not output["model_output"]["prev_sample_mean"].requires_grad
    assert not loss.requires_grad


def test_forward_step_keeps_grad_when_training():
    """``forward_step`` does NOT detach when ``forward_only=False`` (train mode)."""
    from tensordict import TensorDict

    impl = _get_impl_module()
    VeOmniDiffusionEngine = impl.VeOmniDiffusionEngine

    engine = MagicMock(spec=VeOmniDiffusionEngine)
    engine.module = MagicMock()
    engine.scheduler = MagicMock()
    engine.model_config = MagicMock()
    engine.ulysses_sequence_parallel_size = 1
    engine.use_ulysses_sp = False
    engine.get_data_parallel_group = lambda: None

    log_prob = torch.randn(2, 3, requires_grad=True)
    prev_mean = torch.randn(2, 3, requires_grad=True)
    std_dev = torch.randn(2, 3, requires_grad=True)
    sqrt_dt = torch.randn(2, 3, requires_grad=True)

    engine.prepare_model_inputs = MagicMock(return_value=({}, None))
    engine.prepare_model_outputs = MagicMock(
        return_value={
            "log_probs": log_prob,
            "prev_sample_mean": prev_mean,
            "std_dev_t": std_dev,
            "sqrt_dt": sqrt_dt,
        }
    )

    impl.forward_and_sample_previous_step = MagicMock(return_value=(log_prob, prev_mean, std_dev, sqrt_dt))

    # Build a real TensorDict micro_batch so tu.get_tensordict works.
    micro_batch = TensorDict(
        {
            "old_log_probs": torch.randn(2, 3),
            "advantages": torch.randn(2, 3),
        },
        batch_size=2,
    )
    impl.tu.assign_non_tensor(micro_batch, gradient_accumulation_steps=1, sp_size=1)

    # A loss function that returns a tensor with grad.
    def loss_fn(model_output, data, dp_group):
        return log_prob.sum(), {"actor/loss": 1.0}

    loss, output = VeOmniDiffusionEngine.forward_step(
        engine,
        micro_batch=micro_batch,
        loss_function=loss_fn,
        forward_only=False,
        step=0,
    )

    # In train mode, outputs must NOT be detached.
    assert output["model_output"]["log_probs"].requires_grad
    assert loss.requires_grad
