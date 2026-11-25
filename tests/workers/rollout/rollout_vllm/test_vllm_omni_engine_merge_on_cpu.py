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
"""CPU unit tests for engine-side LoRA merge (``model.lora.merge='engine'``).

Covers config/strategy wiring (diffusion + single-stage & multi-stage AR deploy
configs), lazy ``VLLMOmniHijack`` retirement, ``vLLMOmniHttpServer`` four-phase
+ abort delegation, ``MergedLoRASyncSession`` fused onto vLLM-Omni's
``omni_lora_ipc`` trainer/worker engines, abort/lost-reply/``update_id``
recovery, multi-rank Gloo coordination, capability error surfacing, and
regressions for ``merge=False`` and ``merge=True``.
"""

from __future__ import annotations

import asyncio
from argparse import Namespace
from enum import Enum
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import msgspec
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from vllm_omni.config import omni_config
from vllm_omni.config.stage_config import load_deploy_config
from vllm_omni.engine.stage_init_utils import _project_omni_stage_engine_args
from vllm_omni.model_executor.models.qwen3_omni.pipeline import (
    QWEN3_OMNI_PIPELINE,
    QWEN3_OMNI_THINKER_ONLY_PIPELINE,
)

import verl_omni.workers.engine_workers as ew
import verl_omni.workers.rollout.vllm_rollout.utils as rollout_utils
import verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server as server_module
from verl_omni.pipelines.qwen3_omni.omni_rollout_adapter import Qwen3OmniRolloutAdapter
from verl_omni.utils.config import LoRASettings
from verl_omni.workers.rollout.vllm_rollout.merged_lora_sync import (
    MergedLoRASyncSession,
    RolloutWeightSyncClient,
    normalize_peft_config_for_merged_sync,
)
from verl_omni.workers.rollout.vllm_rollout.utils import vLLMOmniColocateWorkerExtension
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy import ARStrategy, lora_engine_extras
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy import DiffusionStrategy

try:
    import vllm_omni.lora.capability as _vllm_capability
    import vllm_omni.lora.ipc_wire as _vllm_ipc_wire
    from vllm_omni.lora.weight_transfer import (
        LoRAAdapterWeightSource as _LoRAAdapterWeightSource,
    )
    from vllm_omni.lora.weight_transfer import (
        MergedLoRAIPCTrainerWeightTransferEngine as _MergedLoRAIPCTrainerWeightTransferEngine,
    )
    from vllm_omni.lora.weight_transfer import (
        MergedLoRAIPCUpdateInfo as _MergedLoRAIPCUpdateInfo,
    )
    from vllm_omni.lora.weight_transfer import (
        MergedLoRAIPCWeightTransferEngine as _MergedLoRAIPCWeightTransferEngine,
    )
    from vllm_omni.lora.weight_transfer import (
        MergedLoRAUpdateIDMismatchError as _MergedLoRAUpdateIDMismatchError,
    )

    _HAS_VLLM_OMNI_PHASE2 = True
except ImportError:
    _vllm_capability = None
    _vllm_ipc_wire = None
    _LoRAAdapterWeightSource = None
    _MergedLoRAIPCTrainerWeightTransferEngine = None
    _MergedLoRAIPCUpdateInfo = None
    _MergedLoRAIPCWeightTransferEngine = None
    _MergedLoRAUpdateIDMismatchError = None
    _HAS_VLLM_OMNI_PHASE2 = False

_require_vllm_omni_phase2 = pytest.mark.skipif(
    not _HAS_VLLM_OMNI_PHASE2,
    reason="requires vllm-omni with phase-2 merged LoRA support (vllm-omni#10998, commit 95513b53+)",
)


def _strategy_server(merge, *, lora_rank: int = 32):
    """Build a minimal server stub whose model_config carries the requested merge."""
    return SimpleNamespace(
        config=SimpleNamespace(
            engine_kwargs={},
            tensor_model_parallel_size=2,
            text_encoder_tp_size=1,
            ulysses_degree=1,
            ring_degree=1,
            data_parallel_size=1,
            pipeline_model_parallel_size=1,
            vae_patch_parallel_size=1,
            vae_parallel_mode="tile",
            vae_use_tiling=False,
            external_lib=None,
            enable_prompt_embed_cache=False,
            prompt_embed_cache_size=0,
        ),
        model_config=OmegaConf.create(
            {
                "lora_rank": lora_rank,
                "lora": {"merge": merge},
                "true_on_policy": False,
                "architecture": "QwenImagePipeline",
                "algorithm": "flow_grpo",
            }
        ),
    )


def _make_fake_reduce_args(tensor: torch.Tensor, handle_tag: bytes) -> tuple:
    """Build a valid 15-tuple ``rebuild_cuda_tensor`` payload for CPU tests."""
    return (
        torch.Tensor,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        0,
        torch.storage.TypedStorage,
        tensor.dtype,
        0,
        handle_tag,
        tensor.numel() * tensor.element_size(),
        0,
        False,
        b"\x02" * 16,
        8,
        None,
        False,
    )


# ---------------------------------------------------------------------------
# Strategy, deploy-config, and runtime-mode wiring
# ---------------------------------------------------------------------------


def test_engine_merge_selects_itself_on_engine_args():
    """Engine merge sets the IPC backend; diffusion additionally sets ``lora_application``."""
    for strategy_cls, diffusion, expect_lora_application in (
        (DiffusionStrategy, True, "merged"),
        (ARStrategy, False, None),
    ):
        engine_args = {"enable_lora": False}
        strategy_cls(_strategy_server("engine"))._apply_engine_merge_engine_args(engine_args, diffusion=diffusion)
        assert engine_args["weight_transfer_config"] == {"backend": "omni_lora_ipc"}
        assert engine_args.get("lora_application") is expect_lora_application


def test_engine_merge_rejects_conflicting_backend_and_lora_application_overrides():
    """Conflicting ``weight_transfer_config`` or ``lora_application`` overrides fail closed."""
    ar_server = _strategy_server("engine")
    ar_server.config.engine_kwargs = {"vllm_omni": {"weight_transfer_config": {"backend": "ipc"}}}
    with pytest.raises(ValueError, match="Conflicting weight_transfer_config"):
        ARStrategy(ar_server).prepare_engine_args({}, Namespace())

    diff_server = _strategy_server("engine")
    diff_server.config.engine_kwargs = {"vllm_omni": {"lora_application": "runtime"}}
    with pytest.raises(ValueError, match="Conflicting lora_application"):
        DiffusionStrategy(diff_server)._apply_engine_merge_engine_args({}, diffusion=True)

    with pytest.raises(ValueError, match="Conflicting weight_transfer_config"):
        DiffusionStrategy(_strategy_server("engine"))._apply_engine_merge_engine_args(
            {"weight_transfer_config": {"backend": "nccl"}},
            diffusion=True,
        )


@pytest.mark.parametrize("merge", [False, True])
def test_non_engine_modes_leave_engine_args_untouched(merge):
    """Runtime-adapter and actor-merge modes set no engine-merge engine args."""
    strategy = DiffusionStrategy(_strategy_server(merge))
    engine_args = {"enable_lora": False}
    strategy._apply_engine_merge_engine_args(engine_args, diffusion=True)
    assert "weight_transfer_config" not in engine_args
    assert "lora_application" not in engine_args


def test_diffusion_strategy_prepare_engine_args_wires_engine_merge(monkeypatch):
    """DiffusionStrategy injects ``weight_transfer_config`` and ``lora_application='merged'``."""
    from verl_omni.pipelines.model_base import VllmOmniPipelineBase

    monkeypatch.setattr(VllmOmniPipelineBase, "get_pipeline_path", lambda **kwargs: None)
    strategy = DiffusionStrategy(_strategy_server("engine"))
    engine_args: dict = {}

    strategy.prepare_engine_args(engine_args, Namespace())

    assert engine_args["weight_transfer_config"] == {"backend": "omni_lora_ipc"}
    assert engine_args["lora_application"] == "merged"


def test_ar_strategy_without_deploy_config_injects_global_wt_and_drops_lora_flags():
    """Externally configured AR pipelines get global ``omni_lora_ipc`` and drop runtime LoRA flags."""
    strategy = ARStrategy(_strategy_server("engine"))
    engine_args = {"enable_lora": True, "max_loras": 1, "max_lora_rank": 32, "fully_sharded_loras": True}

    strategy.prepare_engine_args(engine_args, Namespace())

    assert engine_args["weight_transfer_config"] == {"backend": "omni_lora_ipc"}
    assert "lora_application" not in engine_args
    for key in ("enable_lora", "max_loras", "max_lora_rank", "fully_sharded_loras"):
        assert key not in engine_args


@pytest.mark.parametrize(
    ("pipeline_mode", "pipeline_cfg", "other_stage_ids"),
    [
        ("thinker_only", QWEN3_OMNI_THINKER_ONLY_PIPELINE, ()),
        ("full", QWEN3_OMNI_PIPELINE, (1, 2)),
    ],
)
def test_ar_deploy_config_scopes_engine_merge_to_policy_stage(
    monkeypatch, pipeline_mode, pipeline_cfg, other_stage_ids
):
    """Generated AR deploy configs scope ``omni_lora_ipc`` to the policy stage only.

    Round-trips the generated YAML through ``load_deploy_config`` and
    ``VllmOmniConfig.from_pipeline_config`` so non-policy stages (talker,
    code2wav) never inherit ``omni_lora_ipc`` or ``enable_lora``.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    server = _strategy_server("engine")
    strategy = ARStrategy(server)
    engine_kwargs: dict = {}

    strategy._write_deploy_config(engine_kwargs, "qwen3_omni_moe", Qwen3OmniRolloutAdapter, pipeline_mode)
    try:
        deploy = load_deploy_config(engine_kwargs["deploy_config"])
    finally:
        server._temp_deploy_ctx.cleanup()

    assert strategy.lora_stage_ids is None
    assert strategy._weight_transfer_stage_ids == [0]
    assert strategy.collective_rpc_stage_ids("abort_weight_update") == [0]

    engine_args = {
        "enable_lora": True,
        "max_loras": 1,
        "max_lora_rank": 32,
        "weight_transfer_config": {"backend": "omni_lora_ipc"},
    }
    strategy.prepare_engine_args(engine_args, Namespace())
    assert "weight_transfer_config" not in engine_args
    assert "enable_lora" not in engine_args

    resolved = omni_config.VllmOmniConfig.from_pipeline_config(
        pipeline_cfg,
        user_deploy_config=deploy,
    )
    stage0 = resolved.stage_by_id(0)
    assert stage0.weight_transfer_config == {"backend": "omni_lora_ipc"}
    projected0 = _project_omni_stage_engine_args(stage0)
    assert projected0.get("weight_transfer_config") == {"backend": "omni_lora_ipc"}
    assert "enable_lora" not in projected0

    for sid in other_stage_ids:
        other = resolved.stage_by_id(sid)
        assert other.weight_transfer_config is None
        projected = _project_omni_stage_engine_args(other)
        assert "weight_transfer_config" not in projected
        assert "enable_lora" not in projected


def test_engine_merge_disables_runtime_adapter_request_plumbing():
    """Engine-side merge suppresses ``lora_engine_extras``, ``lora_as_adapter``, and ``list_loras``."""
    model_config = SimpleNamespace(lora={"merge": "engine"}, lora_rank=32)
    assert lora_engine_extras(model_config) == {}

    server = object.__new__(server_module.vLLMOmniHttpServer)
    server.model_config = model_config
    server.engine = SimpleNamespace(list_loras=AsyncMock(side_effect=AssertionError("list_loras must not run")))
    server._lora_request_cache = server_module._LORA_REQUEST_CACHE_MISS
    server._lora_resolve_lock = asyncio.Lock()

    assert server.lora_as_adapter is False
    assert asyncio.run(server._resolve_lora_request()) is None


# ---------------------------------------------------------------------------
# Lazy VLLMOmniHijack retirement (C3)
# ---------------------------------------------------------------------------


def test_worker_extension_new_skips_eager_lora_hijack(monkeypatch):
    """Constructing ``vLLMOmniColocateWorkerExtension`` does not install ``VLLMOmniHijack``."""
    hijack_calls: list[str] = []
    monkeypatch.setattr(rollout_utils.VLLMOmniHijack, "hijack", lambda: hijack_calls.append("hijack"))
    monkeypatch.setattr(rollout_utils, "set_death_signal", lambda: None)
    monkeypatch.setattr(
        "verl_omni.workers.rollout.vllm_rollout.process_hidden_states.apply_hidden_states_patches",
        lambda: None,
    )

    ext = vLLMOmniColocateWorkerExtension()
    assert isinstance(ext, vLLMOmniColocateWorkerExtension)
    assert hijack_calls == []


def test_runtime_adapter_sync_lazily_installs_lora_hijack(monkeypatch):
    """``merge=False`` adapter sync still installs ``VLLMOmniHijack`` before ``add_lora``."""
    hijack_calls: list[str] = []
    monkeypatch.setattr(rollout_utils.VLLMOmniHijack, "hijack", lambda: hijack_calls.append("hijack"))

    class _FakeReceiver:
        def __init__(self, **kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            on_bucket_received([("foo.lora_A.weight", torch.zeros(2, 2))])

    monkeypatch.setattr(
        "verl.workers.rollout.vllm_rollout.bucketed_weight_transfer.BucketedWeightReceiver",
        _FakeReceiver,
    )

    worker = SimpleNamespace(
        device=torch.device("cpu"),
        _pending_lora_peft_config=None,
        _get_zmq_handle=lambda: "ipc:///tmp/test.sock",
        _get_standard_weight_model_and_config=lambda: None,
        remove_lora=MagicMock(),
        add_lora=MagicMock(),
    )

    vLLMOmniColocateWorkerExtension.update_weights_from_ipc(
        worker,
        peft_config={"r": 8, "lora_alpha": 16, "target_modules": ["foo"]},
        base_sync_done=True,
    )

    assert hijack_calls == ["hijack"]
    worker.remove_lora.assert_called_once()
    worker.add_lora.assert_called_once()


# ---------------------------------------------------------------------------
# vLLMOmniHttpServer four-phase + abort delegation
# ---------------------------------------------------------------------------


def test_http_server_delegates_four_phase_and_abort():
    """``vLLMOmniHttpServer`` delegates four-phase and ``abort_weight_update`` to ``AsyncOmni``."""
    engine = SimpleNamespace(
        init_weight_transfer_engine=AsyncMock(),
        start_weight_update=AsyncMock(),
        update_weights=AsyncMock(),
        finish_weight_update=AsyncMock(),
        _weight_transfer_rpc=AsyncMock(return_value=[[True, True]]),
    )
    server = object.__new__(server_module.vLLMOmniHttpServer)
    server.engine = engine

    async def _exercise():
        await server.init_weight_transfer_engine({"packed": False})
        await server.start_weight_update(update_id="uid-1")
        await server.update_weights({"names": ["a"]})
        await server.finish_weight_update()
        aborted = await server.abort_weight_update(update_id="uid-1")
        return aborted

    assert asyncio.run(_exercise()) is True
    engine.init_weight_transfer_engine.assert_awaited_once_with({"packed": False})
    engine.start_weight_update.assert_awaited_once_with(update_id="uid-1")
    engine.update_weights.assert_awaited_once_with({"names": ["a"]})
    engine.finish_weight_update.assert_awaited_once_with()
    engine._weight_transfer_rpc.assert_awaited_once_with("abort_weight_update", ("uid-1",))


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ([True], True),
        ([[False, False]], False),
        ([[False], [True]], True),
    ],
)
def test_http_server_abort_flattens_nested_bool_replies(reply, expected):
    """Nested stage/TP boolean replies collapse to True iff any worker aborted."""
    server = object.__new__(server_module.vLLMOmniHttpServer)
    server.engine = SimpleNamespace(_weight_transfer_rpc=AsyncMock(return_value=reply))

    assert asyncio.run(server.abort_weight_update(update_id="uid-2")) is expected


@pytest.mark.parametrize(
    "bad_reply",
    [
        [],
        [{"error": "worker crashed"}],
        [[{"supported": False, "error": "unsupported"}]],
    ],
)
def test_http_server_abort_fails_closed_on_worker_error_replies(bad_reply):
    """Worker error or empty replies from ``abort_weight_update`` raise ``RuntimeError``."""
    server = object.__new__(server_module.vLLMOmniHttpServer)
    server.engine = SimpleNamespace(_weight_transfer_rpc=AsyncMock(return_value=bad_reply))

    with pytest.raises(RuntimeError, match="abort_weight_update failed"):
        asyncio.run(server.abort_weight_update(update_id="uid-err"))


# ---------------------------------------------------------------------------
# PEFT config normalization, pre-#10998 guard, and loop-thread guard
# ---------------------------------------------------------------------------


def test_normalize_peft_config_handles_sets_tuples_and_enums():
    """PEFT config normalization sorts sets, unwraps enums, and is msgspec-encodable."""

    class _DummyEnum(str, Enum):
        LORA = "LORA"

    peft_config = {
        "peft_type": _DummyEnum.LORA,
        "r": 8,
        "target_modules": {"b_proj", "a_proj"},
        "modules_to_save": frozenset({"norm"}),
        "layers_to_transform": (0, 1),
        "nested": {"exclude_modules": {"lm_head"}},
    }
    normalized = normalize_peft_config_for_merged_sync(peft_config)

    assert normalized == {
        "peft_type": "LORA",
        "r": 8,
        "target_modules": ["a_proj", "b_proj"],
        "modules_to_save": ["norm"],
        "layers_to_transform": [0, 1],
        "nested": {"exclude_modules": ["lm_head"]},
    }
    # Input is not mutated and output survives msgpack serialization.
    assert peft_config["target_modules"] == {"b_proj", "a_proj"}
    assert msgspec.msgpack.decode(msgspec.msgpack.encode(normalized)) == normalized
    assert normalize_peft_config_for_merged_sync(None) is None


def test_merged_lora_sync_session_fails_fast_without_phase2_vllm_omni(monkeypatch):
    """Pre-``vllm-omni#10998`` installs fail fast with a clear version error."""
    import verl_omni.workers.rollout.vllm_rollout.merged_lora_sync as sync_mod

    def _missing():
        raise RuntimeError(
            "actor_rollout_ref.model.lora.merge='engine' requires vLLM-Omni with phase-2 merged LoRA "
            "trainer support (MergedLoRAIPCTrainerWeightTransferEngine from vllm-omni#10998, commit "
            "95513b53 or newer)."
        )

    monkeypatch.setattr(sync_mod, "_load_vllm_omni_trainer_api", _missing)
    session = MergedLoRASyncSession(server_handle=None, is_sender=False, rank=0)

    with pytest.raises(RuntimeError, match="vllm-omni#10998"):
        session.sync({"a.lora_A.weight": torch.zeros(2, 2)}, {"r": 2, "target_modules": ["a"]})


def test_rollout_weight_sync_client_refuses_blocking_event_loop_thread():
    """Calling the blocking Ray bridge from the event loop thread raises instead of deadlocking."""
    handle = SimpleNamespace(start_weight_update=SimpleNamespace(remote=AsyncMock()))

    async def _run():
        client = RolloutWeightSyncClient(handle, loop=asyncio.get_running_loop())
        with pytest.raises(RuntimeError, match="cannot block the active event loop thread"):
            client.start_weight_update(update_id="u1")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Fused vllm-omni#10998 trainer-to-worker end-to-end tests
# ---------------------------------------------------------------------------


class _FakeMergedLoRAManager:
    """In-memory stand-in recording prepare/commit/reset calls from the real worker engine."""

    def __init__(self) -> None:
        self.live_policy: dict[str, torch.Tensor] | None = None
        self.committed_rounds: list[tuple[int, dict[str, torch.Tensor], dict]] = []
        self.fail_on_prepare = False

    def prepare_merged_adapter(self, request):
        if self.fail_on_prepare:
            raise RuntimeError("prepare_merged_adapter boom")
        snapshot = {name: tensor.clone() for name, tensor in request.lora_tensors.items()}
        return (request.lora_int_id, snapshot, dict(request.peft_config or {}))

    def commit_merged_adapter(self, adapter_id, snapshot, peft_config, *, group=None):
        self.live_policy = snapshot
        self.committed_rounds.append((adapter_id, snapshot, peft_config))

    def reset_merged_adapter(self):
        self.live_policy = None


class _CPUMergedWorkerHarness:
    """Wires ``vLLMOmniHttpServer`` directly to a real ``MergedLoRAIPCWeightTransferEngine``."""

    def __init__(self, *, gpu_uuid: str = "gpu-0") -> None:
        assert _HAS_VLLM_OMNI_PHASE2
        self.gpu_uuid = gpu_uuid
        self.tensor_store: dict[bytes, torch.Tensor] = {}
        self.lora_manager = _FakeMergedLoRAManager()
        self.active = False
        self.update_id: str | None = None
        self.events: list[str] = []
        self.seen_payloads: list[dict] = []
        self.fail_next_update = False

        # Construct a real MergedLoRAIPCWeightTransferEngine without GPU init.
        engine = object.__new__(_MergedLoRAIPCWeightTransferEngine)
        engine.config = SimpleNamespace(backend="omni_lora_ipc")
        engine.vllm_config = SimpleNamespace(lora_config=None)
        engine.device = torch.device("cpu")
        engine.model = torch.nn.Module()
        engine._default_model = engine.model
        engine._default_model_config = None
        engine._original_model = engine.model
        engine.packed = False
        engine._packed_importer = SimpleNamespace(close=lambda: None)
        engine.lora_manager = self.lora_manager
        engine.cpu_group = None
        engine._last_abort_was_non_mutating = False
        engine._update_id = None
        engine._skip_next_end_weight_update = False
        self.worker_engine = engine

        harness = self

        def _cpu_receive(update_info):
            # Verify the real MergedLoRAIPCUpdateInfo decoded IPC wire envelopes back to 15-tuples.
            assert isinstance(update_info.ipc_handles, list)
            weights = []
            for name, handle_map in zip(update_info.names, update_info.ipc_handles, strict=True):
                args = handle_map[harness.gpu_uuid]
                assert isinstance(args, tuple) and len(args) == 15
                assert args[0] is torch.Tensor
                handle_tag = args[7]
                weights.append((name, harness.tensor_store[handle_tag].clone()))
            engine.model.load_weights(weights)

        engine._cpu_receive_patch = _cpu_receive

        # Wrap the worker methods to mirror DiffusionWorker/WorkerBase lifecycle guards.
        self.async_omni = SimpleNamespace(
            init_weight_transfer_engine=self._init_wte,
            start_weight_update=self._start_update,
            update_weights=self._update_weights,
            finish_weight_update=self._finish_update,
            _weight_transfer_rpc=self._weight_transfer_rpc,
        )
        self.server = object.__new__(server_module.vLLMOmniHttpServer)
        self.server.engine = self.async_omni
        self.handle = SimpleNamespace(
            init_weight_transfer_engine=SimpleNamespace(remote=self.server.init_weight_transfer_engine),
            start_weight_update=SimpleNamespace(remote=self.server.start_weight_update),
            update_weights=SimpleNamespace(remote=self.server.update_weights),
            finish_weight_update=SimpleNamespace(remote=self.server.finish_weight_update),
            abort_weight_update=SimpleNamespace(remote=self.server.abort_weight_update),
            clear_kv_cache=SimpleNamespace(remote=AsyncMock()),
            set_global_steps=SimpleNamespace(remote=AsyncMock()),
        )

    def make_reduce_fn(self):
        counter = [0]

        def _reduce(tensor: torch.Tensor):
            counter[0] += 1
            tag = f"handle-{counter[0]:04d}".encode("ascii").ljust(16, b"\x00")
            self.tensor_store[tag] = tensor.detach().clone()
            return None, _make_fake_reduce_args(tensor, tag)

        return _reduce

    async def _init_wte(self, init_info: dict) -> None:
        self.events.append("init")
        parsed = self.worker_engine.parse_init_info(init_info)
        self.worker_engine.init_transfer_engine(parsed)

    async def _start_update(self, update_id: str | None = None) -> None:
        self.events.append(f"start:{update_id}")
        if self.active:
            raise RuntimeError("start_weight_update called while a weight update is already active.")
        self.worker_engine.start_weight_update(update_id=update_id)
        self.active = True
        self.update_id = update_id

    async def _update_weights(self, update_info: dict) -> None:
        # Cross-process AR workers msgpack-encode every payload; prove it round-trips cleanly.
        roundtripped = msgspec.msgpack.decode(msgspec.msgpack.encode(update_info))
        self.seen_payloads.append(roundtripped)
        self.events.append(f"update:{roundtripped.get('update_id')}")
        if not self.active:
            raise RuntimeError("start_weight_update must be called before update_weights.")
        if self.fail_next_update:
            self.fail_next_update = False
            raise RuntimeError("injected transport error during update_weights")
        with (
            patch(
                "vllm.distributed.weight_transfer.ipc_engine.IPCWeightTransferEngine.receive_weights",
                side_effect=self.worker_engine._cpu_receive_patch,
            ),
            patch("torch.accelerator.synchronize", lambda *a, **kw: None),
        ):
            try:
                self.worker_engine.update_weights(roundtripped)
            except Exception as exc:
                if isinstance(exc, _MergedLoRAUpdateIDMismatchError):
                    raise
                self.active = False
                self.update_id = None
                raise

    async def _finish_update(self, weight_version: str | None = None) -> None:
        self.events.append("finish")
        if not self.active:
            raise RuntimeError("finish_weight_update called without a matching start_weight_update.")
        try:
            self.worker_engine.finish_weight_update()
        finally:
            self.active = False
            self.update_id = None

    async def _weight_transfer_rpc(self, method: str, args: tuple = ()) -> list:
        assert method == "abort_weight_update"
        update_id = args[0] if args else None
        self.events.append(f"abort:{update_id}")
        if not self.active:
            return [False]
        if update_id is not None and self.update_id != update_id:
            return [False]
        self.active = False
        self.update_id = None
        self.worker_engine.abort_weight_update()
        return [True]


@_require_vllm_omni_phase2
def test_fused_trainer_to_worker_two_consecutive_rounds():
    """Real trainer engine drives two rounds into the real worker engine."""
    harness = _CPUMergedWorkerHarness(gpu_uuid="gpu-0")
    ipc_collects: list[int] = []
    session = MergedLoRASyncSession(
        server_handle=harness.handle,
        is_sender=True,
        rank=0,
        gpu_uuid="gpu-0",
        reduce_tensor_fn=harness.make_reduce_fn(),
        stage_tensor_fn=lambda t: t.detach().contiguous(),
        ipc_collect_fn=lambda: ipc_collects.append(1),
        synchronize_fn=lambda: None,
    )

    async def _run_rounds():
        loop = asyncio.get_running_loop()
        peft_cfg = {"r": 4, "lora_alpha": 8, "target_modules": {"q_proj", "v_proj"}}
        w1 = {
            "base_model.model.v_proj.lora_B.weight": torch.full((4, 4), 1.0),
            "base_model.model.q_proj.lora_A.weight": torch.full((4, 8), 2.0),
        }
        w2 = {
            "base_model.model.v_proj.lora_B.weight": torch.full((4, 4), 3.0),
            "base_model.model.q_proj.lora_A.weight": torch.full((4, 8), 4.0),
        }
        await asyncio.to_thread(session.sync, w1, peft_cfg, loop=loop)
        await asyncio.to_thread(session.sync, w2, peft_cfg, loop=loop)

    asyncio.run(_run_rounds())

    # init runs once on round 1; both rounds send wire-encoded handles + update_id and commit.
    assert harness.events.count("init") == 1
    assert harness.events.count("finish") == 2
    assert len(ipc_collects) == 2
    assert len(harness.lora_manager.committed_rounds) == 2

    # Verify wire format carried vllm_omni.lora.ipc_wire envelopes and sorted names/target_modules.
    payload0 = harness.seen_payloads[0]
    assert payload0["names"] == [
        "base_model.model.q_proj.lora_A.weight",
        "base_model.model.v_proj.lora_B.weight",
    ]
    assert payload0["peft_config"]["target_modules"] == ["q_proj", "v_proj"]
    assert isinstance(payload0["update_id"], str) and payload0["update_id"]
    for handle_map in payload0["ipc_handles"]:
        assert _vllm_ipc_wire.is_encoded_ipc_handle(handle_map["gpu-0"])

    # Live policy reflects round 2's weights.
    torch.testing.assert_close(
        harness.lora_manager.live_policy["base_model.model.q_proj.lora_A.weight"],
        torch.full((4, 8), 4.0),
    )


@_require_vllm_omni_phase2
def test_nonzero_global_rank_replica_leader_publishes_to_its_replica():
    """A replica leader on global actor rank > 0 (e.g. replica 1 under TP=2) stays a sender."""
    harness = _CPUMergedWorkerHarness(gpu_uuid="gpu-2")
    session = MergedLoRASyncSession(
        server_handle=harness.handle,
        is_sender=True,
        rank=2,
        gpu_uuid="gpu-2",
        reduce_tensor_fn=harness.make_reduce_fn(),
        stage_tensor_fn=lambda t: t.detach().contiguous(),
        ipc_collect_fn=lambda: None,
        synchronize_fn=lambda: None,
    )

    async def _run():
        loop = asyncio.get_running_loop()
        peft_cfg = {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj"]}
        w = {"base_model.model.q_proj.lora_A.weight": torch.full((4, 8), 5.0)}
        await asyncio.to_thread(session.sync, w, peft_cfg, loop=loop)

    asyncio.run(_run())

    assert harness.events.count("init") == 1
    assert len(harness.lora_manager.committed_rounds) == 1
    torch.testing.assert_close(
        harness.lora_manager.live_policy["base_model.model.q_proj.lora_A.weight"],
        torch.full((4, 8), 5.0),
    )


@_require_vllm_omni_phase2
def test_fused_trainer_aborts_cleanly_and_recovers_after_failed_update():
    """Failed ``update_weights`` calls ``abort_weight_update(update_id)``, keeps prior policy, and recovers."""
    harness = _CPUMergedWorkerHarness(gpu_uuid="gpu-0")
    session = MergedLoRASyncSession(
        server_handle=harness.handle,
        is_sender=True,
        rank=0,
        gpu_uuid="gpu-0",
        reduce_tensor_fn=harness.make_reduce_fn(),
        stage_tensor_fn=lambda t: t.detach().contiguous(),
        ipc_collect_fn=lambda: None,
        synchronize_fn=lambda: None,
    )

    async def _run():
        loop = asyncio.get_running_loop()
        peft_cfg = {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj"]}
        w1 = {"base_model.model.q_proj.lora_A.weight": torch.full((4, 8), 1.0)}
        w_bad = {"base_model.model.q_proj.lora_A.weight": torch.full((4, 8), 99.0)}
        w2 = {"base_model.model.q_proj.lora_A.weight": torch.full((4, 8), 2.0)}

        await asyncio.to_thread(session.sync, w1, peft_cfg, loop=loop)
        harness.fail_next_update = True
        with pytest.raises(RuntimeError, match="injected transport error"):
            await asyncio.to_thread(session.sync, w_bad, peft_cfg, loop=loop)
        # Previous policy stays live and worker session is unwedged for the next round.
        assert harness.active is False
        torch.testing.assert_close(
            harness.lora_manager.live_policy["base_model.model.q_proj.lora_A.weight"],
            torch.full((4, 8), 1.0),
        )
        await asyncio.to_thread(session.sync, w2, peft_cfg, loop=loop)

    asyncio.run(_run())

    assert any(event.startswith("abort:") for event in harness.events)
    assert len(harness.lora_manager.committed_rounds) == 2
    torch.testing.assert_close(
        harness.lora_manager.live_policy["base_model.model.q_proj.lora_A.weight"],
        torch.full((4, 8), 2.0),
    )


@_require_vllm_omni_phase2
def test_update_id_protects_active_session_from_stale_abort_and_chunk():
    """Lost-reply abort or stale chunk with a foreign ``update_id`` cannot disturb an active session."""
    harness = _CPUMergedWorkerHarness(gpu_uuid="gpu-0")

    async def _run():
        await harness.server.init_weight_transfer_engine({"packed": False})
        await harness.server.start_weight_update(update_id="active-session-id")

        # Foreign caller's abort returns False and leaves active session open.
        assert await harness.server.abort_weight_update(update_id="stale-session-id") is False
        assert harness.active is True
        assert harness.update_id == "active-session-id"

        # Foreign caller's chunk raises MergedLoRAUpdateIDMismatchError without unwinding active session.
        t = torch.ones((2, 2))
        _, raw_args = harness.make_reduce_fn()(t)
        stale_chunk = {
            "names": ["base_model.model.q_proj.lora_A.weight"],
            "dtype_names": ["float32"],
            "shapes": [[2, 2]],
            "ipc_handles": [{"gpu-0": _vllm_ipc_wire.encode_ipc_handle_args(raw_args)}],
            "peft_config": {"r": 2, "lora_alpha": 4, "target_modules": ["q_proj"]},
            "lora_int_id": 1,
            "update_id": "stale-session-id",
        }
        with pytest.raises(_MergedLoRAUpdateIDMismatchError):
            await harness.server.update_weights(stale_chunk)
        assert harness.active is True
        assert harness.update_id == "active-session-id"

        # Matching caller finishes cleanly.
        good_chunk = {**stale_chunk, "update_id": "active-session-id"}
        await harness.server.update_weights(good_chunk)
        await harness.server.finish_weight_update()

    asyncio.run(_run())
    assert harness.active is False
    assert len(harness.lora_manager.committed_rounds) == 1


# ---------------------------------------------------------------------------
# Multi-rank Gloo fused trainer test (world_size=2)
# ---------------------------------------------------------------------------


def _gloo_worker_entry(rank: int, world_size: int, store_path: str, fail_sender: bool, result_queue) -> None:
    """Spawned Gloo rank exercising ``MergedLoRASyncSession`` across two trainer ranks."""
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group(backend="gloo", store=store, rank=rank, world_size=world_size)
    try:
        sent_payloads: list[dict] = []
        aborted_ids: list[str | None] = []

        def _fake_remote(name: str):
            def _fn(*args, **kwargs):
                if name == "update_weights":
                    sent_payloads.append(args[0])
                    if fail_sender:
                        raise RuntimeError("sender rank RPC failure")
                elif name == "abort_weight_update":
                    aborted_ids.append(kwargs.get("update_id"))
                    return True
                return None

            return SimpleNamespace(remote=_fn)

        handle = SimpleNamespace(
            init_weight_transfer_engine=_fake_remote("init_weight_transfer_engine"),
            start_weight_update=_fake_remote("start_weight_update"),
            update_weights=_fake_remote("update_weights"),
            finish_weight_update=_fake_remote("finish_weight_update"),
            abort_weight_update=_fake_remote("abort_weight_update"),
        )
        gpu_uuid = f"gpu-{rank}"
        session = MergedLoRASyncSession(
            server_handle=handle if rank == 0 else None,
            is_sender=(rank == 0),
            rank=rank,
            gpu_uuid=gpu_uuid,
            call_fn=lambda fn, *args, **kwargs: fn(*args, **kwargs),
            reduce_tensor_fn=lambda tensor: (None, _make_fake_reduce_args(tensor, f"r{rank}".encode().ljust(16, b"0"))),
            stage_tensor_fn=lambda t: t.detach().contiguous(),
            ipc_collect_fn=lambda: None,
            synchronize_fn=lambda: None,
        )
        weights = {
            "base_model.model.b.lora_B.weight": torch.zeros(2, 4),
            "base_model.model.a.lora_A.weight": torch.zeros(4, 2),
        }
        peft_cfg = {"r": 4, "lora_alpha": 8, "target_modules": {"a", "b"}}
        error_msg = None
        try:
            session.sync(weights, peft_cfg)
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
        result_queue.put((rank, error_msg, sent_payloads, aborted_ids))
    finally:
        dist.destroy_process_group()


@_require_vllm_omni_phase2
@pytest.mark.parametrize("fail_sender", [False, True])
def test_multirank_gloo_merged_lora_sync_merges_handles_and_propagates_errors(tmp_path, fail_sender):
    """Two Gloo ranks merge UUID handle dicts on rank 0 and propagate sender failures to rank 1."""
    store_path = str(tmp_path / f"gloo_store_{int(fail_sender)}")
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    mp.spawn(
        _gloo_worker_entry,
        args=(2, store_path, fail_sender, result_queue),
        nprocs=2,
        join=True,
    )
    results = dict(
        (rank, (err, payloads, aborts)) for rank, err, payloads, aborts in (result_queue.get(), result_queue.get())
    )

    if not fail_sender:
        assert results[0][0] is None
        assert results[1][0] is None
        assert len(results[0][1]) == 1
        assert results[1][1] == []
        payload = results[0][1][0]
        assert payload["names"] == [
            "base_model.model.a.lora_A.weight",
            "base_model.model.b.lora_B.weight",
        ]
        # Rank 0 received merged wire-encoded handles for both GPU UUIDs.
        for handle_map in payload["ipc_handles"]:
            assert set(handle_map.keys()) == {"gpu-0", "gpu-1"}
            assert _vllm_ipc_wire.is_encoded_ipc_handle(handle_map["gpu-0"])
            assert _vllm_ipc_wire.is_encoded_ipc_handle(handle_map["gpu-1"])
    else:
        assert results[0][0] is not None and "sender rank RPC failure" in results[0][0]
        assert results[1][0] is not None and "rank 0" in results[1][0]
        assert len(results[0][2]) == 1 and results[0][2][0] is not None


def _gloo_two_replica_entry(rank: int, world_size: int, store_path: str, fail_rank2: bool, result_queue) -> None:
    """Spawned 4-rank Gloo topology with 2 replicas (leaders at global ranks 0 and 2)."""
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group(backend="gloo", store=store, rank=rank, world_size=world_size)
    try:
        is_sender = rank in (0, 2)
        sent_payloads: list[dict] = []
        aborted_ids: list[str | None] = []

        def _fake_remote(name: str):
            def _fn(*args, **kwargs):
                if name == "update_weights":
                    sent_payloads.append(args[0])
                    if fail_rank2 and rank == 2:
                        raise RuntimeError("replica-1 leader failure")
                elif name == "abort_weight_update":
                    aborted_ids.append(kwargs.get("update_id"))
                    return True
                return None

            return SimpleNamespace(remote=_fn)

        handle = (
            SimpleNamespace(
                init_weight_transfer_engine=_fake_remote("init_weight_transfer_engine"),
                start_weight_update=_fake_remote("start_weight_update"),
                update_weights=_fake_remote("update_weights"),
                finish_weight_update=_fake_remote("finish_weight_update"),
                abort_weight_update=_fake_remote("abort_weight_update"),
            )
            if is_sender
            else None
        )
        session = MergedLoRASyncSession(
            server_handle=handle,
            is_sender=is_sender,
            rank=rank,
            gpu_uuid=f"gpu-{rank}",
            call_fn=lambda fn, *args, **kwargs: fn(*args, **kwargs),
            reduce_tensor_fn=lambda tensor: (None, _make_fake_reduce_args(tensor, f"r{rank}".encode().ljust(16, b"0"))),
            stage_tensor_fn=lambda t: t.detach().contiguous(),
            ipc_collect_fn=lambda: None,
            synchronize_fn=lambda: None,
        )
        weights = {"base_model.model.a.lora_A.weight": torch.zeros(4, 2)}
        peft_cfg = {"r": 4, "lora_alpha": 8, "target_modules": ["a"]}
        error_msg = None
        try:
            session.sync(weights, peft_cfg)
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
        result_queue.put((rank, error_msg, sent_payloads, aborted_ids))
    finally:
        dist.destroy_process_group()


@_require_vllm_omni_phase2
@pytest.mark.parametrize("fail_rank2", [False, True])
def test_multirank_gloo_two_replicas_four_ranks_sync_and_second_sender_failure(tmp_path, fail_rank2):
    """4 ranks / 2 replicas (leaders at ranks 0 and 2) both publish merged handles and propagate rank-2 failure."""
    store_path = str(tmp_path / f"gloo_2rep_{int(fail_rank2)}")
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    mp.spawn(
        _gloo_two_replica_entry,
        args=(4, store_path, fail_rank2, result_queue),
        nprocs=4,
        join=True,
    )
    results = {rank: (err, payloads, aborts) for rank, err, payloads, aborts in (result_queue.get() for _ in range(4))}

    if not fail_rank2:
        for r in range(4):
            assert results[r][0] is None
        for leader in (0, 2):
            assert len(results[leader][1]) == 1
            handle_map = results[leader][1][0]["ipc_handles"][0]
            assert set(handle_map.keys()) == {"gpu-0", "gpu-1", "gpu-2", "gpu-3"}
        for follower in (1, 3):
            assert results[follower][1] == []
    else:
        assert results[2][0] is not None and "replica-1 leader failure" in results[2][0]
        assert len(results[2][2]) == 1 and results[2][2][0] is not None
        for peer in (0, 1, 3):
            assert results[peer][0] is not None and "rank 2" in results[peer][0]


@_require_vllm_omni_phase2
def test_sync_merged_lora_cancellation_aborts_and_drains_worker_thread():
    """Cancelling ``_sync_merged_lora_weights`` aborts before finish and waits for the worker thread to settle."""
    harness = _CPUMergedWorkerHarness(gpu_uuid="gpu-0")
    update_started = asyncio.Event()
    allow_update_return = asyncio.Event()
    orig_update = harness.server.update_weights

    async def _slow_update(update_info: dict) -> None:
        update_started.set()
        await allow_update_return.wait()
        await orig_update(update_info)

    harness.handle.update_weights = SimpleNamespace(remote=_slow_update)
    session = MergedLoRASyncSession(
        server_handle=harness.handle,
        is_sender=True,
        rank=0,
        gpu_uuid="gpu-0",
        reduce_tensor_fn=harness.make_reduce_fn(),
        stage_tensor_fn=lambda t: t.detach().contiguous(),
        ipc_collect_fn=lambda: None,
        synchronize_fn=lambda: None,
    )
    worker = _make_actor_worker(engine_merge=True, rollout_rank=0)
    worker.rollout.server_handle = harness.handle
    worker._merged_lora_sync_session = session

    async def _run():
        weights = {"base_model.model.q_proj.lora_A.weight": torch.full((4, 8), 7.0)}
        peft_cfg = {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj"]}
        task = asyncio.create_task(
            ew.ActorRolloutRefWorker._sync_merged_lora_weights(worker, weights, peft_cfg, lambda op: op({}))
        )
        await update_started.wait()
        task.cancel()
        await asyncio.sleep(0.02)
        # Task must stay pending until the worker thread finishes abort cleanup.
        assert not task.done()
        allow_update_return.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(_run())

    # finish_weight_update was skipped because cancel_event fired before it; abort cleaned up the worker.
    assert "finish" not in harness.events
    assert any(event.startswith("abort:") for event in harness.events)
    assert harness.active is False
    assert harness.lora_manager.committed_rounds == []


# ---------------------------------------------------------------------------
# ActorRolloutRefWorker.update_weights integration + merge=False/True regressions
# ---------------------------------------------------------------------------


def _make_actor_worker(*, merge: bool = False, engine_merge: bool = False, rollout_rank: int = 0):
    """Build a minimal ``ActorRolloutRefWorker`` for testing ``update_weights`` routing."""
    worker = object.__new__(ew.ActorRolloutRefWorker)
    engine = MagicMock(spec=["module", "get_per_tensor_param", "is_param_offload_enabled"])
    engine.is_param_offload_enabled = False
    engine.module = SimpleNamespace(peft_config={"default": SimpleNamespace(to_dict=lambda: {"r": 8})})
    engine.get_per_tensor_param.return_value = (
        iter([("base_model.model.q_proj.lora_A.weight", torch.zeros(2, 4))]),
        {"r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]},
    )
    worker.actor = SimpleNamespace(engine=engine)

    server_handle = MagicMock()
    server_handle.clear_kv_cache.remote = AsyncMock()
    server_handle.set_global_steps.remote = AsyncMock()
    rollout = MagicMock()
    rollout.zmq_handle = "ipc:///tmp/test-engine-merge.sock"
    rollout.use_shm = False
    rollout.rollout_rank = rollout_rank
    rollout._ensure_server_handle.return_value = rollout_rank == 0
    rollout.server_handle = server_handle if rollout_rank == 0 else None
    rollout._execute_method = AsyncMock(return_value=None)
    rollout.update_weights = AsyncMock()
    rollout.resume = AsyncMock()
    rollout.release = AsyncMock()
    worker.rollout = rollout

    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(
            free_cache_engine=False,
            checkpoint_engine=SimpleNamespace(backend="naive", update_weights_bucket_megabytes=16),
        )
    )
    worker._rank = rollout_rank
    worker.lora_settings = LoRASettings(
        rank=8,
        alpha=16,
        adapter_path=None,
        merge=merge,
        adapters=("default",),
        enabled=True,
        engine_merge=engine_merge,
    )
    worker.base_sync_done = True
    worker.layered_summon = False
    worker.rollout_adapter = "default"
    worker._zmq_update_seq = 0
    worker._merged_lora_sync_session = None
    worker._offload_actor_and_empty_cache = lambda timings=None: None
    return worker


def test_actor_update_weights_routes_engine_merge_through_sync_session_and_stamps():
    """Engine-merge ``update_weights`` skips ZMQ ``BucketedWeightSender``, runs session sync, and stamps step."""
    worker = _make_actor_worker(engine_merge=True, rollout_rank=0)
    sync_calls: list[tuple] = []

    class _FakeSession:
        def __init__(self, *, server_handle, is_sender, rank):
            self.server_handle = server_handle
            self.is_sender = is_sender
            self.rank = rank

        def sync(self, lora_weights, peft_config, *, loop=None, cancel_event=None):
            sync_calls.append((sorted(lora_weights), peft_config, self.is_sender, loop is not None, cancel_event))

    with (
        patch("verl_omni.workers.rollout.vllm_rollout.merged_lora_sync.MergedLoRASyncSession", _FakeSession),
        patch.object(ew, "BucketedWeightSender", side_effect=AssertionError("ZMQ sender must not run in engine merge")),
        patch.object(ew, "log_gpu_memory_usage", MagicMock()),
        patch.object(ew, "set_expandable_segments", MagicMock()),
        patch.object(
            ew,
            "get_torch_device",
            return_value=MagicMock(is_available=MagicMock(return_value=False)),
        ),
    ):
        asyncio.run(ew.ActorRolloutRefWorker.update_weights(worker, global_steps=11, mode="naive"))

    assert len(sync_calls) == 1
    assert sync_calls[0][0] == ["base_model.model.q_proj.lora_A.weight"]
    assert sync_calls[0][2] is True
    assert sync_calls[0][3] is True
    worker.rollout._execute_method.assert_not_awaited()
    worker.rollout.server_handle.clear_kv_cache.remote.assert_awaited_once()
    worker.rollout.server_handle.set_global_steps.remote.assert_awaited_once_with(11)


def test_actor_update_weights_regressions_for_merge_false_and_merge_true():
    """``merge=False`` still drives ZMQ adapter sync; ``merge=True`` still drives full-weight rollout sync."""
    # Regression 1: merge=False keeps BucketedWeightSender + update_weights_from_ipc.
    w_false = _make_actor_worker(merge=False, engine_merge=False, rollout_rank=0)
    sender = MagicMock(async_send_weights=AsyncMock())
    with (
        patch.object(ew, "BucketedWeightSender", return_value=sender),
        patch.object(ew, "log_gpu_memory_usage", MagicMock()),
        patch.object(ew, "set_expandable_segments", MagicMock()),
        patch.object(ew, "get_torch_device", return_value=MagicMock(is_available=MagicMock(return_value=False))),
    ):
        asyncio.run(ew.ActorRolloutRefWorker.update_weights(w_false, global_steps=5, mode="naive"))
    sender.async_send_weights.assert_awaited_once()
    w_false.rollout._execute_method.assert_awaited_once()

    # Regression 2: merge=True takes the standard full-weight rollout.update_weights path.
    w_true = _make_actor_worker(merge=True, engine_merge=False, rollout_rank=0)
    with (
        patch.object(
            ew, "BucketedWeightSender", side_effect=AssertionError("ZMQ fast path must not run for merge=True")
        ),
        patch.object(ew, "log_gpu_memory_usage", MagicMock()),
        patch.object(ew, "set_expandable_segments", MagicMock()),
    ):
        asyncio.run(ew.ActorRolloutRefWorker.update_weights(w_true, global_steps=6, mode="naive"))
    w_true.rollout.update_weights.assert_awaited_once()


# ---------------------------------------------------------------------------
# Engine capability error surfacing (vllm_omni.lora.capability)
# ---------------------------------------------------------------------------


@_require_vllm_omni_phase2
def test_capability_registry_surfaces_named_errors_for_unsupported_models():
    """Unsupported AR/diffusion pipelines raise ``MergedLoRACapabilityError`` with a named reason."""
    # Synthetic models bearing the registered class names of unsupported families.
    minicpm_cls = type("MiniCPMO45OmniForConditionalGeneration", (torch.nn.Module,), {})
    sd3_cls = type("StableDiffusion3Pipeline", (torch.nn.Module,), {})
    boogu_cls = type("BooguImagePipeline", (torch.nn.Module,), {})

    with pytest.raises(_vllm_capability.MergedLoRACapabilityError, match="not_capable"):
        _vllm_capability.resolve_merged_lora_spec(minicpm_cls(), stage="ar")
    with pytest.raises(_vllm_capability.MergedLoRACapabilityError, match="not_capable"):
        _vllm_capability.resolve_merged_lora_spec(sd3_cls(), stage="diffusion")
    with pytest.raises(_vllm_capability.MergedLoRACapabilityError, match="not_capable"):
        _vllm_capability.resolve_merged_lora_spec(boogu_cls(), stage="diffusion")

    # Supported pipelines remain registered as SUPPORTED.
    assert (
        _vllm_capability.DIFFUSION_MERGED_LORA_REGISTRY["QwenImagePipeline"].status
        is _vllm_capability.MergedLoRACapabilityStatus.SUPPORTED
    )
    assert (
        _vllm_capability.AR_MERGED_LORA_REGISTRY["Qwen3OmniMoeThinkerForConditionalGeneration"].status
        is _vllm_capability.MergedLoRACapabilityStatus.SUPPORTED
    )
