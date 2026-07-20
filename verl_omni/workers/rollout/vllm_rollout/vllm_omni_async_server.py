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
import argparse
import asyncio
import fcntl
import getpass
import json
import logging
import os
import socket
import threading
import time
import zlib
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import ray
import torch
import torchvision.transforms as T
import vllm_omni.entrypoints.cli.serve
from vllm_omni.entrypoints.cli.serve import run_headless as run_omni_headless
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.import_utils import import_external_libs
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.utils import run_uvicorn
from verl.workers.rollout.vllm_rollout.utils import (
    SuppressSignalInThread,
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
)
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica
from vllm import SamplingParams
from vllm.entrypoints.openai.api_server import build_app
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.entrypoints import AsyncOmni
from vllm_omni.entrypoints.openai.api_server import omni_init_app_state
from vllm_omni.inputs.data import OmniCustomPrompt, OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.utils.tracking_parser import TrackingNamespace

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.workers.config import DiffusionModelConfig, DiffusionRolloutConfig
from verl_omni.workers.rollout.vllm_rollout.placement_guard import (
    estimate_outer_rollout_replicas,
    validate_vllm_omni_rollout_placement,
)
from verl_omni.workers.rollout.replica import DiffusionOutput

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def _append_vllm_logprob_debug_jsonl(event: str, payload: dict[str, Any]) -> None:
    output_path = os.environ.get("VERL_OMNI_VLLM_LOGPROB_DEBUG_JSONL", "")
    if not output_path:
        return
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        record = {
            "event": event,
            "time": time.time(),
            "pid": os.getpid(),
            "host": socket.gethostname(),
            **payload,
        }
        with open(output_path, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        logger.exception("Failed to append vLLM-Omni logprob debug jsonl")


class vLLMOmniHttpServer(vLLMHttpServer):
    """vLLM-Omni http server in single node, this is equivalent to launch server with command line:
    ```
    vllm serve --tensor-parallel-size=8 ...
    ```
    """

    # -----------------------------------------------------------------------
    # Initialisation hooks
    # -----------------------------------------------------------------------

    def _init_model_config(self, model_config):
        """AR mode uses HFModelConfig; diffusion uses DiffusionModelConfig.

        Mode is selected by ``engine_kwargs.vllm_omni.output_mode`` ("ar" vs the
        default "diffusion").
        """
        engine_kwargs = getattr(self.config, "engine_kwargs", None) or {}
        omni_kwargs = engine_kwargs.get("vllm_omni", {}) or {}
        self._ar_mode = omni_kwargs.get("output_mode", "diffusion") == "ar"

        if self._ar_mode:
            return omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        return omega_conf_to_dataclass(model_config, dataclass_type=DiffusionModelConfig)

    def _validate_configs(self) -> None:
        """AR mode: derive max_model_len. Diffusion: no max_position_embeddings."""
        if self._ar_mode:
            if self.config.max_model_len is None:
                self.config.max_model_len = self.config.prompt_length + self.config.response_length
            require_raw = self._env_flag("VERL_OMNI_REQUIRE_RAW_ROLLOUT_LOGPROBS")
            require_processed = self._env_flag("VERL_OMNI_REQUIRE_PROCESSED_ROLLOUT_LOGPROBS")
            if require_raw and require_processed:
                raise ValueError(
                    "vLLM-Omni AR rollout cannot require both raw_logprobs and processed_logprobs. "
                    "Set only one of VERL_OMNI_REQUIRE_RAW_ROLLOUT_LOGPROBS or "
                    "VERL_OMNI_REQUIRE_PROCESSED_ROLLOUT_LOGPROBS."
                )
            if (
                require_raw
                and getattr(self.config, "calculate_log_probs", False)
                and getattr(self.config, "logprobs_mode", None) != "raw_logprobs"
            ):
                raise ValueError(
                    "vLLM-Omni AR rollout is configured to return rollout logprobs, "
                    "but this parity probe requires raw_logprobs so Megatron actor recomputation "
                    "and rollout logprobs use the same probability semantics. "
                    f"Got logprobs_mode={getattr(self.config, 'logprobs_mode', None)!r}; "
                    "set actor_rollout_ref.rollout.logprobs_mode=raw_logprobs."
                )
            if (
                require_processed
                and getattr(self.config, "calculate_log_probs", False)
                and getattr(self.config, "logprobs_mode", None) != "processed_logprobs"
            ):
                raise ValueError(
                    "vLLM-Omni AR rollout is configured to return rollout logprobs, "
                    "but this run requires processed_logprobs. "
                    f"Got logprobs_mode={getattr(self.config, 'logprobs_mode', None)!r}; "
                    "set actor_rollout_ref.rollout.logprobs_mode=processed_logprobs."
                )

    def _post_init(self, cuda_visible_devices: str) -> None:
        """Diffusion needs a PIL→tensor converter; AR does not."""
        if not self._ar_mode:
            self._to_tensor = T.PILToTensor()
        super()._post_init(cuda_visible_devices)

    # -----------------------------------------------------------------------
    # launch_server hooks
    # -----------------------------------------------------------------------

    def _get_override_generation_config(self) -> dict:
        """AR mode uses the parent's LLM sampling config; diffusion has none."""
        if self._ar_mode:
            return vLLMHttpServer._get_override_generation_config(self)
        return {}

    def _get_engine_kwargs_key(self) -> str:
        return "vllm_omni"

    def _get_worker_extension_cls(self) -> str:
        return "verl_omni.workers.rollout.vllm_rollout.utils.vLLMOmniColocateWorkerExtension"

    def _get_cli_modules(self) -> list:
        return [vllm_omni.entrypoints.cli.serve]

    def _get_cli_description(self) -> str:
        return "vLLM-Omni CLI"

    def _preprocess_engine_kwargs(self, engine_kwargs: dict) -> None:
        """Strip the mode selector; in AR mode also drop diffusion-only kwargs and
        normalize underscore keys vLLM-Omni expects with dashes."""
        engine_kwargs.pop("output_mode", None)
        if self._ar_mode:
            engine_kwargs.pop("custom_pipeline", None)
            stage_init_timeout = engine_kwargs.get("stage_init_timeout") or engine_kwargs.get("stage-init-timeout")
            init_timeout = engine_kwargs.get("init_timeout") or engine_kwargs.get("init-timeout")
            if stage_init_timeout is not None:
                stage_init_timeout = int(stage_init_timeout)
                os.environ.setdefault("VERL_OMNI_VLLM_STARTUP_HANDSHAKE_TIMEOUT", str(stage_init_timeout))
                if init_timeout is None:
                    engine_kwargs["init_timeout"] = max(stage_init_timeout, 600)
                    logger.info(
                        "Defaulting vLLM-Omni init_timeout to %s from stage_init_timeout",
                        engine_kwargs["init_timeout"],
                    )

            for underscore_key in (
                "stage_configs_path",
                "deploy_config",
                "stage_overrides",
                "async_chunk",
                "stage_init_timeout",
                "init_timeout",
            ):
                if underscore_key in engine_kwargs:
                    engine_kwargs[underscore_key.replace("_", "-")] = engine_kwargs.pop(underscore_key)

    # -----------------------------------------------------------------------
    # Server lifecycle
    # -----------------------------------------------------------------------

    def _configure_omni_distributed_args(self, args: argparse.Namespace, *, headless: bool) -> None:
        """Map verl's multi-node vLLM args onto vLLM-Omni single-stage launch args."""
        if self.nnodes <= 1 or not self._ar_mode:
            return

        def _int_env(name: str) -> int | None:
            raw_value = os.environ.get(name)
            if raw_value in (None, ""):
                return None
            try:
                return int(raw_value)
            except ValueError as exc:
                raise ValueError(f"{name} must be an integer port, got {raw_value!r}") from exc

        omni_master_port = _int_env("VERL_OMNI_MASTER_ZMQ_PORT")
        if omni_master_port is None:
            omni_master_port = _int_env("VERL_OMNI_MASTER_ZMQ_PORT_BASE")
        if omni_master_port is None:
            omni_master_port = self._master_port

        vllm_dist_master_port = _int_env("VERL_OMNI_VLLM_DIST_MASTER_PORT")
        if vllm_dist_master_port is not None:
            args.master_addr = self._master_address
            args.master_port = vllm_dist_master_port
            logger.info(
                "Using rollout server[0] address %s with controlled vLLM distributed port %s for vLLM-Omni AR rollout",
                self._master_address,
                vllm_dist_master_port,
            )

        args.stage_id = 0
        args.omni_master_address = self._master_address
        args.omni_master_port = omni_master_port
        args.omni_dp_size_local = 1
        args.worker_backend = "multi_process"
        args.headless = headless
        logger.info(
            "Using vLLM-Omni master server %s:%s for AR rollout registration",
            self._master_address,
            omni_master_port,
        )
        if not hasattr(args, "omni_lb_policy") or args.omni_lb_policy is None:
            args.omni_lb_policy = "random"
        if not hasattr(args, "omni_heartbeat_timeout") or args.omni_heartbeat_timeout is None:
            args.omni_heartbeat_timeout = 30.0

    def _release_omni_master_port_reservation(self) -> None:
        """Allow vLLM-Omni's master server to bind the port reserved by verl."""
        sock = getattr(self, "_master_sock", None)
        if sock is None:
            return
        sock.close()
        self._master_sock = None

    @staticmethod
    def _visible_device_count() -> int | None:
        raw_devices = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("ROCR_VISIBLE_DEVICES")
        if not raw_devices:
            return None
        return len([device.strip() for device in raw_devices.split(",") if device.strip()])

    @staticmethod
    def _stage_configs_path_from_args(args: argparse.Namespace) -> str | None:
        return getattr(args, "stage_configs_path", None)

    @staticmethod
    def _config_int(config: Any, name: str, default: int = 1) -> int:
        try:
            value = getattr(config, name)
        except (AttributeError, KeyError):
            value = config.get(name, default) if isinstance(config, dict) else default
        return int(value or default)

    def _run_ar_placement_preflight(self, args: argparse.Namespace):
        """Fail fast when verl outer DP and vLLM-Omni inner replicas are mixed."""
        stage_configs_path = self._stage_configs_path_from_args(args)
        tp_size = self._config_int(self.config, "tensor_model_parallel_size")
        dp_size = self._config_int(self.config, "data_parallel_size")
        pp_size = self._config_int(self.config, "pipeline_model_parallel_size")
        outer_replicas = estimate_outer_rollout_replicas(
            nnodes=int(getattr(self, "nnodes", 1) or 1),
            gpus_per_node=int(getattr(self, "gpus_per_node", 1) or 1),
            tensor_model_parallel_size=tp_size,
            data_parallel_size=dp_size,
            pipeline_model_parallel_size=pp_size,
        )
        preflight = validate_vllm_omni_rollout_placement(
            stage_configs_path=stage_configs_path,
            outer_replicas=outer_replicas,
            visible_device_count=self._visible_device_count(),
            allow_physical_stage_devices=self._env_flag("VERL_OMNI_ALLOW_PHYSICAL_STAGE_DEVICES"),
        )
        logger.warning(
            "vLLM-Omni rollout placement preflight: outer_replicas=%s visible_device_count=%s "
            "stage_configs_path=%s stages=%s",
            preflight.outer_replicas,
            preflight.visible_device_count,
            stage_configs_path,
            [asdict(stage) for stage in preflight.stages],
        )
        return preflight

    def _reserve_vllm_worker_master_port(self) -> tuple[int, socket.socket]:
        """Reserve a controlled localhost TCPStore port for vLLM worker TP init."""
        explicit_port = os.environ.get("VERL_OMNI_VLLM_WORKER_MASTER_PORT")
        if explicit_port:
            candidates = [int(explicit_port)]
        else:
            pool_start = int(os.environ.get("VERL_OMNI_VLLM_WORKER_MASTER_PORT_POOL_START", "65100"))
            pool_end = int(os.environ.get("VERL_OMNI_VLLM_WORKER_MASTER_PORT_POOL_END", "65535"))
            if pool_end < pool_start:
                raise ValueError(
                    "Invalid vLLM worker MASTER_PORT pool: "
                    f"start={pool_start} end={pool_end}"
                )
            pool_size = pool_end - pool_start + 1
            seed = (
                os.environ.get("VERL_OMNI_VLLM_PORT_SEED")
                or os.environ.get("LUBAN_JOB_ID")
                or os.environ.get("AIP_JOB_ID")
                or os.environ.get("VC_JOB_ID")
                or os.environ.get("JOB_ID")
                or os.environ.get("APP_ID")
                or os.environ.get("K8S_APP_ID")
                or os.environ.get("VERL_RAY_JOB_ID")
                or "verl_omni_vllm_worker_master"
            )
            actor_slot = int(os.environ.get("VERL_OMNI_VLLM_PORT_ACTOR_SLOT", "0") or 0)
            start_offset = (zlib.crc32(f"{seed}_worker_master".encode("utf-8")) + actor_slot * 17) % pool_size
            candidates = [pool_start + ((start_offset + idx) % pool_size) for idx in range(pool_size)]

        lock_root = Path(
            os.environ.get(
                "VERL_OMNI_VLLM_PORT_LOCK_DIR",
                f"/tmp/verl_omni_vllm_ports_{getpass.getuser()}",
            )
        )
        lock_root.mkdir(parents=True, exist_ok=True)
        stage_core_diag_dir = lock_root / "stage_core_diag"
        stage_core_diag_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("VERL_OMNI_STAGE_CORE_DIAG_DIR", str(stage_core_diag_dir))
        host = socket.gethostname()
        pid = os.getpid()

        for port in candidates:
            lock_path = lock_root / f"worker-master.{host}.{port}.lock"
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(("127.0.0.1", port))
                sock.listen(1)
            except OSError:
                sock.close()
                os.close(fd)
                continue

            os.ftruncate(fd, 0)
            os.write(fd, f"host={host}\npid={pid}\nport={port}\n".encode("utf-8"))
            self._vllm_worker_master_port_lock_fd = fd
            self._vllm_worker_master_port_lock_path = str(lock_path)
            os.environ["VERL_OMNI_VLLM_WORKER_MASTER_PORT_LEASE"] = str(lock_path)
            logger.warning(
                "Reserved vLLM worker MASTER_PORT lease: port=%s host=%s pid=%s lock_path=%s",
                port,
                host,
                pid,
                lock_path,
            )
            return port, sock

        if explicit_port:
            raise RuntimeError(f"Configured VERL_OMNI_VLLM_WORKER_MASTER_PORT={explicit_port} is not bindable")
        raise RuntimeError(
            "No free vLLM worker MASTER_PORT in "
            f"{os.environ.get('VERL_OMNI_VLLM_WORKER_MASTER_PORT_POOL_START', '65100')}-"
            f"{os.environ.get('VERL_OMNI_VLLM_WORKER_MASTER_PORT_POOL_END', '65535')}"
        )

    def _configure_vllm_internal_port_base(self) -> None:
        """Pin vLLM's internal ZMQ port allocator to a per-actor range.

        vLLM 0.22 uses ``VLLM_PORT`` as the starting point for internal ports.
        Without it, ports are random ephemeral values, which can collide with
        Luban host ports before vLLM has a chance to retry the ZMQ bind.
        """
        respect_existing_port = os.environ.get("VERL_OMNI_RESPECT_EXISTING_VLLM_PORT", "0") == "1"
        has_port_pool = os.environ.get("VERL_OMNI_VLLM_PORT_POOL_START") or os.environ.get(
            "VERL_OMNI_VLLM_PORT_POOL_END"
        )
        if os.environ.get("VLLM_PORT") and (respect_existing_port or not has_port_pool):
            import vllm.envs as vllm_envs

            vllm_envs.disable_envs_cache()
            self._install_vllm_port_allocator(int(os.environ["VLLM_PORT"]))
            logger.info(
                "Using existing VLLM_PORT=%s for vLLM internal ports (vllm_envs=%s)",
                os.environ["VLLM_PORT"],
                vllm_envs.VLLM_PORT,
            )
            return

        import vllm.envs as vllm_envs

        vllm_envs.disable_envs_cache()

        pool_start = int(os.environ.get("VERL_OMNI_VLLM_PORT_POOL_START", "61000"))
        pool_end = int(os.environ.get("VERL_OMNI_VLLM_PORT_POOL_END", "65000"))
        stride = int(os.environ.get("VERL_OMNI_VLLM_PORT_STRIDE", "128"))
        actor_slots = int(os.environ.get("VERL_OMNI_VLLM_PORT_ACTOR_SLOTS", "16"))
        if stride <= 0 or actor_slots <= 0 or pool_end < pool_start:
            raise ValueError(
                "Invalid vLLM internal port pool: "
                f"start={pool_start} end={pool_end} stride={stride} actor_slots={actor_slots}"
            )

        block_width = stride * actor_slots
        block_count = max(1, (pool_end - pool_start + 1) // block_width)
        seed = (
            os.environ.get("VERL_OMNI_VLLM_PORT_SEED")
            or os.environ.get("LUBAN_JOB_ID")
            or os.environ.get("AIP_JOB_ID")
            or os.environ.get("VC_JOB_ID")
            or os.environ.get("JOB_ID")
            or os.environ.get("APP_ID")
            or os.environ.get("K8S_APP_ID")
            or os.environ.get("VERL_RAY_JOB_ID")
            or "verl_omni_vllm"
        )
        ray_min_worker_port = int(os.environ.get("RAY_MIN_WORKER_PORT", "0") or 0)
        ray_max_worker_port = int(os.environ.get("RAY_MAX_WORKER_PORT", "0") or 0)
        preferred_block_index = zlib.crc32(seed.encode("utf-8")) % block_count
        block_index = preferred_block_index
        for candidate in range(block_count):
            candidate_index = (preferred_block_index + candidate) % block_count
            candidate_start = pool_start + candidate_index * block_width
            candidate_end = min(candidate_start + block_width - 1, pool_end)
            overlaps_ray_worker_range = (
                ray_min_worker_port > 0
                and ray_max_worker_port > 0
                and candidate_start <= ray_max_worker_port
                and candidate_end >= ray_min_worker_port
            )
            if not overlaps_ray_worker_range:
                block_index = candidate_index
                break
        else:
            raise ValueError(
                "No vLLM internal port block avoids the Ray worker port range: "
                f"pool={pool_start}-{pool_end} block_width={block_width} "
                f"ray_worker_range={ray_min_worker_port}-{ray_max_worker_port}"
            )
        preferred_actor_slot = int(os.environ.get("VERL_OMNI_VLLM_PORT_ACTOR_SLOT", "-1") or -1)
        if preferred_actor_slot < 0:
            preferred_actor_slot = (int(self.replica_rank) * max(int(self.nnodes), 1) + int(self.node_rank)) % actor_slots
        actor_slot = self._claim_vllm_port_actor_slot(
            seed=seed,
            block_index=block_index,
            actor_slots=actor_slots,
            preferred_slot=preferred_actor_slot,
        )
        port_base = pool_start + block_index * block_width + actor_slot * stride
        if port_base > pool_end:
            raise ValueError(
                "Derived VLLM_PORT is outside the configured pool: "
                f"port={port_base} pool={pool_start}-{pool_end}"
            )

        os.environ["VLLM_PORT"] = str(port_base)
        os.environ["VERL_OMNI_VLLM_PORT_ACTOR_SLOT"] = str(actor_slot)
        vllm_envs.disable_envs_cache()
        self._install_vllm_port_allocator(port_base)
        logger.info(
            "Using VLLM_PORT=%s for vLLM internal ports "
            "(vllm_envs=%s seed=%s pool=%s-%s stride=%s actor_slot=%s/%s replica=%s node=%s)",
            os.environ["VLLM_PORT"],
            vllm_envs.VLLM_PORT,
            seed,
            pool_start,
            pool_end,
            stride,
            actor_slot,
            actor_slots,
            self.replica_rank,
            self.node_rank,
        )

    def _claim_vllm_port_actor_slot(
        self,
        *,
        seed: str,
        block_index: int,
        actor_slots: int,
        preferred_slot: int = 0,
    ) -> int:
        """Claim a per-host vLLM port slot for this Ray actor.

        Multiple rollout actors may land on the same Luban host. Their vLLM
        engine-core workers all use localhost TCPStore ports, so deterministic
        rank-derived slots are not enough when Ray's actor rank metadata is
        absent or duplicated. Keep an advisory flock open for this actor's
        lifetime so same-host actors pick distinct slots.
        """
        existing_fd = getattr(self, "_vllm_port_slot_lock_fd", None)
        existing_slot = getattr(self, "_vllm_port_actor_slot", None)
        if existing_fd is not None and existing_slot is not None:
            return int(existing_slot)

        lock_root = Path(
            os.environ.get(
                "VERL_OMNI_VLLM_PORT_LOCK_DIR",
                f"/tmp/verl_omni_vllm_ports_{getpass.getuser()}",
            )
        )
        lock_root.mkdir(parents=True, exist_ok=True)
        safe_seed = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in seed)[:96]
        host = socket.gethostname()
        pid = os.getpid()
        preferred_slot %= actor_slots

        for offset in range(actor_slots):
            slot = (preferred_slot + offset) % actor_slots
            lock_path = lock_root / f"{safe_seed}.block{block_index}.slot{slot}.lock"
            fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue
            os.ftruncate(fd, 0)
            os.write(fd, f"host={host}\npid={pid}\nslot={slot}\n".encode("utf-8"))
            self._vllm_port_slot_lock_fd = fd
            self._vllm_port_actor_slot = slot
            return slot

        raise RuntimeError(
            "No free vLLM actor port slot on this host: "
            f"seed={seed!r} block={block_index} actor_slots={actor_slots} lock_dir={lock_root}"
        )

    def _install_vllm_port_allocator(self, port_base: int) -> None:
        """Make vLLM port allocation monotonic inside this server actor.

        vLLM's default ``VLLM_PORT`` behavior restarts the scan from the same
        base on every call. vLLM-Omni reserves some ports before they are bound,
        so later torch/ZMQ setup can reuse those promised ports and self-collide.
        """
        import vllm.utils.network_utils as network_utils

        if getattr(network_utils, "_verl_omni_monotonic_port_allocator", False):
            return

        lock = threading.Lock()
        stride = int(os.environ.get("VERL_OMNI_VLLM_PORT_STRIDE", "128"))
        stage_core_offset = int(os.environ.get("VERL_OMNI_VLLM_STAGE_CORE_PORT_OFFSET", "64"))
        if 0 < stage_core_offset < stride:
            port_end = port_base + stage_core_offset - 1
        else:
            port_end = port_base + stride - 1
        cursor = {"next": port_base}
        original_get_open_port = network_utils._get_open_port

        def get_open_port() -> int:
            with lock:
                while cursor["next"] <= port_end:
                    port = original_get_open_port(
                        start_port=cursor["next"],
                        max_attempts=port_end - cursor["next"] + 1,
                    )
                    cursor["next"] = port + 1
                    return port
                raise RuntimeError(
                    "vLLM parent port slice exhausted: "
                    f"start={port_base} end={port_end}"
                )

        def get_open_ports_list(count: int = 5) -> list[int]:
            return [get_open_port() for _ in range(count)]

        network_utils.get_open_port = get_open_port
        network_utils.get_open_ports_list = get_open_ports_list
        network_utils._verl_omni_monotonic_port_allocator = True

        patch_targets = [
            ("vllm.v1.executor.multiproc_executor", "get_open_port", get_open_port),
            ("vllm.v1.executor.uniproc_executor", "get_open_port", get_open_port),
            ("vllm.v1.executor.ray_executor", "get_open_port", get_open_port),
            ("vllm.v1.executor.ray_executor_v2", "get_open_port", get_open_port),
            ("vllm.v1.engine.utils", "get_open_port", get_open_port),
            ("vllm_omni.engine.stage_engine_startup", "get_open_ports_list", get_open_ports_list),
            ("vllm_omni.distributed.omni_coordinator.runtime", "get_open_ports_list", get_open_ports_list),
        ]
        for module_name, attr_name, replacement in patch_targets:
            try:
                module = __import__(module_name, fromlist=[attr_name])
            except Exception:
                continue
            if hasattr(module, attr_name):
                setattr(module, attr_name, replacement)

        logger.info("Installed monotonic vLLM port allocator starting at %s ending at %s", port_base, port_end)

    def _ensure_tracking_namespace(self, args: argparse.Namespace) -> argparse.Namespace:
        if hasattr(args, "get_explicit_kwargs_dict"):
            return args
        return TrackingNamespace(
            unfiltered_ns=args,
            explicit_keys=frozenset(vars(args).keys()),
        )

    async def run_server(self, args: argparse.Namespace):
        args = self._ensure_tracking_namespace(args)
        self._configure_vllm_internal_port_base()
        self._configure_omni_distributed_args(args, headless=False)
        if self.nnodes > 1 and self._ar_mode:
            self._release_omni_master_port_reservation()
        engine_args = OmniEngineArgs.from_cli_args(args)
        engine_args = asdict(engine_args)

        if self._ar_mode:
            for timeout_key in ("stage_init_timeout", "init_timeout"):
                timeout_value = getattr(args, timeout_key, None)
                if timeout_value is not None:
                    engine_args[timeout_key] = int(timeout_value)
            placement_preflight = self._run_ar_placement_preflight(args)
            if placement_preflight.max_stage_replicas <= 1:
                os.environ.setdefault("VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE", "1")
            logger.info(
                "vLLM-Omni AR startup timeouts: stage_init_timeout=%s init_timeout=%s",
                engine_args.get("stage_init_timeout"),
                engine_args.get("init_timeout"),
            )
            # AR mode: no diffusion pipeline. Drop None entries from
            # compilation_config that OmniEngineArgs may leave behind.
            engine_args["logprobs_mode"] = getattr(self.config, "logprobs_mode", "processed_logprobs")
            logger.info("vLLM-Omni AR rollout logprobs_mode=%s", engine_args["logprobs_mode"])
            if isinstance(engine_args.get("compilation_config"), dict):
                engine_args["compilation_config"] = {
                    k: v for k, v in engine_args["compilation_config"].items() if v is not None
                }
        else:
            # inject multi-stage yaml config
            deploy_config = getattr(args, "deploy_config", None)
            if deploy_config:
                engine_args["deploy_config"] = deploy_config

            import_external_libs(self.config.external_lib)

            self.config.resolve_algorithm(self.model_config)

            pipeline_path = VllmOmniPipelineBase.get_pipeline_path(
                architecture=self.model_config.architecture,
                algorithm=self.model_config.algorithm,
            )
            # TODO (mike): read custom_pipeline from engine_args
            if pipeline_path is not None:
                engine_args["enable_dummy_pipeline"] = True
                engine_args["custom_pipeline_args"] = {"pipeline_class": pipeline_path}

        if getattr(self.config, "step_execution", False):
            engine_args["step_execution"] = True

        worker_master_port, worker_master_sock = self._reserve_vllm_worker_master_port()

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(worker_master_port)
        logger.warning(
            "Using controlled MASTER_PORT=%s for vLLM-Omni workers lease=%s use_for_stage_core=%s",
            os.environ["MASTER_PORT"],
            os.environ.get("VERL_OMNI_VLLM_WORKER_MASTER_PORT_LEASE"),
            os.environ.get("VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE"),
        )
        worker_master_sock.close()

        engine_client = AsyncOmni(**engine_args)
        app = build_app(args)
        await omni_init_app_state(engine_client, app.state, args)

        self.engine = engine_client
        self._server_port, self._server_task = await run_uvicorn(app, args, self._server_address)

    async def run_headless(self, args: argparse.Namespace):
        """Run headless server in a separate thread."""
        args = self._ensure_tracking_namespace(args)
        self._configure_vllm_internal_port_base()
        self._configure_omni_distributed_args(args, headless=True)
        if self._ar_mode:
            placement_preflight = self._run_ar_placement_preflight(args)
            if placement_preflight.max_stage_replicas <= 1:
                os.environ.setdefault("VERL_OMNI_USE_MASTER_PORT_FOR_STAGE_CORE_TCPSTORE", "1")
        args.api_server_count = 0

        def run_headless_wrapper():
            with SuppressSignalInThread():
                run_omni_headless(args)

        def on_run_headless_done(future: asyncio.Future):
            try:
                exc = future.exception()
                if exc:
                    logger.exception("vLLM-Omni run_headless failed with exception: %s", exc)
                else:
                    logger.warning("vLLM-Omni run_headless completed successfully, but it's not expected.")
            except Exception as e:
                logger.exception("get result from vLLM-Omni run_headless failed: %s", e)
            finally:
                os._exit(1)

        self.task = asyncio.create_task(asyncio.to_thread(run_headless_wrapper))
        self.task.add_done_callback(on_run_headless_done)

    # -----------------------------------------------------------------------
    # wake_up hook: Omni does not restore KV cache on wake-up
    # -----------------------------------------------------------------------

    def _get_wake_up_tags(self) -> list[str]:
        return ["weights"]

    async def wake_up(self, tags: list[str] | None = None):
        """Override parent to use collective_rpc instead of engine.wake_up().

        The parent (verl ``1927ad33``+) calls ``self.engine.wake_up(tags=...)``
        which triggers CUDA initialisation in this HTTP server process when
        running under vLLM-Omni (AsyncOmni engine).
        Use ``collective_rpc`` instead.

        # TODO (long): drop this override once vllm-omni wake_up
        without triggering GPU initialisation.
        """
        if self.node_rank != 0:
            return
        await self.engine.collective_rpc(
            "wake_up", kwargs={"tags": tags if tags is not None else self._get_wake_up_tags()}
        )

    async def _sleep_hybrid(self):
        """Preserve non-actor pipeline weights during hybrid training sleep.

        vLLM-Omni diffusion pipelines include components such as the text
        encoder and VAE that are loaded by the rollout server, but are not part
        of the trainable actor and therefore are not included in full-model
        weight syncs. Use level-1 sleep so those weights are offloaded and can
        be restored on wake-up instead of discarded by level-2 sleep.
        """
        # TODO (andy): use `sleep_level=2` in the future when the
        #  trainer side incorporates the whole components of the model.
        await self.engine.collective_rpc("sleep", kwargs={"level": 1})
        await self.engine.reset_encoder_cache()

    # -----------------------------------------------------------------------
    # abort hooks: AsyncOmni owns request state directly
    # -----------------------------------------------------------------------

    async def abort_all_requests(self, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort in-flight AsyncOmni requests without vLLM's output_processor."""
        if self.node_rank != 0:
            return {"aborted_count": 0, "request_ids": []}

        try:
            request_states = getattr(self.engine, "request_states", {})
            request_ids = list(request_states.keys())
            if request_ids:
                await self.engine._abort_internal_requests(request_ids)
            if reset_prefix_cache:
                await self.engine.reset_prefix_cache()

            logger.info("Aborted %s vLLM-Omni request(s): %s", len(request_ids), request_ids)
            return {"aborted_count": len(request_ids), "request_ids": request_ids}
        except Exception as e:
            logger.error("Error aborting vLLM-Omni requests: %s", e)
            return {"aborted_count": 0, "request_ids": [], "error": str(e)}

    async def abort_request(self, request_id: str, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort one AsyncOmni request by internal or external request id."""
        if self.node_rank != 0:
            return {"aborted": False, "request_id": request_id}

        try:
            request_states = getattr(self.engine, "request_states", {})
            if request_id in request_states:
                await self.engine._abort_internal_requests([request_id])
                aborted = True
            else:
                external_matches = [
                    rid
                    for rid, state in request_states.items()
                    if getattr(state, "external_request_id", None) == request_id
                ]
                if external_matches:
                    await self.engine.abort(request_id)
                    aborted = True
                else:
                    return {"aborted": False, "request_id": request_id, "error": f"Request {request_id} not found"}

            if reset_prefix_cache:
                await self.engine.reset_prefix_cache()

            logger.info("Aborted vLLM-Omni request: %s", request_id)
            return {"aborted": aborted, "request_id": request_id}
        except Exception as e:
            logger.error("Error aborting vLLM-Omni request %s: %s", request_id, e)
            return {"aborted": False, "request_id": request_id, "error": str(e)}

    async def resume_generation(self):
        """AsyncOmni aborts requests directly and does not pause the engine."""
        return

    # -----------------------------------------------------------------------
    # generate: shared pipeline; mode-specific steps branch on self._ar_mode
    # (_preprocess_input / _run_generation / _process_output).
    # -----------------------------------------------------------------------

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        negative_prompt_ids: Optional[list[int]] = None,
        prompt_mask: torch.BoolTensor | None = None,
        priority: int = 0,
    ) -> DiffusionOutput | TokenOutput:
        prompt_ids = normalize_token_ids(prompt_ids)
        self._validate_generate_multimodal_args(
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        multi_modal_data = self._build_multi_modal_data(image_data, video_data)
        lora_request = await self._resolve_lora_request()
        prompt, params = self._preprocess_input(
            prompt_ids, sampling_params, multi_modal_data, lora_request, negative_prompt_ids, prompt_mask
        )
        final_res = await self._run_generation(prompt, params, request_id, lora_request, priority)
        return self._process_output(final_res, params, sampling_params)

    # -----------------------------------------------------------------------
    # Shared helpers for the AR and diffusion generate paths
    # -----------------------------------------------------------------------

    def _validate_generate_multimodal_args(
        self,
        *,
        image_data: Optional[list[Any]],
        video_data: Optional[list[Any]],
        audio_data: Optional[list[Any]],
        mm_processor_kwargs: Optional[dict[str, Any]],
    ) -> None:
        """Fail clearly for multimodal inputs that are not wired for this mode."""
        if self._ar_mode:
            provided = []
            if image_data is not None:
                provided.append("image_data")
            if video_data is not None:
                provided.append("video_data")
            if audio_data is not None:
                provided.append("audio_data")
            if mm_processor_kwargs:
                provided.append("mm_processor_kwargs")
            if provided:
                raise NotImplementedError(
                    "vLLM-Omni AR text rollout currently supports token-only prompts; "
                    f"got unsupported multimodal args: {', '.join(provided)}"
                )
            return

        unsupported = []
        if audio_data is not None:
            unsupported.append("audio_data")
        if mm_processor_kwargs:
            unsupported.append("mm_processor_kwargs")
        if unsupported:
            raise NotImplementedError(
                "vLLM-Omni diffusion rollout does not currently wire these verl multimodal args: "
                f"{', '.join(unsupported)}"
            )

    @staticmethod
    def _build_multi_modal_data(image_data: Optional[list[Any]], video_data: Optional[list[Any]]) -> dict[str, Any]:
        """Assemble the vLLM multi_modal_data dict from optional image/video inputs."""
        multi_modal_data: dict[str, Any] = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data
        return multi_modal_data

    async def _resolve_lora_request(self) -> Optional[LoRARequest]:
        """Build the actor LoRA request if a LoRA adapter is currently loaded.

        Wraps ``list_loras`` in ``try/except TypeError`` (a strict superset of the
        plain membership check): some engine backends return a non-iterable, in
        which case we assume the adapter is loaded. The diffusion path is unchanged
        in the normal (iterable) case.
        """
        if not self.lora_as_adapter:
            return None
        try:
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
        except TypeError:
            lora_loaded = True
        if not lora_loaded:
            return None
        return LoRARequest(lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH)

    @staticmethod
    def _map_stop_reason(finish_reason: Optional[str]) -> Optional[str]:
        """Map a vLLM finish_reason to verl's stop_reason vocabulary."""
        if finish_reason == "abort":
            return "aborted"
        if finish_reason in ("stop", "length"):
            return "completed"
        return finish_reason

    @staticmethod
    def _env_flag(name: str) -> bool:
        return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _logprob_entry_summary(entry: Any) -> dict[str, Any]:
        if entry is None:
            return {"missing": True}
        return {
            "logprob": round(float(getattr(entry, "logprob", float("nan"))), 6),
            "rank": getattr(entry, "rank", None),
            "decoded_token": getattr(entry, "decoded_token", None),
        }

    def _maybe_log_ar_logprob_stats(
        self,
        token_ids: list[int],
        log_probs: list[float],
        output_logprobs: Optional[list[dict[Any, Any]]] = None,
    ) -> None:
        raw_limit = os.environ.get("VERL_OMNI_LOGPROB_DEBUG_LIMIT", "0")
        try:
            limit = int(raw_limit)
        except ValueError:
            return
        if limit <= 0 or not log_probs:
            return

        values = [float(value) for value in log_probs]
        sample_count = min(limit, len(values), len(token_ids))
        sample = [(int(token_ids[i]), round(values[i], 6)) for i in range(sample_count)]
        zero_token_sample = [int(token_ids[i]) for i, value in enumerate(values[: len(token_ids)]) if value == 0.0][
            :limit
        ]
        debug_payload = (
            f"vLLM-Omni AR logprob debug: mode={getattr(self.config, 'logprobs_mode', None)} "
            f"count={len(values)} zero={sum(1 for value in values if value == 0.0)} "
            f"min={min(values):.6f} mean={sum(values) / len(values):.6f} max={max(values):.6f} "
            f"sample={sample} zero_token_sample={zero_token_sample}"
        )
        logger.warning(debug_payload)
        print(debug_payload, flush=True)
        row_summaries = []

        if output_logprobs is not None:
            for i, token_logprobs in enumerate(output_logprobs[:sample_count]):
                token_id = int(token_ids[i])
                has_token = token_id in token_logprobs
                selected = token_logprobs.get(token_id)
                first_items = list(token_logprobs.items())[: min(4, len(token_logprobs))]
                max_item = None
                if token_logprobs:
                    max_item = max(
                        token_logprobs.items(),
                        key=lambda item: float(getattr(item[1], "logprob", float("-inf"))),
                    )
                row_summaries.append(
                    {
                        "i": i,
                        "token_id": token_id,
                        "row_len": len(token_logprobs),
                        "key_type": type(first_items[0][0]).__name__ if first_items else None,
                        "has_token": has_token,
                        "selected": self._logprob_entry_summary(selected),
                        "first_items": [
                            (int(key) if isinstance(key, int) else key, self._logprob_entry_summary(value))
                            for key, value in first_items
                        ],
                        "max_item": None
                        if max_item is None
                        else (
                            int(max_item[0]) if isinstance(max_item[0], int) else max_item[0],
                            self._logprob_entry_summary(max_item[1]),
                        ),
                    }
                )
            rows_payload = f"vLLM-Omni AR logprob rows: {row_summaries}"
            logger.warning(rows_payload)
            print(rows_payload, flush=True)

        _append_vllm_logprob_debug_jsonl(
            "verl_omni_extraction",
            {
                "mode": getattr(self.config, "logprobs_mode", None),
                "count": len(values),
                "zero_count": sum(1 for value in values if value == 0.0),
                "min": min(values),
                "mean": sum(values) / len(values),
                "max": max(values),
                "sample": sample,
                "zero_token_sample": zero_token_sample,
                "rows": row_summaries,
            },
        )

    # -----------------------------------------------------------------------
    # Mode-specific pipeline steps
    # -----------------------------------------------------------------------

    def _preprocess_input(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        multi_modal_data: dict[str, Any],
        lora_request: Optional[LoRARequest],
        negative_prompt_ids: Optional[list[int]],
        prompt_mask: torch.BoolTensor | None = None,
    ):
        """Build the engine prompt + sampling params for the active mode.

        Returns ``(prompt, params)`` consumed by ``_run_generation``.
        """
        if self._ar_mode:
            max_possible_tokens = self.config.max_model_len - len(prompt_ids)
            if max_possible_tokens <= 0:
                raise ValueError(
                    f"Prompt length ({len(prompt_ids)}) meets or exceeds the model's maximum context length "
                    f"({self.config.max_model_len}), leaving no space for generation."
                )

            if "max_tokens" in sampling_params:
                max_tokens = sampling_params.pop("max_tokens")
            elif "max_new_tokens" in sampling_params:
                max_tokens = sampling_params.pop("max_new_tokens")
            else:
                max_tokens = min(
                    self.config.response_length,
                    self.config.prompt_length + self.config.response_length - len(prompt_ids),
                )
            max_tokens = max(0, min(max_tokens, max_possible_tokens))

            # Match verl native vLLM async and NeMo RL: request sampled-token
            # logprobs only. vLLM returns the sampled token even when logprobs=0.
            # preserve explicit int counts (incl. 0), fall back to None otherwise.
            logprobs = sampling_params.pop("logprobs", None)
            if logprobs is True:
                sampling_params["logprobs"] = 0
            elif isinstance(logprobs, int) and not isinstance(logprobs, bool):
                sampling_params["logprobs"] = logprobs
            else:
                sampling_params["logprobs"] = None
            sampling_params.setdefault("repetition_penalty", getattr(self.config, "repetition_penalty", 1.0))
            params = SamplingParams(max_tokens=max_tokens, **sampling_params)

            prompt = {"prompt_token_ids": prompt_ids}
            if multi_modal_data:
                prompt["multi_modal_data"] = multi_modal_data
            return prompt, params

        # diffusion
        default_params_list = self.engine.default_sampling_params_list

        custom_prompt: OmniCustomPrompt = {"prompt_token_ids": prompt_ids}
        if prompt_mask is not None:
            custom_prompt["prompt_mask"] = prompt_mask
        if len(default_params_list) > 1:
            # Multi-stage pipelines tag the diffusion stage so the orchestrator can route inputs correctly.
            custom_prompt["modalities"] = ["image"]
        if negative_prompt_ids is not None:
            custom_prompt["negative_prompt_ids"] = negative_prompt_ids
        if multi_modal_data:
            custom_prompt["multi_modal_data"] = multi_modal_data
            custom_prompt["extra_args"] = {"multi_modal_data": multi_modal_data}

        sampling_kwargs: dict[str, Any] = {}
        extra_args: dict[str, Any] = {}
        for k, v in sampling_params.items():
            if hasattr(OmniDiffusionSamplingParams, k):
                sampling_kwargs[k] = v
            else:
                extra_args[k] = v
        sampling_kwargs["extra_args"] = extra_args
        if lora_request is not None:
            sampling_kwargs["lora_request"] = lora_request
        diffusion_sampling_params = OmniDiffusionSamplingParams(**sampling_kwargs)
        # Multi-stage models use defaults for non-diffusion stages.
        params = default_params_list[:-1] + [diffusion_sampling_params]
        return custom_prompt, params

    async def _run_generation(self, prompt, params, request_id: str, lora_request, priority: int):
        """Drive the engine and return the final OmniRequestOutput."""
        if self._ar_mode:
            generator = self.engine.generate(
                prompt=prompt,
                sampling_params=params,
                request_id=request_id,
                lora_request=lora_request,
                priority=priority,
            )
        else:
            generator = self.engine.generate(
                prompt=prompt,
                request_id=request_id,
                sampling_params_list=params,
            )
        final_res: Optional[OmniRequestOutput] = None
        async for output in generator:
            final_res = output
        return final_res

    def _process_output(self, final_res, params, sampling_params: dict[str, Any]):
        """Convert the engine output into the active mode's verl output dataclass."""
        if self._ar_mode:
            if final_res is None:
                raise RuntimeError("AR mode: vLLM-Omni engine yielded no output for the prompt.")

            req_output = final_res.request_output
            if req_output is None:
                raise RuntimeError("AR mode expects request_output with token IDs, but got None.")

            extra_fields = {"global_steps": self.global_steps}
            token_ids = req_output.outputs[0].token_ids
            log_probs = None
            if params.logprobs is not None:
                output_logprobs = req_output.outputs[0].logprobs
                if output_logprobs is None:
                    raise RuntimeError("AR mode requested logprobs, but vLLM-Omni returned None.")
                if len(output_logprobs) < len(token_ids):
                    raise RuntimeError(
                        "AR mode received fewer logprob rows than generated tokens: "
                        f"logprobs={len(output_logprobs)} tokens={len(token_ids)}"
                    )
                log_probs = []
                for i, token_logprobs in enumerate(output_logprobs[: len(token_ids)]):
                    token_id = token_ids[i]
                    if token_id not in token_logprobs:
                        raise RuntimeError(
                            "AR mode sampled-token logprob is missing from vLLM-Omni output: "
                            f"index={i} token_id={token_id}"
                        )
                    log_probs.append(token_logprobs[token_id].logprob)
                self._maybe_log_ar_logprob_stats(token_ids, log_probs, output_logprobs)

            finish_reason = req_output.outputs[0].finish_reason
            stop_reason = self._map_stop_reason(finish_reason)

            num_preempted = None
            if hasattr(req_output.outputs[0], "num_preempted"):
                num_preempted = req_output.outputs[0].num_preempted

            return TokenOutput(
                token_ids=token_ids,
                log_probs=log_probs,
                stop_reason=stop_reason,
                num_preempted=num_preempted,
                extra_fields=extra_fields,
            )

        # diffusion
        assert final_res is not None
        diffusion_output = final_res.images[0]
        if isinstance(diffusion_output, torch.Tensor):
            diffusion_output = diffusion_output.float()
        elif isinstance(diffusion_output, np.ndarray):
            diffusion_output = torch.from_numpy(diffusion_output).float()
        else:
            diffusion_output = self._to_tensor(diffusion_output).float() / 255.0

        # Extract extra data from custom_output (populated by DiffusionEngine)
        mm_output = final_res.custom_output or {}

        if sampling_params.get("logprobs", False):
            all_log_probs = mm_output.get("all_log_probs")
            log_probs = all_log_probs[0] if all_log_probs is not None else None
        else:
            log_probs = None

        def _maybe_unbatch(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value[0] if value.dim() > 0 else value
            if isinstance(value, np.ndarray):
                return value[0] if value.ndim > 0 else value
            if isinstance(value, list | tuple):
                return value[0] if value else None
            return value

        extra_fields = {k: _maybe_unbatch(v) for k, v in mm_output.items() if k != "all_log_probs"}
        extra_fields["global_steps"] = self.global_steps

        if final_res.request_output is not None and hasattr(final_res.request_output, "finish_reason"):
            finish_reason = final_res.request_output.finish_reason or "stop"
        else:
            finish_reason = "stop"

        stop_reason = self._map_stop_reason(finish_reason)

        num_preempted = None
        if final_res.request_output is not None and hasattr(final_res.request_output, "num_preempted"):
            num_preempted = final_res.request_output.num_preempted

        return DiffusionOutput(
            diffusion_output=diffusion_output,
            log_probs=log_probs,
            stop_reason=stop_reason,
            num_preempted=num_preempted,
            extra_fields=extra_fields,
        )

    async def wait_for_requests_to_drain(self):
        # TODO (mike): implement this once DP is supported.
        pass


class vLLMOmniReplica(vLLMReplica):
    def __init__(
        self,
        replica_rank: int,
        config: DiffusionRolloutConfig | RolloutConfig,
        model_config: DiffusionModelConfig | HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = ray.remote(vLLMOmniHttpServer)

    def _validate_launch_requirements(self) -> None:
        """No-op: the parent check validates vllm.__version__ which is
        irrelevant for vllm-omni (a separate package)."""
        pass

    def _get_server_name_prefix(self) -> str:
        return "vllm_omni_"
