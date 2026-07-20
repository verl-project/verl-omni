# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import os
import re
import socket
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu

SCRIPT = (
    Path(__file__).resolve().parent
    / "run_gspo_qwen3_omni_megatron_full_32gpu_fully_async.sh"
)
CONDA_ENV = Path("/nfs/ml-training-ssd/users/liuwei/verl_mega_async")


def _make_preflight_inputs(tmp_path: Path) -> dict[str, str]:
    model_path = tmp_path / "model"
    model_path.mkdir(exist_ok=True)
    train_file = tmp_path / "train.parquet"
    val_file = tmp_path / "test.parquet"
    train_file.write_bytes(b"")
    val_file.write_bytes(b"")
    stage_config = tmp_path / "stage.yaml"
    stage_config.write_text(
        "\n".join(
            [
                "tensor_parallel_size: 4",
                'devices: "0,1,2,3"',
                "max_model_len: 1024",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "MODEL_PATH": str(model_path),
        "TRAIN_FILES": str(train_file),
        "VAL_FILES": str(val_file),
        "STAGE_CONFIG": str(stage_config),
        "OUTPUT_ROOT": str(tmp_path / "outputs"),
    }


def _run_config_only(tmp_path: Path, **env_overrides: str) -> str:
    if not (CONDA_ENV / "bin" / "activate").exists():
        pytest.skip(f"CONDA_ENV is not available: {CONDA_ENV}")

    env = os.environ.copy()
    env.update(_make_preflight_inputs(tmp_path))
    env.update(
        {
            "CONDA_ENV": str(CONDA_ENV),
            "CONFIG_ONLY": "1",
            "RUN_ID": "port_contract_unit",
            "RAY_PORT_SEED": "port_contract_unit",
            "RAY_WORKER_PORT_SEED": "port_contract_unit",
        }
    )
    env.update(env_overrides)

    proc = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=SCRIPT.parents[2],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
        timeout=60,
    )
    return proc.stdout


def _run_config_only_failure(tmp_path: Path, **env_overrides: str) -> str:
    if not (CONDA_ENV / "bin" / "activate").exists():
        pytest.skip(f"CONDA_ENV is not available: {CONDA_ENV}")

    env = os.environ.copy()
    env.update(_make_preflight_inputs(tmp_path))
    env.update(
        {
            "CONDA_ENV": str(CONDA_ENV),
            "CONFIG_ONLY": "1",
            "RUN_ID": "port_contract_unit",
            "RAY_PORT_SEED": "port_contract_unit",
            "RAY_WORKER_PORT_SEED": "port_contract_unit",
        }
    )
    env.update(env_overrides)

    proc = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=SCRIPT.parents[2],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=60,
    )
    assert proc.returncode != 0, proc.stdout
    return proc.stdout


def _ray_ports_line(output: str) -> str:
    match = re.search(r"^\[info\] Ray ports: .*$", output, flags=re.MULTILINE)
    assert match, output
    return match.group(0)


def _worker_range(output: str) -> tuple[int, int]:
    ports_line = _ray_ports_line(output)
    match = re.search(r"worker_range=(\d+)-(\d+)", ports_line)
    assert match, ports_line
    return int(match.group(1)), int(match.group(2))


def _port_block_free(start: int, span: int) -> bool:
    sockets: list[socket.socket] = []
    try:
        for port in range(start, start + span):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("", port))
            sockets.append(sock)
    except OSError:
        return False
    finally:
        for sock in sockets:
            sock.close()
    return True


def _find_free_port_pool(span: int, blocks: int) -> tuple[int, int]:
    for start in range(20000, 60000 - span * blocks, span * blocks):
        if all(_port_block_free(start + span * block, span) for block in range(blocks)):
            return start, start + span * blocks - 1
    pytest.skip("No free local port pool available for Ray worker port contract test")


def test_config_only_uses_hash_derived_ray_ports_without_dashboard(tmp_path: Path):
    output = _run_config_only(tmp_path)

    ports_line = _ray_ports_line(output)
    assert "dashboard_enabled=0" in ports_line
    assert "dashboard=0" in ports_line
    assert "head_pool=20000-29999" in ports_line
    assert "worker_pool=30000-59999" in ports_line
    assert "worker_span=256" in ports_line
    assert "worker_min_free=256" in ports_line
    assert (
        "vLLM internal port pool: seed=port_contract_unit pool=61000-65099 stride=512 "
        "stage_core_offset=128 stage_core_guard=64 stage_core_spread=128 "
        "stage_core_min_tail=64 stage_core_direct_gap=32 actor_slots=4"
    ) in output
    assert "CONFIG_ONLY=1, static configuration preflight passed" in output
    assert "rollout tp=4 dp=1 replicas=4" in output
    assert "vLLM distributed rendezvous inactive: rollout_dp=1; using standalone TP4 replicas=4" in output
    assert "vLLM existing VLLM_PORT policy: respect_existing=0" in output
    assert "vLLM stage-core TCPStore first-port override: use_master_port=1" in output
    assert "VERL_OMNI_VLLM_DIST_MASTER_ADDR" not in output

    head = int(re.search(r"head=(\d+)", ports_line).group(1))
    assert 20000 <= head <= 29999


def test_resource_preflight_queries_gcs_without_starting_a_ray_driver():
    script = SCRIPT.read_text(encoding="utf-8")

    preflight = script[script.index("wait_for_ray_resources() {") :]
    assert "from ray._raylet import GcsClient" in preflight
    assert "get_all_node_info(timeout=5)" in preflight
    assert "ray.init(" not in preflight


def test_launcher_forwards_wrapper_hydra_overrides_to_fully_async_main():
    script = SCRIPT.read_text(encoding="utf-8")
    launch = script[script.index("python3 -m verl.experimental.fully_async_policy.fully_async_main ") :]

    assert '    "$@"\npython_rc=$?' in launch
    ld_library_line = next(line for line in launch.splitlines() if "env_vars.LD_LIBRARY_PATH" in line)
    assert ld_library_line.endswith("\\")
    assert not ld_library_line.endswith("\\\\")


def test_config_only_rejects_vllm_omni_internal_dp(tmp_path: Path):
    output = _run_config_only_failure(tmp_path, ROLLOUT_DP="4")

    assert "ROLLOUT_DP=4 is not supported for this vLLM-Omni AR async launcher" in output
    assert "ROLLOUT_DP=1" in output


def test_config_only_rejects_too_narrow_vllm_stage_core_stride(tmp_path: Path):
    output = _run_config_only_failure(tmp_path, VERL_OMNI_VLLM_PORT_STRIDE="128")

    assert "leaves too little stage-core tail" in output


def test_config_only_dashboard_opt_in_uses_separate_dashboard_pool(tmp_path: Path):
    output = _run_config_only(tmp_path, RAY_INCLUDE_DASHBOARD="1")

    ports_line = _ray_ports_line(output)
    assert "dashboard_enabled=1" in ports_line
    dashboard = int(re.search(r"dashboard=(\d+)", ports_line).group(1))
    assert 10000 <= dashboard <= 19999


def test_config_only_respects_explicit_ray_port_override(tmp_path: Path):
    output = _run_config_only(tmp_path, RAY_PORT="12345")

    ports_line = _ray_ports_line(output)
    assert "head=12345" in ports_line
    assert "RAY_ADDRESS=" in output
    assert ":12345" in output


def test_config_only_shifts_ray_worker_block_when_preferred_block_is_busy(tmp_path: Path):
    span = 8
    pool_start, pool_end = _find_free_port_pool(span=span, blocks=2)
    env = {
        "RAY_WORKER_PORT_POOL_START": str(pool_start),
        "RAY_WORKER_PORT_POOL_END": str(pool_end),
        "RAY_WORKER_PORT_SPAN": str(span),
        "RAY_WORKER_PORT_MIN_FREE": str(span),
        "RAY_WORKER_PORT_LOCAL_PROBE": "1",
        "RAY_WORKER_PORT_SEED": "ray_worker_busy_block_unit",
        "RAY_WORKER_PORT_LOCK_DIR": str(tmp_path / "ray_worker_port_locks"),
        "RAY_PORT": "12345",
    }
    first_output = _run_config_only(tmp_path, **env)
    first_start, first_end = _worker_range(first_output)

    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("", first_start))
    blocker.listen(1)
    try:
        second_output = _run_config_only(tmp_path, **env)
    finally:
        blocker.close()

    second_start, second_end = _worker_range(second_output)
    assert (second_start, second_end) != (first_start, first_end)
    assert second_start >= pool_start
    assert second_end <= pool_end
    assert f"worker_preferred_block=" in second_output
    assert f"worker_local_probe=1" in second_output


def test_config_only_keeps_ray_worker_block_when_capacity_is_sufficient(tmp_path: Path):
    span = 8
    pool_start, pool_end = _find_free_port_pool(span=span, blocks=2)
    env = {
        "RAY_WORKER_PORT_POOL_START": str(pool_start),
        "RAY_WORKER_PORT_POOL_END": str(pool_end),
        "RAY_WORKER_PORT_SPAN": str(span),
        "RAY_WORKER_PORT_MIN_FREE": str(span - 1),
        "RAY_WORKER_PORT_LOCAL_PROBE": "1",
        "RAY_WORKER_PORT_SEED": "ray_worker_capacity_unit",
        "RAY_WORKER_PORT_LOCK_DIR": str(tmp_path / "ray_worker_port_locks"),
        "RAY_PORT": "12345",
    }
    first_output = _run_config_only(tmp_path, **env)
    first_start, first_end = _worker_range(first_output)

    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("", first_start))
    blocker.listen(1)
    try:
        second_output = _run_config_only(tmp_path, **env)
    finally:
        blocker.close()

    second_start, second_end = _worker_range(second_output)
    assert (second_start, second_end) == (first_start, first_end)
    assert f"worker_min_free={span - 1}" in second_output
