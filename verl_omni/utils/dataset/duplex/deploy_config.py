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
"""Duplex deploy-config generation for the entry-gate scripts.

The calibration audit and the M0 parity check drive standalone ``DuplexOmni``
engines: unlike the training rollout — whose AR strategy writes the deploy
config from the adapter's stage topology — the scripts must hand the engine
client a thinker-only duplex deploy config themselves, or the engine boots
the pipeline in turn mode and every session open fails. One writer, two
scripts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

#: A cold duplex stage needs far more than the engine's 300 s init default.
STAGE_INIT_TIMEOUT_DEFAULT = 1800
#: Environment variable that overrides ``STAGE_INIT_TIMEOUT_DEFAULT`` in the scripts.
STAGE_INIT_TIMEOUT_ENV = "VERL_OMNI_DUPLEX_STAGE_INIT_TIMEOUT"

__all__ = [
    "STAGE_INIT_TIMEOUT_DEFAULT",
    "STAGE_INIT_TIMEOUT_ENV",
    "stage_init_timeout_from_env",
    "write_deploy_config_for",
    "write_duplex_deploy_config",
]


def stage_init_timeout_from_env() -> int:
    """The scripts' default per-stage init timeout; ``STAGE_INIT_TIMEOUT_ENV`` overrides it."""
    return int(os.environ.get(STAGE_INIT_TIMEOUT_ENV, STAGE_INIT_TIMEOUT_DEFAULT))


def write_deploy_config_for(
    *,
    output_path: str | Path,
    deploy_config: str | Path | None,
    tensor_parallel_size: int = 1,
) -> str:
    """Write the deploy config at ``deploy_config``, or next to ``output_path`` when unset.

    Args:
        output_path: The script's ``--output`` report; the default config lands
            beside it as ``<stem>.deploy.yaml``.
        deploy_config: An explicit ``--deploy-config`` path, or ``None``.
        tensor_parallel_size: Per-stage TP for the single duplex Thinker stage.

    Returns:
        The written path as a string.
    """
    target = Path(deploy_config) if deploy_config else Path(output_path).with_suffix(".deploy.yaml")
    return write_duplex_deploy_config(target, tensor_parallel_size=tensor_parallel_size)


def write_duplex_deploy_config(path: str | Path, *, tensor_parallel_size: int = 1) -> str:
    """Write the thinker-only duplex deploy config YAML and return its path.

    Args:
        path: Where to write the YAML (parent directories are created).
        tensor_parallel_size: Per-stage TP for the single duplex Thinker
            stage. TP=1 is the only validated standalone layout; a bare
            top-level TP>1 is rejected by the multi-stage engine.

    Returns:
        The written path as a string.
    """
    import yaml

    from verl_omni.pipelines.minicpm.duplex_rollout_adapter import MiniCPMODuplexRolloutAdapter

    adapter = MiniCPMODuplexRolloutAdapter
    adapter.ensure_pipeline_registered(pipeline_mode="duplex")
    stages = adapter.build_stage_configs(pipeline_mode="duplex")
    deploy: dict[str, Any] = {
        "pipeline": adapter.get_pipeline_id(pipeline_mode="duplex"),
        **adapter.deploy_config_overrides(pipeline_mode="duplex"),
        "stages": [
            {
                "stage_id": stage.stage_id,
                "tensor_parallel_size": int(tensor_parallel_size),
                "engine_extras": dict(adapter.get_stage_engine_extras(stage.stage_id, pipeline_mode="duplex")),
            }
            for stage in stages
        ],
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(deploy, sort_keys=False), encoding="utf-8")
    return str(target)
