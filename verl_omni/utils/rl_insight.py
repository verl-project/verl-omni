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
"""RL-Insight setup shared by verl-omni training entrypoints."""

import logging
import os

import ray

logger = logging.getLogger(__name__)


def enable_rl_insight(config) -> None:
    """Enable RL-Insight in this process when selected as a trainer logger."""
    trainer_logger = config.trainer.get("logger", [])
    configured_loggers = [trainer_logger] if isinstance(trainer_logger, str) else trainer_logger or []
    if "rl_insight" in configured_loggers:
        os.environ["VERL_RL_INSIGHT_ENABLE"] = "1"
        if ray.is_initialized():
            logger.warning(
                "RL-Insight was enabled after Ray initialization; existing workers may not receive "
                "VERL_RL_INSIGHT_ENABLE. Set it before ray.init() or restart Ray."
            )
